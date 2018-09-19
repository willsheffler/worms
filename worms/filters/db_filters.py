from pyrosetta.rosetta.utility import vector1_unsigned_long
from pyrosetta.rosetta.core import *
from pyrosetta.rosetta.core.scoring.dssp import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta import *
from pyrosetta import *
from collections import namedtuple, defaultdict
from pyrosetta.rosetta.core.select.residue_selector import *
import numpy as np
from pyrosetta import rosetta
pyrosetta.Pose = pyrosetta.rosetta.core.pose.Pose
from pyrosetta import toolbox
from pyrosetta.toolbox import *
from pyrosetta.toolbox import generate_resfile
import math
import sys

from worms.filters.alignment_validator import AlignmentValidator, PoseInfo
from worms.filters.contact_analyzer import ContactAnalyzer, PoseMap
from worms.filters.interface_contacts import count_contacts_accross_junction, identify_helical_segments


def get_affected_positions(pose, prov):
    analyzer = ContactAnalyzer()
    input_pose_maps = list()
    src_pose_range = defaultdict(list)
    final_pose_range = defaultdict(list)
    is_symmetric = {}
    for lb, ub, src_pose, src_lb, src_ub in prov:
        src_chains = src_pose.split_by_chain()
        src_asu = src_chains[1]
        if len(list(src_chains)
               ) > 1 and src_chains[1].size() == src_chains[2].size():
            is_symmetric[src_pose] = True
            src_asu_size = src_asu.size()
            src_lb_asu = (src_lb - 1) % src_asu_size + 1
            src_ub_asu = (src_ub - 1) % src_asu_size + 1
        else:
            is_symmetric[src_pose] = False
            src_lb_asu = src_lb
            src_ub_asu = src_ub
        src_pose_range[src_pose].append([src_lb_asu, src_ub_asu])
        final_pose_range[src_pose].append([lb, ub])
    for src_pose in src_pose_range.keys():
        #asym_pose = pyrosetta.rosetta.core.pose.Pose()
        #pyrosetta.rosetta.core.pose.symmetry.extract_asymmetric_unit(src_pose, asym_pose,Fa
        if is_symmetric[src_pose]:

            src_chains = src_pose.split_by_chain()
            src_asu = src_chains[1]
            src_asu_size = src_asu.size()
        else:
            src_asu = src_pose
            src_asu_size = src_pose.size()
        input_pose_maps.append(
            PoseMap(
                src_asu.clone(), src_pose_range[src_pose],
                final_pose_range[src_pose]
            )
        )
        # # src_asu.dump_pdb('TEST.pdb')
        # PRINTDBG(
        #     'size %s src_map: %s final map: %s' % (
        #         src_asu.size(), src_pose_range[src_pose],
        #         final_pose_range[src_pose]
        #     )
        # )

    last_chain = pose.chain(prov[0][1])
    final_junction_res = []
    for i, splice in enumerate(prov[1:]):
        lb = splice[0]
        current_chain = pose.chain(lb)
        if current_chain == last_chain:
            final_junction_res.append(lb)
        else:
            last_chain = current_chain

    modified_positions, new_contact_positions, lost_contact_positions = analyzer.get_affected_positions(
        pose, input_pose_maps
    )

    return modified_positions, new_contact_positions, lost_contact_positions, final_junction_res


def make_junct_strs(db, criteria, ssdag, idx):

    body_names = list()
    for i in range(len(idx)):
        fn = bytes(ssdag.bbs[i][ssdag.verts[i].ibblock[idx[i]]].file).decode()
        dbentry = db.get_json_entry(fn)
        body_names.append(dbentry['name'])

    enres = [ssdag.verts[i].ires[idx[i], 0] + 1 for i in range(len(idx))]
    exres = [ssdag.verts[i].ires[idx[i], 1] + 1 for i in range(len(idx))]
    Nseg = len(idx) - 1 if criteria.is_cyclic else len(idx)
    junct_str = '%s_ex%s_en%s_%s_' % (
        body_names[0], exres[0], enres[1], body_names[1]
    )
    junct_str1 = '%-20s %4d %4d %-20s ' % (
        body_names[0], exres[0], enres[1], body_names[1]
    )
    for i in range(1, len(idx) - 1):
        junct_str = junct_str + 'ex%s_en%s_%s_' % (
            exres[i], enres[i + 1], body_names[i + 1]
        )
        junct_str1 = junct_str1 + '%4d %4d %-20s ' % (
            exres[i], enres[i + 1], body_names[i + 1]
        )
    if criteria.is_cyclic:
        junct_str = junct_str + 'ex%s_en%s' % (exres[Nseg - 1], enres[Nseg])
        junct_str1 = junct_str1 + '%4d %4d ' % (exres[Nseg - 1], enres[Nseg])
    return junct_str, junct_str1


def run_db_filters(
        databases,
        criteria,
        ssdag,
        iresult,
        idx,
        pose,
        prov,
        postfilt_splice_max_rms,
        postfilt_splice_rms_length,
        postfilt_splice_ncontact_cut,
        postfilt_splice_ncontact_no_helix_cut,
        postfilt_splice_nhelix_contacted_cut,
        **kw,
):
    bbdb, _ = databases
    AV = AlignmentValidator(
        superimpose_rmsd=postfilt_splice_max_rms + 0.2,
        superimpose_length=postfilt_splice_rms_length
    )
    junct_str, junct_str1 = make_junct_strs(databases[0], criteria, ssdag, idx)

    ori_segment_map = []
    final_segment_map = []
    filter = 'Pass'

    # AV.test_pair_alignment and prefiltering don't agree perfectly,
    # but in this case, I trust the prefilter numbers
    super_grade = 'A'
    # for i in range(len(idx) - 1):
    # if ssdag.verts[i].dirn[1] == 0:  # NC
    #     poseN = bbdb.pose(
    #         ssdag.bbs[i][ssdag.verts[i].ibblock[idx[i]]].file
    #     )
    #     poseC = bbdb.pose(
    #         ssdag.bbs[i + 1][ssdag.verts[i + 1].ibblock[idx[i + 1]]].file
    #     )
    #     chainN = int(ssdag.verts[i].ichain[idx[i], 1] + 1)
    #     chainC = int(ssdag.verts[i + 1].ichain[idx[i + 1], 0] + 1)
    #     resN = int(ssdag.verts[i].ires[idx[i], 1] + 1)
    #     resC = int(ssdag.verts[i + 1].ires[idx[i + 1], 0] + 1)
    # elif ssdag.verts[i].dirn[1] == 1:  # CN
    #     poseC = bbdb.pose(
    #         ssdag.bbs[i][ssdag.verts[i].ibblock[idx[i]]].file
    #     )
    #     poseN = bbdb.pose(
    #         ssdag.bbs[i + 1][ssdag.verts[i + 1].ibblock[idx[i + 1]]].file
    #     )
    #     chainN = int(ssdag.verts[i + 1].ichain[idx[i + 1], 0] + 1)
    #     chainC = int(ssdag.verts[i].ichain[idx[i], 1] + 1)
    #     resN = int(ssdag.verts[i + 1].ires[idx[i + 1], 0] + 1)
    #     resC = int(ssdag.verts[i].ires[idx[i], 1] + 1)
    # else:
    #     print('Warning: no direction information')
    #     continue

    # chainsN = poseN.split_by_chain()
    # chainsC = poseC.split_by_chain()
    # assert 1 <= chainN <= len(chainsN)
    # assert 1 <= chainC <= len(chainsC)
    # ofstN = sum(len(chainsN[i + 1]) for i in range(chainN - 1))
    # ofstC = sum(len(chainsC[i + 1]) for i in range(chainC - 1))
    # pi1 = PoseInfo(chainsN[chainN])
    # pi2 = PoseInfo(chainsC[chainC])
    # assert 1 <= (resN - ofstN) <= len(pi1._pose)
    # assert 1 <= (resC - ofstC) <= len(pi2._pose)

    # test, result = AV.test_pair_alignment(
    #     pi1, pi2, resN - ofstN, resC - ofstC
    # )
    # if test is None:
    #     filter = 'Fail_sup'
    #     super_grade = 'F'
    # elif result < postfilt_splice_max_rms:
    #     super_grade = 'A'
    # else:
    #     super_grade = 'B'

    last_chain = pose.chain(prov[0][1]
                            )  #chain number  of upper bound of first segment
    final_junction_res = []
    for i, splice in enumerate(prov[1:]):
        lb, ub, *_ = splice
        current_chain = pose.chain(lb)
        if current_chain == last_chain:  #in same chain so bonafide junction
            final_junction_res.append(lb)
        else:
            last_chain = current_chain
    min_contacts = 999
    min_contacts_no_helix = 999
    min_helices_contacted = 999
    num_contacts = list()
    num_contacts_no_helix = list()
    num_helices_contacted = list()
    #                for jct in [final_junction_res[0],final_junction_res[2]]:

    for jct in final_junction_res:
        nc, nc_no_helix, n_helix_contacted, n_helix_contacted_before, n_helix_contacted_after = count_contacts_accross_junction(
            pose, jct
        )
        # nc,n_helix_contacted=count_contacts_accross_junction(pose,jct)
        # print(
        #     'iresult %s jct_res %s contacts %s contacts without junction helix %s n_helices_contacted by junction helix %s , preceding helix %s , and following helix %s'
        #     % (
        #         iresult, jct, nc, nc_no_helix, n_helix_contacted,
        #         n_helix_contacted_before, n_helix_contacted_after
        #     )
        # )

        if nc < min_contacts: min_contacts = nc
        if nc_no_helix < min_contacts_no_helix:
            min_contacts_no_helix = nc_no_helix
        n_helix = min(
            n_helix_contacted, n_helix_contacted_before,
            n_helix_contacted_after
        )
        num_contacts.append(nc)
        num_contacts_no_helix.append(nc_no_helix)
        num_helices_contacted.append(n_helix)
        if n_helix < min_helices_contacted: min_helices_contacted = n_helix
        if min_contacts < postfilt_splice_ncontact_cut - 10:
            nc_grade = 'F'
            filter = 'Fail_con'
        elif min_contacts < postfilt_splice_ncontact_cut:
            nc_grade = 'B'
        else:
            nc_grade = 'A'
        if min_contacts_no_helix < postfilt_splice_ncontact_no_helix_cut:
            nc_no_helix_grade = 'B'
        else:
            nc_no_helix_grade = 'A'
        if min_helices_contacted < postfilt_splice_nhelix_contacted_cut:
            helix_contact_grade = 'B'
        else:
            helix_contact_grade = 'A'
    grade = super_grade + nc_grade + nc_no_helix_grade + helix_contact_grade
    if filter[0] != 'F': filter = grade

    ss = Dssp(pose).get_dssp_secstruct()
    close_to_junction = []
    for resN in final_junction_res:
        in_helix, before_helix, after_helix, helix_id = identify_helical_segments(
            ss, resN
        )
        if len(before_helix) is 0:
            print('bad before helix', final_junction_res)
        start_helix_list = sorted(before_helix[min(before_helix.keys())])
        end_helix_list = sorted(after_helix[max(after_helix.keys())])
        if start_helix_list == []:
            start_res = 1
        else:
            start_res = start_helix_list[0]
        if end_helix_list == []:
            end_res = pose.size()
        else:
            end_res = end_helix_list[-1]
        close = [i for i in range(start_res, end_res)
                 ]  # include loop residues also
        # PRINTDBG(
        #     'before helix: %s %s %s %s' % (
        #         resN, before_helix[min(before_helix.keys())],
        #         after_helix[max(after_helix.keys())], in_helix
        #     )
        # )
        # PRINTDBG('begin, end %s %s' % (start_res, end_res))
        close_to_junction = close_to_junction + close
    # PRINTDBG('close: %s' % close_to_junction)
    input_pose_maps = []
    # for i in range(len(w[iresult])):

    src_pose_range = defaultdict(list)
    final_pose_range = defaultdict(list)
    is_symmetric = {}
    for lb, ub, src_pose, src_lb, src_ub in prov:
        src_chains = src_pose.split_by_chain()
        src_asu = src_chains[1]
        if len(list(src_chains)
               ) > 1 and src_chains[1].size() == src_chains[2].size():
            is_symmetric[src_pose] = True
            src_asu_size = src_asu.size()
            src_lb_asu = (src_lb - 1) % src_asu_size + 1
            src_ub_asu = (src_ub - 1) % src_asu_size + 1
        else:
            is_symmetric[src_pose] = False
            src_lb_asu = src_lb
            src_ub_asu = src_ub
        src_pose_range[src_pose].append([src_lb_asu, src_ub_asu])
        final_pose_range[src_pose].append([lb, ub])
    for src_pose in src_pose_range.keys():
        #asym_pose = pyrosetta.rosetta.core.pose.Pose()
        #pyrosetta.rosetta.core.pose.symmetry.extract_asymmetric_unit(src_pose, asym_pose,Fa
        if is_symmetric[src_pose]:

            src_chains = src_pose.split_by_chain()
            src_asu = src_chains[1]
            src_asu_size = src_asu.size()
        else:
            src_asu = src_pose
            src_asu_size = src_pose.size()
        input_pose_maps.append(
            PoseMap(
                src_asu.clone(), src_pose_range[src_pose],
                final_pose_range[src_pose]
            )
        )
        #           src_asu.dump_pdb('TEST.pdb')
        # PRINTDBG(
        #     'size %s src_map: %s final map: %s' % (
        #         src_asu.size(), src_pose_range[src_pose],
        #         final_pose_range[src_pose]
        #     )
        # )


#          pose_map_mab = PoseMap(w.pose(iresult), original_ranges_mab, final_ranges_mab)

# sys.stdout.flush()
# PRINTDBG(
#     'close to junction: %s input_pose_maps: %s' %
#     (close_to_junction, input_pose_maps)
# )

    return (
        junct_str, junct_str1, filter, grade, final_junction_res, min_contacts,
        min_contacts_no_helix, min_helices_contacted, num_contacts,
        num_contacts_no_helix, num_helices_contacted
    )
