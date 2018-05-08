"""TODO: Summary
"""
from pyrosetta import *
from pyrosetta import rosetta
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core import *
from pyrosetta.rosetta.core.scoring.dssp import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.core.select.residue_selector import *
import argparse
import json
from worms import *
import math
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from collections import defaultdict, namedtuple, Counter

sys.path.insert(0, '/home/sheffler/src/rif/buildR/lib.linux-x86_64-3.6')
from rif import rcl, vis  # 'rosetta compatibility layer'


def count_contacts_accross_junction(pose, resN):
    """TODO: Summary

    Args:
        pose (TYPE): Description
        resN (TYPE): Description

    Returns:
        TYPE: Description
    """
    ss = Dssp(pose).get_dssp_secstruct()
    if ss[resN] != 'H':
        print('Warning: junction residue not helix:  %s' % resN)
        return -1, -1, -1, -1, -1
    in_helix, before_helix, after_helix, helix_id = identify_helical_segments(
        ss, resN)
    before_contact_res = get_contacts(in_helix,
                                      before_helix[-1] + before_helix[-2],
                                      after_helix[1] + after_helix[2], pose)
    after_contact_res = get_contacts(in_helix, after_helix[1] + after_helix[2],
                                     before_helix[-1] + before_helix[-2], pose)
    contact_res_no_helix = get_contacts([],
                                        before_helix[-1] + before_helix[-2],
                                        after_helix[1] + after_helix[2], pose)
    begin_res = before_helix[min(before_helix.keys())][0]
    end_res = after_helix[max(after_helix.keys())][-1]

    return (len(before_contact_res) + len(after_contact_res),
            len(contact_res_no_helix),
            get_number_helices_contacted(in_helix, helix_id, pose),
            get_number_helices_contacted(before_helix[-1], helix_id, pose),
            get_number_helices_contacted(after_helix[1], helix_id, pose),
            begin_res, end_res)


def get_number_helices_contacted(helix, helix_id, pose):
    """TODO: Summary

    Args:
        helix (TYPE): Description
        helix_id (TYPE): Description
        pose (TYPE): Description

    Returns:
        TYPE: Description
    """
    res_selector = ResidueIndexSelector()
    for index in helix:
        res_selector.append_index(index)
    res_indices = res_selector.apply(pose)
    nb_selector = NeighborhoodResidueSelector(res_indices, 8, False)
    nb_indices = nb_selector.apply(pose)
    contact_res = [
        index for index in range(1,
                                 len(nb_indices) + 1) if nb_indices[index]
    ]
    helices_contacted = set()
    for res in contact_res:
        if res in helix_id.keys(): helices_contacted.add(helix_id[res])
    return len(helices_contacted)


def get_contacts(helix, set1, set2, pose):
    """TODO: Summary

    Args:
        helix (TYPE): Description
        set1 (TYPE): Description
        set2 (TYPE): Description
        pose (TYPE): Description

    Returns:
        TYPE: Description
    """
    res_selector = ResidueIndexSelector()
    for index in helix:
        res_selector.append_index(index)
    for index in set1:
        res_selector.append_index(index)

    res_indices = res_selector.apply(pose)
    nb_selector = NeighborhoodResidueSelector(res_indices, 8, False)
    nb_indices = nb_selector.apply(pose)
    contact_res = [
        index for index in range(1,
                                 len(nb_indices) + 1) if nb_indices[index]
    ]
    nearby_contact_res = set(contact_res).intersection(set(set2))
    return nearby_contact_res


def identify_helical_segments(ss, resN):
    """TODO: Summary

    Args:
        ss (TYPE): Description
        resN (TYPE): Description

    Returns:
        TYPE: Description
    """
    # identify residues in same helix
    helix_id = {}
    in_helix = []
    resi = resN
    resT = len(ss)
    while ss[resi] == 'H' and resi > 0:
        in_helix.append(resi)
        helix_id[resi] = 0
        resi = resi - 1
    H_begin = resi

    resi = resN
    while ss[resi] == 'H' and resi < resT:
        in_helix.append(resi)
        helix_id[resi] = 0
        resi = resi + 1
    H_end = resi - 1

    # identify residues in preceding three helices
    # actually, just need one dict, use -1 for helix before and +1 for helix after
    before_helix = defaultdict(list)
    h_index = 0
    in_H = False
    for i in range(H_begin - 1, 0, -1):
        if ss[i] == 'H':
            if not in_H:
                h_index = h_index - 1
                in_H = True
                if h_index == -3: break
            before_helix[h_index].append(i)
            helix_id[i] = h_index

        else:
            in_H = False


# identify residues in following two helices
    after_helix = defaultdict(list)
    h_index = 0
    in_H = False
    for i in range(H_end + 1, resT):
        if ss[i] == 'H':
            if not in_H:
                h_index = h_index + 1
                in_H = True
                if h_index == 3: break
            after_helix[h_index].append(i)
            helix_id[i] = h_index

        else:
            in_H = False
    return in_helix, before_helix, after_helix, helix_id


def PRINTDBG(msg):
    """TODO: Summary

    Args:
        msg (TYPE): Description

    Returns:
        TYPE: Description
    """
    # print(msg)
    return


# holds pose-related info and lazily constructs subposes for secondary
# structure elements


class PoseInfo:
    """TODO: Summary
    """

    def __init__(self, pose):
        """TODO: Summary

        Args:
            pose (TYPE): Description
        """
        self._pose = pose
        #        self._chains=pose.split_by_chain()
        #        self._Nchains=len(chains)
        # self._dssp = Dssp(pose).get_dssp_secstruct() for chain in self._chains]

        # lookup of pose to _dssp
        self._dssp = Dssp(pose).get_dssp_secstruct()
        # number of uninterrupted matching DSSP positions to the left
        self._length_lhs = None
        # number of uninterrupted matching DSSP positions to the right
        self._length_rhs = None
        # pose array with one subpose per secondary structure element:
        # subpose[residue_index]
        self._subposes = [None for x in range(0, pose.size())]

    # returns a pose for the full secondary-structure-element that includes the position requested
    # position0 falls is at index get_run_length_lhs(position) within the
    # returned pose
    def get_subpose(self, position0):
        """TODO: Summary

        Args:
            position0 (TYPE): Description

        Returns:
            TYPE: Description
        """
        if self._subposes[position0] is None:
            start0 = position0 - self.get_run_length_lhs(position0)
            end0 = position0 + self.get_run_length_rhs(position0)
            subpose = pose_from_sequence('')  # how do i create an empty pose?
            rosetta.core.pose.append_subpose_to_pose(
                subpose, self._pose, start0 + 1, end0 + 1, False)
            for x in range(start0, end0 + 1):
                self._subposes[x] = subpose
        return self._subposes[position0]

    # determines the remaining secondary structure element length on the right
    # of a position
    def get_run_length_rhs(self, position0):
        """TODO: Summary

        Args:
            position0 (TYPE): Description

        Returns:
            TYPE: Description
        """
        if self._length_rhs is None:
            self._length_rhs = [0 for x in self._dssp]
            for position in range(len(self._dssp) - 2, 0, -1):
                if (self._dssp[position] == self._dssp[position + 1]):
                    self._length_rhs[
                        position] = self._length_rhs[position + 1] + 1
        return self._length_rhs[position0]

    # determines the remaining secondary structure element length on the left
    # of a position
    def get_run_length_lhs(self, position0):
        """TODO: Summary

        Args:
            position0 (TYPE): Description

        Returns:
            TYPE: Description
        """
        if self._length_lhs is None:
            self._length_lhs = [0 for x in self._dssp]
            for position in range(1, len(self._dssp)):
                if (self._dssp[position] == self._dssp[position - 1]):
                    self._length_lhs[
                        position] = self._length_lhs[position - 1] + 1
        return self._length_lhs[position0]

    def dssp_all_positions(self):
        """TODO: Summary

        Returns:
            TYPE: Description
        """
        return self._dssp  # start at 1 or 0?

    def dssp_single_position(self, position0):
        """TODO: Summary

        Args:
            position0 (TYPE): Description

        Returns:
            TYPE: Description
        """
        return self._dssp[position0]


class AlignmentValidator:
    """TODO: Summary
    """

    def __init__(self, superimpose_rmsd=0.4, superimpose_length=10):
        """TODO: Summary

        Args:
            superimpose_rmsd (float, optional): Description
            superimpose_length (int, optional): Description
        """
        assert (0 < superimpose_rmsd)
        assert (0 < superimpose_length)
        #        self._pose_infos = {} # map of pose to PoseInfo
        self._superimpose_rmsd = superimpose_rmsd
        self._superimpose_length = superimpose_length

    def get_dssp(self, pdb_name, pose):
        """TODO: Summary

        Args:
            pdb_name (TYPE): Description
            pose (TYPE): Description

        Returns:
            TYPE: Description
        """
        if pdb_name not in self._pose_infos:
            self._pose_infos[pdb_name] = PoseInfo(pose)

        pose_info = self._pose_infos[pdb_name]
        return pose_info.dssp_all_positions()

    def get_allowed_alignment_positions(self, pdb_name, pose, sse_type='H'):
        """TODO: Summary

        Args:
            pdb_name (TYPE): Description
            pose (TYPE): Description
            sse_type (str, optional): Description

        Returns:
            TYPE: Description
        """
        if pdb_name not in self._pose_infos:
            self._pose_infos[pdb_name] = PoseInfo(pose)

        pose_info = self._pose_infos[pose]
        return [
            i + 1 for i in range(0, pose.size())
            if pose_info.dssp(i) == sse_type
        ]

    def test_pair_alignment(self,
                            pose_info_n,
                            pose_info_c,
                            index_n1,
                            index_c1,
                            superimpose_rmsd=None,
                            superimpose_length=None):
        """TODO: Summary

        Args:
            pose_info_n (TYPE): Description
            pose_info_c (TYPE): Description
            index_n1 (TYPE): Description
            index_c1 (TYPE): Description
            superimpose_rmsd (None, optional): Description
            superimpose_length (None, optional): Description

        Returns:
            TYPE: Description
        """
        index_n0 = index_n1 - 1
        index_c0 = index_c1 - 1

        if superimpose_rmsd is None:
            superimpose_rmsd = self._superimpose_rmsd

        if superimpose_length is None:
            superimpose_length = self._superimpose_length

        if pose_info_n._pose.size(
        ) < superimpose_length or pose_info_c._pose.size() < superimpose_length:
            print('failed size filter')
            return None, min(pose_info_n._pose.size(),
                             pose_info_c._pose.size())
        # look up or lazily create pose-related info (dssp, sse-length tables, subpose tables)
#        if pose_n not in self._pose_infos:
#            self._pose_infos[pose_n] = PoseInfo(pose_n)

#       if pose_c not in self._pose_infos:
#            self._pose_infos[pose_c] = PoseInfo(pose_c)

#        pose_info_n = self._pose_infos[pose_n]
#        pose_info_c = self._pose_infos[pose_c]

        sse_shared_left = min(
            pose_info_n.get_run_length_lhs(index_n0),
            pose_info_c.get_run_length_lhs(index_c0))
        sse_shared_right = min(
            pose_info_n.get_run_length_rhs(index_n0),
            pose_info_c.get_run_length_rhs(index_c0))
        sse_shared_length = sse_shared_left + sse_shared_right + 1

        if (sse_shared_left + sse_shared_right + 1 < superimpose_length):
            print(
                'failed superimpose_length position index_n0,index_c0=%d,%d' %
                (index_n0, index_c0))
            return None, sse_shared_left + sse_shared_right

        #PRINTDBG('index_n0 ' + str(index_n0))
        #PRINTDBG('index_c0 ' + str(index_c0))
        #PRINTDBG('sse_shared_left ' + str(sse_shared_left))
        #PRINTDBG('sse_shared_right ' + str(sse_shared_right))

        assert (index_n0 - sse_shared_left + 1 >= 0)
        assert (index_n0 + sse_shared_right < pose_info_n._pose.size())
        assert (index_c0 - sse_shared_left + 1 >= 0)
        assert (index_c0 + sse_shared_right < pose_info_c._pose.size())
        # look up the poses for each sse, determine the align indices,
        # reinitialize the coordinates to the original values,
        sse_n = pose_info_n.get_subpose(index_n0)
        sse_c = pose_info_c.get_subpose(index_c0)
        sse_index_n = pose_info_n.get_run_length_lhs(index_n0)
        sse_index_c = pose_info_c.get_run_length_lhs(index_c0)
        assert (0 <= sse_index_n and sse_index_n < sse_n.size())
        assert (0 <= sse_index_c and sse_index_c < sse_c.size())

        #PRINTDBG('sse_n ' + str(sse_n))
        #PRINTDBG('sse_c ' + str(sse_c))
        #PRINTDBG('sse_n_length ' + str(sse_n.size()))
        #PRINTDBG('sse_c_length ' + str(sse_c.size()))
        #PRINTDBG('sse_index_n ' + str(sse_index_n))
        #PRINTDBG('sse_index_c ' + str(sse_index_c))

        self.align_pose_at_position1_sheffler(sse_n, sse_c, sse_index_n + 1,
                                              sse_index_c + 1)
        residue_distances2 = [
            self.residue_ncac_avg_distance2(
                sse_n.residue(sse_index_n + x + 1),
                sse_c.residue(sse_index_c + x + 1))
            for x in range(-sse_shared_left, sse_shared_right)
        ]
        #PRINTDBG('residue distances = ' + str(residue_distances))

        sum_distance2 = 0
        max_sum_distance2 = superimpose_length * superimpose_rmsd**2
        PRINTDBG(
            'superimpose_length = %f, superimpose_rmsd = %f, max_sum_distance2 = %f, residue count = %d'
            % (superimpose_length, superimpose_rmsd, max_sum_distance2,
               len(residue_distances2)))
        lowest_observed_distance2 = math.inf
        for index in range(0, len(residue_distances2)):
            sum_distance2 = sum_distance2 + residue_distances2[index]
            PRINTDBG('add %f' % residue_distances2[index])
            if superimpose_length <= index:
                subtract_index = index - superimpose_length
                sum_distance2 = sum_distance2 - \
                    residue_distances2[index - superimpose_length]
                PRINTDBG(
                    'subtract %f' % residue_distances2[index
                                                       - superimpose_length])
                assert (subtract_index >= 0)
            PRINTDBG('compute')
            if superimpose_length <= index + 1 and sum_distance2 < lowest_observed_distance2:
                lowest_observed_distance2 = sum_distance2

            if lowest_observed_distance2 <= max_sum_distance2:
                return 'PASS', np.sqrt(
                    lowest_observed_distance2 / superimpose_length)
        # else:
        #    print('Filter failure:  lowest observed distance = %s %s'%(lowest_observed_distance2,max_sum_distance2))
        # if dump_once:
        #    # ALIGN SUCCESS
        #    sse_n.dump_pdb('sse_n.pdb')
        #    sse_c.dump_pdb('sse_c.pdb')
        #    dump_once = False
        #print('residue_pair found: ' + str([index_n0, index_c0]))
        #PRINTDBG('rmsd: ' + str(rmsd))
        # else:
        #    # ALIGN FAIL
        #    if dump_once:
        #        sse_n.dump_pdb('fail_sse_n.pdb')
        #        sse_c.dump_pdb('fail_sse_c.pdb')
        #        dump_once = False

        print('failed rmsd filter %f, sum_distance2 %f, threshold %f' %
              (np.sqrt(lowest_observed_distance2 / superimpose_length),
               sum_distance2, max_sum_distance2))
        return None, np.sqrt(lowest_observed_distance2 / superimpose_length)

    # align pose_move to pose_ref at the indicated positions (using 1-indexing)
    def align_pose_at_position1_sheffler(self, pose_move, pose_ref,
                                         position_move, position_ref):
        """TODO: Summary

        Args:
            pose_move (TYPE): Description
            pose_ref (TYPE): Description
            position_move (TYPE): Description
            position_ref (TYPE): Description
        """
        stubs_ref = rcl.bbstubs(
            pose_ref, [position_ref])['raw']  # gets 'stub' for reslist
        stubs_move = rcl.bbstubs(
            pose_move, [position_move])['raw']  # raw field is position matrix
        # PRINTDBG(stubs_ref.shape)  # homo coords numpy array n x 4 x 4
        # PRINTDBG(stubs_move.shape)
        # a @ b is np.matmul(a, b)
        xalign = stubs_ref @ np.linalg.inv(stubs_move)
        # PRINTDBG(xalign.shape)
        rcl.xform_pose(xalign[0], pose_move)

        #ncac_distance = residue_ncac_distance(pose_move.residue(position_move), pose_ref.residue(position_ref))
        #PRINTDBG('NCAC_distance = ' + str(ncac_distance))
        #assert(ncac_distance < 0.1)

    # average of the distance-squared for each atom N, C, and CA
    # i.e. if distances are 0.5, 1, and 1.5 angstroms, the result is (0.5^2 +
    # 1^2 + 1.5^2) / 3
    def residue_ncac_avg_distance2(self, res1, res2):
        """TODO: Summary

        Args:
            res1 (TYPE): Description
            res2 (TYPE): Description

        Returns:
            TYPE: Description
        """
        # N, CA, C ... index based lookup is faster than string-lookup
        atom_positions = [1, 2, 3]
        err2 = 0
        for position in atom_positions:
            # what's the utility function for vector3.distance(v1, v2)?
            v1 = res1.xyz(position)
            v2 = res2.xyz(position)
            err2 = err2 + (v1.x - v2.x)**2 + \
                (v1.y - v2.y)**2 + (v1.z - v2.z)**2
            #err2 = err2 + sum( [x * x for x in err])
        distance2 = err2 / 3  # 3 atom positions
        return distance2


# pass BakerFilter AND pose_info_all dict into grow so can filter and use
# cached pose info
class BakerFilter:
    """TODO: Summary

    Attributes:
        AV (TYPE): Description
        input (str): Description
        n_helix_contacted_threshold (TYPE): Description
        num_contact_no_helix_threshold (TYPE): Description
        num_contact_threshold (TYPE): Description
        pose_info_all (dict): Description
        score0_cutoff (TYPE): Description
        scorefxn (TYPE): Description
        superimpose_length (TYPE): Description
        superimpose_rmsd (TYPE): Description
    """

    def __init__(self,
                 score0_cutoff=1.0,
                 num_contact_threshold=40,
                 num_contact_no_helix_threshold=3,
                 n_helix_contacted_threshold=3,
                 superimpose_length=9,
                 superimpose_rmsd=0.7,
                 pose_info_all=None):
        """TODO: Summary

        Args:
            score0_cutoff (float, optional): Description
            num_contact_threshold (int, optional): Description
            num_contact_no_helix_threshold (int, optional): Description
            n_helix_contacted_threshold (int, optional): Description
            superimpose_length (int, optional): Description
            superimpose_rmsd (float, optional): Description
            pose_info_all (None, optional): Description
        """
        self.score0_cutoff = score0_cutoff
        self.num_contact_threshold = num_contact_threshold
        self.num_contact_no_helix_threshold = num_contact_no_helix_threshold
        self.n_helix_contacted_threshold = n_helix_contacted_threshold
        self.superimpose_length = superimpose_length
        self.superimpose_rmsd = superimpose_rmsd
        self.scorefxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function(
            "score0")
        self.AV = AlignmentValidator(
            superimpose_rmsd=superimpose_rmsd,
            superimpose_length=superimpose_length)
        if pose_info_all is None:
            self.pose_info_all = {}
            self.input = 'Pose'
        else:
            self.pose_info_all = pose_info_all
            self.input = 'PoseInfo'

    def filter_worm(self,
                    pose,
                    junction_res,
                    src_pose_N_jct_res,
                    src_pose_C_jct_res,
                    src_pose_N=None,
                    src_pose_C=None,
                    src_pose_N_name=None,
                    src_pose_C_name=None,
                    src_pose_N_chain=None,
                    src_pose_C_chain=None):
        """TODO: Summary

        Args:
            pose (TYPE): Description
            junction_res (TYPE): Description
            src_pose_N_jct_res (TYPE): Description
            src_pose_C_jct_res (TYPE): Description
            src_pose_N (None, optional): Description
            src_pose_C (None, optional): Description
            src_pose_N_name (None, optional): Description
            src_pose_C_name (None, optional): Description
            src_pose_N_chain (None, optional): Description
            src_pose_C_chain (None, optional): Description

        Returns:
            TYPE: Description
        """
        if self.input == 'Pose':
            if src_pose_N not in self.pose_info_all:
                self.pose_info_all[src_pose_N] = PoseInfo(src_pose_N)
            if src_pose_C not in self.pose_info_all:
                self.pose_info_all[src_pose_C] = PoseInfo(src_pose_C)
            src_pose_N_info = self.pose_info_all[src_pose_N]
            src_pose_C_info = self.pose_info_all[src_pose_C]

        elif self.input == 'PoseInfo':
            src_pose_N_info = self.pose_info_all[(src_pose_N_name,
                                                  src_pose_N_chain)]
            src_pose_C_info = self.pose_info_all[(src_pose_C_name,
                                                  src_pose_C_name)]

        else:
            print('undefined input type')

        return self.filter_junction(pose, junction_res, src_pose_N_info,
                                    src_pose_C_info, src_pose_N_jct_res,
                                    src_pose_C_jct_res)

    def filter_junction(self, pose, junction_res, src_pose_N_info,
                        src_pose_C_info, src_pose_N_jct_res,
                        src_pose_C_jct_res):
        """TODO: Summary

        Args:
            pose (TYPE): Description
            junction_res (TYPE): Description
            src_pose_N_info (TYPE): Description
            src_pose_C_info (TYPE): Description
            src_pose_N_jct_res (TYPE): Description
            src_pose_C_jct_res (TYPE): Description

        Returns:
            TYPE: Description
        """
        nc, nc_no_helix, n_helix_contacted, n_helix_contacted_before, n_helix_contacted_after, begin_res, end_res = count_contacts_accross_junction(
            pose, junction_res)
        jct_pose = Pose()
        for res_id in range(begin_res, end_res):
            jct_pose.append_residue_by_bond(pose.residue(res_id))


#     mode=pyrosetta.rosetta.core.chemical.type_set_mode_from_string("centroid")
#     centroid_jct_pose=jct_pose.clone()
#     pyrosetta.rosetta.core.util.switch_to_residue_type_set(centroid_jct_pose,mode )

        score0 = self.scorefxn(jct_pose)
        #        AV.test_pair_alignment(,resN,resC)
        test, result = self.AV.test_pair_alignment(
            src_pose_N_info, src_pose_C_info, src_pose_N_jct_res,
            src_pose_C_jct_res)

        # Assign grades
        if test is None:
            super_grade = 'F'
        elif result < self.superimpose_rmsd:
            super_grade = 'A'
        else:
            super_grade = 'B'
        if nc < self.num_contact_threshold - 7:
            nc_grade = 'F'
        elif nc < self.num_contact_threshold:
            nc_grade = 'B'
        else:
            nc_grade = 'A'
        if nc_no_helix < self.num_contact_no_helix_threshold:
            nc_no_helix_grade = 'B'
        else:
            nc_no_helix_grade = 'A'
        if n_helix_contacted < self.n_helix_contacted_threshold:
            helix_contact_grade = 'B'
        else:
            helix_contact_grade = 'A'
        grade = super_grade + nc_grade + nc_no_helix_grade + helix_contact_grade
        return grade

        # if nc >= self.num_contact_threshold and nc_no_helix >= self.num_contact_no_helix_threshold and n_helix_contacted > self.n_helix_contacted_threshold and score0 < self.score0_cutoff and test == 'PASS' and result < self.superimpose_rmsd:
        # return 'Pass'
        # else:
        ##     print(nc,self.num_contact_threshold,nc_no_helix,self.num_contact_no_helix_threshold,n_helix_contacted,self.n_helix_contacted_threshold,score0,self.score0_cutoff,result,self.superimpose_rmsd, test)
        # return 'Fail'
