from pyrosetta.rosetta.utility import vector1_unsigned_long
from pyrosetta.rosetta.core import *
from pyrosetta.rosetta import *
from pyrosetta import *
from pyrosetta.rosetta.core.scoring.dssp import *
from pyrosetta.rosetta.core.scoring import *
import numpy as np
from pyrosetta import rosetta
import math
import sys

from worms import util


def PRINTDBG(msg):
    #print(msg)
    return


# holds pose-related info and lazily constructs subposes for secondary structure elements
class PoseInfo:
    def __init__(self, pose):
        self._pose = pose
        #        self._chains=pose.split_by_chain()
        #        self._Nchains=len(chains)
        #        self._dssp = Dssp(pose).get_dssp_secstruct() for chain in self._chains]

        self._dssp = Dssp(pose).get_dssp_secstruct()  # lookup of pose to _dssp
        self._length_lhs = None  # number of uninterrupted matching DSSP positions to the left
        self._length_rhs = None  # number of uninterrupted matching DSSP positions to the right
        self._subposes = [
            None for x in range(0, pose.size())
        ]  # pose array with one subpose per secondary structure element: subpose[residue_index]

    # returns a pose for the full secondary-structure-element that includes the position requested
    # position0 falls is at index get_run_length_lhs(position) within the returned pose
    def get_subpose(self, position0):
        if self._subposes[position0] is None:
            start0 = position0 - self.get_run_length_lhs(position0)
            end0 = position0 + self.get_run_length_rhs(position0)
            subpose = pose_from_sequence('')  # how do i create an empty pose?
            pose.append_subpose_to_pose(
                subpose, self._pose, start0 + 1, end0 + 1, False
            )
            for x in range(start0, end0 + 1):
                self._subposes[x] = subpose
        return self._subposes[position0]

    # determines the remaining secondary structure element length on the right of a position
    def get_run_length_rhs(self, position0):
        if self._length_rhs is None:
            self._length_rhs = [0 for x in self._dssp]
            for position in range(len(self._dssp) - 2, 0, -1):
                if (self._dssp[position] == self._dssp[position + 1]):
                    self._length_rhs[position
                                     ] = self._length_rhs[position + 1] + 1
        return self._length_rhs[position0]

    # determines the remaining secondary structure element length on the left of a position
    def get_run_length_lhs(self, position0):
        if self._length_lhs is None:
            self._length_lhs = [0 for x in self._dssp]
            for position in range(1, len(self._dssp)):
                if (self._dssp[position] == self._dssp[position - 1]):
                    self._length_lhs[position
                                     ] = self._length_lhs[position - 1] + 1
        return self._length_lhs[position0]

    def dssp_all_positions(self):
        return self._dssp  # start at 1 or 0?

    def dssp_single_position(self, position0):
        return self._dssp[position0]


class AlignmentValidator:
    def __init__(self, superimpose_rmsd=0.4, superimpose_length=10):
        assert (0 < superimpose_rmsd)
        assert (0 < superimpose_length)
        #        self._pose_infos = {} # map of pose to PoseInfo
        self._superimpose_rmsd = superimpose_rmsd
        self._superimpose_length = superimpose_length

    def get_dssp(self, pdb_name, pose):
        if pdb_name not in self._pose_infos:
            self._pose_infos[pdb_name] = PoseInfo(pose)

        pose_info = self._pose_infos[pdb_name]
        return pose_info.dssp_all_positions()

    def get_allowed_alignment_positions(self, pdb_name, pose, sse_type='H'):
        if pdb_name not in self._pose_infos:
            self._pose_infos[pdb_name] = PoseInfo(pose)

        pose_info = self._pose_infos[pose]
        return [
            i + 1
            for i in range(0, pose.size())
            if pose_info.dssp(i) == sse_type
        ]

    def test_pair_alignment(
            self,
            pose_info_n,
            pose_info_c,
            index_n1,
            index_c1,
            superimpose_rmsd=None,
            superimpose_length=None
    ):
        index_n0 = index_n1 - 1
        index_c0 = index_c1 - 1

        if superimpose_rmsd is None:
            superimpose_rmsd = self._superimpose_rmsd

        if superimpose_length is None:
            superimpose_length = self._superimpose_length

        if pose_info_n._pose.size(
        ) < superimpose_length or pose_info_c._pose.size() < superimpose_length:
            print('failed size filter')
            return None, min(
                pose_info_n._pose.size(), pose_info_c._pose.size()
            )
        # look up or lazily create pose-related info (dssp, sse-length tables, subpose tables)


#        if pose_n not in self._pose_infos:
#            self._pose_infos[pose_n] = PoseInfo(pose_n)

#       if pose_c not in self._pose_infos:
#            self._pose_infos[pose_c] = PoseInfo(pose_c)

#        pose_info_n = self._pose_infos[pose_n]
#        pose_info_c = self._pose_infos[pose_c]

        sse_shared_left = min(
            pose_info_n.get_run_length_lhs(index_n0),
            pose_info_c.get_run_length_lhs(index_c0)
        )
        sse_shared_right = min(
            pose_info_n.get_run_length_rhs(index_n0),
            pose_info_c.get_run_length_rhs(index_c0)
        )
        sse_shared_length = sse_shared_left + sse_shared_right + 1

        if (sse_shared_left + sse_shared_right + 1 < superimpose_length):
            print(
                'failed superimpose_length position index_n0,index_c0=%d,%d' %
                (index_n0, index_c0)
            )
            return None, sse_shared_left + sse_shared_right

        #PRINTDBG('index_n0 ' + str(index_n0))
        #PRINTDBG('index_c0 ' + str(index_c0))
        #PRINTDBG('sse_shared_left ' + str(sse_shared_left))
        #PRINTDBG('sse_shared_right ' + str(sse_shared_right))

        assert (index_n0 - sse_shared_left + 1 >= 0)
        assert (index_n0 + sse_shared_right < pose_info_n._pose.size())
        assert (index_c0 - sse_shared_left + 1 >= 0)
        assert (index_c0 + sse_shared_right < pose_info_c._pose.size())
        # look up the poses for each sse, determine the align indices, reinitialize the coordinates to the original values,
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

        self.align_pose_at_position1_sheffler(
            sse_n, sse_c, sse_index_n + 1, sse_index_c + 1
        )
        residue_distances2 = [
            self.residue_ncac_avg_distance2(
                sse_n.residue(sse_index_n + x + 1),
                sse_c.residue(sse_index_c + x + 1)
            ) for x in range(-sse_shared_left, sse_shared_right)
        ]
        #PRINTDBG('residue distances = ' + str(residue_distances))

        sum_distance2 = 0
        max_sum_distance2 = superimpose_length * superimpose_rmsd**2
        PRINTDBG(
            'superimpose_length = %f, superimpose_rmsd = %f, max_sum_distance2 = %f, residue count = %d'
            % (
                superimpose_length, superimpose_rmsd, max_sum_distance2,
                len(residue_distances2)
            )
        )
        # print('DBG', superimpose_length, len(residue_distances2))
        lowest_observed_distance2 = math.inf
        for index in range(0, len(residue_distances2)):
            sum_distance2 = sum_distance2 + residue_distances2[index]
            PRINTDBG('add %f' % residue_distances2[index])
            if superimpose_length <= index:
                subtract_index = index - superimpose_length
                sum_distance2 = sum_distance2 - residue_distances2[
                    index - superimpose_length
                ]
                PRINTDBG(
                    'subtract %f' %
                    residue_distances2[index - superimpose_length]
                )
                assert (subtract_index >= 0)
            PRINTDBG('compute')
            if superimpose_length <= index + 1 and sum_distance2 < lowest_observed_distance2:
                lowest_observed_distance2 = sum_distance2

            if lowest_observed_distance2 <= max_sum_distance2:
                return 'PASS ', np.sqrt(
                    lowest_observed_distance2 / superimpose_length
                )
        # else:
        #    print('Filter failure:  lowest observed distance = %s %s'%(lowest_observed_distance2,max_sum_distance2))
        #if dump_once:
        #    # ALIGN SUCCESS
        #    sse_n.dump_pdb('sse_n.pdb')
        #    sse_c.dump_pdb('sse_c.pdb')
        #    dump_once = False
        #print('residue_pair found: ' + str([index_n0, index_c0]))
        #PRINTDBG('rmsd: ' + str(rmsd))
        #else:
        #    # ALIGN FAIL
        #    if dump_once:
        #        sse_n.dump_pdb('fail_sse_n.pdb')
        #        sse_c.dump_pdb('fail_sse_c.pdb')
        #        dump_once = False

        print(
            'failed rmsd filter %f, sum_distance2 %f, threshold %f' % (
                np.sqrt(lowest_observed_distance2 / superimpose_length),
                sum_distance2, max_sum_distance2
            )
        )
        return None, np.sqrt(lowest_observed_distance2 / superimpose_length)

    # align pose_move to pose_ref at the indicated positions (using 1-indexing)
    def align_pose_at_position1_sheffler(
            self, pose_move, pose_ref, position_move, position_ref
    ):
        stubs_ref, _ = util.get_bb_stubs(pose_ref, [position_ref])
        stubs_move, _ = util.get_bb_stubs(pose_move, [position_move])
        # print(stubs_ref.shape)  # homo coords numpy array n x 4 x 4
        # print(stubs_move.shape)

        xalign = stubs_ref @ np.linalg.inv(stubs_move)
        # print(xalign.shape)
        util.xform_pose(xalign[0], pose_move)

        #ncac_distance = residue_ncac_distance(pose_move.residue(position_move), pose_ref.residue(position_ref))
        #print('NCAC_distance = ' + str(ncac_distance))
        #assert(ncac_distance < 0.1)

    # average of the distance-squared for each atom N, C, and CA
    # i.e. if distances are 0.5, 1, and 1.5 angstroms, the result is (0.5^2 + 1^2 + 1.5^2) / 3
    def residue_ncac_avg_distance2(self, res1, res2):
        atom_positions = [
            1, 2, 3
        ]  # N, CA, C ... index based lookup is faster than string-lookup
        err2 = 0
        for position in atom_positions:
            v1 = res1.xyz(
                position
            )  # what's the utility function for vector3.distance(v1, v2)?
            v2 = res2.xyz(position)
            err2 = err2 + (v1.x - v2.x)**2 + (v1.y -
                                              v2.y)**2 + (v1.z - v2.z)**2
            #err2 = err2 + sum( [x * x for x in err])
        distance2 = err2 / 3  # 3 atom positions
        return distance2
