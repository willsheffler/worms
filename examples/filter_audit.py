import sys
import itertools as it
import shutil

import numpy as np

from worms.app import parse_args
from worms.edge import splice_metrics_pair, _ires_from_conn
from worms.bblock import bb_splice_res
from worms.util import splice_poses

from pyrosetta.rosetta.core.pose import append_subpose_to_pose

from worms.filters.alignment_validator import AlignmentValidator, PoseInfo
from worms.filters.contact_analyzer import ContactAnalyzer, PoseMap
from worms.filters.interface_contacts import (
   count_contacts_accross_junction,
   identify_helical_segments,
)

def filter_audit():
   print("filter_audit")
   print(sys.argv)
   # shutil.rmtree('worms_filter_audit_cache', ignore_errors=1)
   args = sys.argv[1:]
   if not args:
      args += "--geometry Null()".split()
      args += "--bbconn _N het C_ het".split()
      args += "--cachedirs worms_filter_audit_cache".split()
      args += "--dbfiles worms/data/repeat_1_2.json".split()
      # args += '--dbfiles worms/data/master_database_generation2.json'.split()

   crit, kw = parse_args(args)
   bbdb, spdb = kw["db"]
   kw["splice_clash_d2"] = 3.0**2

   bbsN = bbdb.query("Het:C")
   bbsC = bbdb.query("Het:N")
   print("len(bbsN)", len(bbsN), "len(bbsC)", len(bbsC))

   av = AlignmentValidator(
      superimpose_rmsd=kw["splice_max_rms"],
      superimpose_length=kw["splice_rms_range"] * 2 + 1,
   )

   for bb1, bb2 in it.product(bbsN, bbsC):
      rms, nclash, ncontact, ncnh, nhc = splice_metrics_pair(
         bb1,
         bb2,
         kw["splice_max_rms"],
         kw["splice_clash_d2"],
         kw["splice_contact_d2"],
         kw["splice_rms_range"],
         kw["splice_clash_contact_range"],
         kw["splice_max_chain_length"],
         splice_clash_contact_by_helix=True,
         skip_on_fail=False,
      )
      print("splices shape", rms.shape)
      splice_res_c = bb_splice_res(bb1, dirn=1)
      splice_res_n = bb_splice_res(bb2, dirn=0)
      assert np.all(_ires_from_conn(bb1.connections, 1) == splice_res_c)
      assert np.all(_ires_from_conn(bb2.connections, 0) == splice_res_n)
      assert len(splice_res_c) == rms.shape[0]
      assert len(splice_res_n) == rms.shape[1]

      pose1 = bbdb.pose(bb1.file)
      pose2 = bbdb.pose(bb2.file)
      pi1 = PoseInfo(pose1)
      pi2 = PoseInfo(pose2)

      # silly sanity check
      assert np.allclose(
         bb1.ncac[splice_res_n[0], 0],
         [
            pose1.residue(splice_res_n[0] + 1).xyz("N").x,
            pose1.residue(splice_res_n[0] + 1).xyz("N").y,
            pose1.residue(splice_res_n[0] + 1).xyz("N").z,
            1,
         ],
      )

      post_rms = np.zeros_like(rms)
      post_ncontacts = np.zeros_like(ncontact)

      for i, ir in enumerate(splice_res_c):
         for j, jr in enumerate(splice_res_n):
            # if nclash[i, j] > 0: continue

            # test, result = av.testing_pair_alignment(pi1, pi2, ir + 1, jr + 1)
            # if test is not None:
            # post_rms[i, j] = result
            pose = splice_poses(pose1, pose2, ir, jr)

            # print('debug:', ir, jr)
            # if nclash[i, j]:
            #     pose.dump_pdb('pose_clash_%i_%i.pdb' % (ir, jr))
            # else:
            #     pose.dump_pdb('pose_noclh_%i_%i.pdb' % (ir, jr))

            pose.update_residue_neighbors()
            nc, post_ncnh, post_nhc, post_nhcb, post_nhca = count_contacts_accross_junction(
               pose, ir)
            post_ncontacts[i, j] = nc
            print(
               ir,
               jr,
               " ",
               int(nclash[i, j]),
               " ",
               ncontact[i, j],
               nc,
               " ",
               ncnh[i, j],
               post_ncnh,
               " ",
               nhc[i, j],
               post_nhc,
               post_nhcb,
               post_nhca,
            )
            # sys.exit()

      # for i, ir in enumerate(splice_res_c):
      # for j, jr in enumerate(splice_res_n):
      # if rms[i, j] > 0 and post_rms[i, j] > 0:
      # print(i, j, ir, jr, rms[i, j], post_rms[i, j])

      # build pose and run_db_filters?
      break

filter_audit()
