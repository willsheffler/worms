from collections import defaultdict

import numpy as np

from pyrosetta import rosetta, init
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.scoring.dssp import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.core.select.residue_selector import *

def count_contacts_accross_junction(pose, resN):
   ss = Dssp(pose).get_dssp_secstruct()
   if ss[resN] != "H":
      print("Warning: junction residue not helix:  %s" % resN)
      return -1, -1, -1, -1, -1
   in_helix, before_helix, after_helix, helix_id = identify_helical_segments(ss, resN)
   before = before_helix[-1] + before_helix[-2]
   after = after_helix[1] + after_helix[2]

   before_contact_res = get_contacts(in_helix, before, after, pose)
   after_contact_res = get_contacts(in_helix, after, before, pose)

   before_contact_res_no_helix = get_contacts([], before, after, pose)
   after_contact_res_no_helix = get_contacts([], after, before, pose)

   return (
      len(before_contact_res) + len(after_contact_res),
      len(before_contact_res_no_helix) + len(after_contact_res_no_helix),
      get_number_helices_contacted(in_helix, helix_id, pose),
      get_number_helices_contacted(before_helix[-1], helix_id, pose),
      get_number_helices_contacted(after_helix[1], helix_id, pose),
   )

def get_number_helices_contacted(helix, helix_id, pose):
   res_selector = ResidueIndexSelector()
   for index in helix:
      res_selector.append_index(index)
   res_indices = res_selector.apply(pose)
   nb_selector = NeighborhoodResidueSelector(res_indices, 8, False)
   nb_indices = nb_selector.apply(pose)
   contact_res = [i for i in range(1, len(nb_indices) + 1) if nb_indices[i]]
   helices_contacted = set()
   for res in contact_res:
      if res in helix_id.keys():
         helices_contacted.add(helix_id[res])
   return len(helices_contacted)

def get_contacts(helix, set1, set2, pose):
   res_selector = ResidueIndexSelector()
   for index in helix:
      res_selector.append_index(index)
   for index in set1:
      res_selector.append_index(index)

   res_indices = res_selector.apply(pose)
   # print(
   # 'res_indices',
   # '+'.join(str(x) for x in (np.where(res_indices)[0] + 1))
   # )
   nb_selector = NeighborhoodResidueSelector(res_indices, 8, False)
   nb_indices = nb_selector.apply(pose)
   contact_res = [index for index in range(1, len(nb_indices) + 1) if nb_indices[index]]
   # print('contact_res', '+'.join(str(x) for x in contact_res))
   nearby_contact_res = set(contact_res).intersection(set(set2))
   # print('nearby_contact_res', '+'.join(str(x) for x in nearby_contact_res))
   return nearby_contact_res

def identify_helical_segments(ss, resN):
   # identify residues in same helix
   helix_id = {}
   in_helix = []
   resi = resN
   resT = len(ss)
   while ss[resi] == "H" and resi > 0:
      in_helix.append(resi)
      helix_id[resi] = 0
      resi = resi - 1
   H_begin = resi

   resi = resN
   while ss[resi] == "H" and resi < resT:
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
      if ss[i] == "H":
         if not in_H:
            h_index = h_index - 1
            in_H = True
            if h_index == -4:
               break
         before_helix[h_index].append(i)
         helix_id[i] = h_index

      else:
         in_H = False

   # identify residues in following two helices
   after_helix = defaultdict(list)
   h_index = 0
   in_H = False
   for i in range(H_end + 1, resT):
      if ss[i] == "H":
         if not in_H:
            h_index = h_index + 1
            in_H = True
            if h_index == 4:
               break
         after_helix[h_index].append(i)
         helix_id[i] = h_index

      else:
         in_H = False
   return in_helix, before_helix, after_helix, helix_id
