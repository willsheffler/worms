from pyrosetta.rosetta.utility import vector1_unsigned_long
from pyrosetta.rosetta.core import *
from pyrosetta.rosetta.core.scoring.dssp import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta import *
from pyrosetta import *
from collections import namedtuple
from pyrosetta.rosetta.core.select.residue_selector import *
import numpy as np
from pyrosetta import rosetta
from pyrosetta import toolbox
from pyrosetta.toolbox import *
from pyrosetta.toolbox import generate_resfile
import math
import sys

init()

def PRINTDBG(msg):
   # print(msg)
   return

PoseMap = namedtuple("PoseMap", ["pose", "original_ranges", "final_ranges"])

class ContactAnalyzer:
   def __init__(self):
      assert True  # wtf?

   # Calculate the positions affected by splicing and output those to a resfile. The name defaults to whatever is stored
   # in the PDB-info of the pose if a filename is not provided. That falls back to amino acid sequence if the pose doesn't
   # contain a name.
   def write_resfile(
         self,
         output_pose,
         close_to_junction,
         input_pose_maps=[],
         resfile_name=None,
         operation="ANYAA",
   ):
      affected_positions, lost_contact_positions, new_contact_positions = self.get_affected_positions(
         output_pose, input_pose_maps)

      # Generate a dictionary of residue index to operation where operation is a resfile command. This is what
      # the resfile API requires if you want to provide a list of designable residues rather than a list of
      # NOT-designable residues
      operation_table = {}

      PRINTDBG("AFFECTED POSITIONS %s" % (affected_positions))
      PRINTDBG("CLOSE TO JUNCTION %s" % (close_to_junction))
      for i in affected_positions:
         if i in close_to_junction:
            operation_table[i] = operation

      resfile_name = (output_pose.pdb_info().name() + ".resfile" if resfile_name is None else
                      resfile_name.replace(".resfile", "") + ".resfile")

      # Write the file
      pyrosetta.toolbox.generate_resfile.generate_resfile_from_pose(
         output_pose,
         resfile_name,
         pack=True,
         design=True,
         freeze=[],
         specific=operation_table,
      )

   def get_affected_positions(self, output_pose, input_pose_maps=[]):

      output_pose.update_residue_neighbors()

      # The list of positions whose environment has changed and should be designed (lost contacts, new contacts)
      # The indices correspond to the post fusion pose
      modified_residue_positions = []
      lost_contact_positions = []
      new_contact_positions = []
      # Determine which residues have lost native contacts on account of fusion
      for input_pose_map in input_pose_maps:
         # validate user input
         self.validate_range_inputs(input_pose_map)
         input_pose_map.pose.update_residue_neighbors()
         lost_contacts = self.get_lost_contacts(input_pose_map)
         # print('lost contacts:',lost_contacts)
         modified_residue_positions.extend(lost_contacts)

      PRINTDBG("designable due to lost contacts:" + str(modified_residue_positions))
      lost_contact_positions = modified_residue_positions
      # Determine which positions make new contacts (clash/contact) or potential contacts (distance cutoff)
      # on account of fusion. All of these contacts are between sections that were originally in different poses.
      for input_pose_map in input_pose_maps:
         new_contacts = self.get_new_contacts(output_pose, input_pose_map.final_ranges)
         modified_residue_positions.extend(new_contacts)
         new_contact_positions.extend(new_contacts)
         PRINTDBG("designable due to new contacts:" + str(new_contacts))

      list.sort(modified_residue_positions)

      PRINTDBG("affected positions = " + str(modified_residue_positions))
      return modified_residue_positions, new_contact_positions, lost_contact_positions

   # Return the indices in the post-fusion pose that were previously interacting with subpose residues that have
   # been removed through fusion
   def get_lost_contacts(self, input_pose_map):
      # Generate a selector containining residues that remain in the final structure and a conversion from
      # pre to post-fusion indices
      post_fusion_index_lookup = {}
      remaining_selector = ResidueIndexSelector()
      for original_range, final_range in zip(input_pose_map.original_ranges,
                                             input_pose_map.final_ranges):
         for index in range(original_range[0], original_range[1] + 1):
            remaining_selector.append_index(index)
            post_fusion_index_lookup[index] = (index - original_range[0] + final_range[0])
      remaining_indices = remaining_selector.apply(input_pose_map.pose)
      PRINTDBG("lost contacts selection - remaining indices:" + str(remaining_indices))

      # Invert the selection to identify all the removed indices
      removed_selector = NotResidueSelector(remaining_selector)
      removed_indices = removed_selector.apply(input_pose_map.pose)
      PRINTDBG("lost contacts selection - removed indices:" + str(removed_indices))

      # Determine contacts of the removed residues, i.e. residues whose environment has changed
      affected_selector = CloseContactResidueSelector()
      affected_selector.central_residue_group_selector(removed_selector)
      PRINTDBG("original threshold = %0.2f" % affected_selector.threshold())
      affected_selector.threshold(4)
      PRINTDBG("modified threshold = %0.2f" % affected_selector.threshold())
      affected_indices = affected_selector.apply(input_pose_map.pose)
      PRINTDBG("lost contacts selection - nearby to removed indices:" + str(affected_indices))

      # Identify those contacts that are in the final structure but lost a contact and reindex them to the final structure
      lost_contacts = [
         i for i in range(1,
                          input_pose_map.pose.size() + 1)
         if affected_indices[i] and remaining_indices[i]
      ]
      lost_contacts_post_fusion = [post_fusion_index_lookup[i] for i in lost_contacts]
      PRINTDBG("lost contacts selection (result) - nearby to removed indices and remaining:" +
               str(lost_contacts_post_fusion))

      return lost_contacts_post_fusion

   # Return the indices in the post-fusion pose that belong to a section and make contacts outside of that section
   def get_new_contacts(self, post_fusion_pose, subpose_final_ranges):
      # Select the subset of residues in the final pose that come from a subpose
      #        print(post_fusion_pose.size(),'size of final pose')
      subpose_selector = ResidueIndexSelector()
      for subpose_range in subpose_final_ranges:
         PRINTDBG("subpose_range: %s" % subpose_range)
         for index in range(subpose_range[0], subpose_range[1] + 1):
            subpose_selector.append_index(index)
      subpose_indices = subpose_selector.apply(post_fusion_pose)
      PRINTDBG("new contacts selection - one subpose:" + str(subpose_indices))

      # Determine their contacts to residues that came from other subposes
      neighborhood_selector = NeighborhoodResidueSelector(
         subpose_indices, 8, False)  # In the neighborhood of the subpose but NOT the subpose
      # neighborhood_selector.set_distance(8)
      # neighborhood_selector.set_focus(subpose_indices)
      # neighborhood_selector.set_include_focus_in_subset(False)
      neighborhood_indices = neighborhood_selector.apply(post_fusion_pose)
      PRINTDBG("new contacts selection - neighborhood of subpose:" + str(neighborhood_indices))

      # Convert vector1<bool> to list of True indices
      result = [
         index for index in range(1,
                                  len(neighborhood_indices) + 1) if neighborhood_indices[index]
      ]
      PRINTDBG("new contacts selection (result) - ranges in final pose:" +
               str(neighborhood_indices))
      return result

   def validate_range_inputs(self, input_map):
      original_ranges = input_map.original_ranges
      PRINTDBG("original: %s" % original_ranges)
      final_ranges = input_map.final_ranges
      assert original_ranges is not None and final_ranges is not None
      assert len(original_ranges) == len(final_ranges)
      for i in range(0, len(original_ranges)):
         original_residue_count = original_ranges[i][1] - original_ranges[i][0]
         final_residue_count = final_ranges[i][1] - final_ranges[i][0]
         assert original_residue_count > 0 and final_residue_count > 0
         assert original_residue_count == final_residue_count
         assert original_ranges[i][0] > 0 and final_ranges[i][0] > 0
         assert original_ranges[i][1] <= input_map.pose.size()
