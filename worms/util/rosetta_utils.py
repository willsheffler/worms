'''
little rosetta pose related utils
'''

import functools
import numpy as np
from deferred_import import deferred_import

ros = deferred_import('pyrosetta.rosetta')

def numpy_stub_from_rosetta_stub(rosstub):
   npstub = np.zeros((4, 4))
   for i in range(3):
      npstub[..., i, 3] = rosstub.v[i]
      for j in range(3):
         npstub[..., i, j] = rosstub.M(i + 1, j + 1)
   npstub[..., 3, 3] = 1.0
   return npstub

def rosetta_stub_from_numpy_stub(npstub, ros):
   rosstub = ros.core.kinematics.Stub()
   rosstub.M.xx = npstub[0, 0]
   rosstub.M.xy = npstub[0, 1]
   rosstub.M.xz = npstub[0, 2]
   rosstub.M.yx = npstub[1, 0]
   rosstub.M.yy = npstub[1, 1]
   rosstub.M.yz = npstub[1, 2]
   rosstub.M.zx = npstub[2, 0]
   rosstub.M.zy = npstub[2, 1]
   rosstub.M.zz = npstub[2, 2]
   rosstub.v.x = npstub[0, 3]
   rosstub.v.y = npstub[1, 3]
   rosstub.v.z = npstub[2, 3]
   return rosstub

def get_bb_stubs(ros, pose, which_resi=None):
   if which_resi is None:
      which_resi = list(range(1, pose.size() + 1))
   npstubs, n_ca_c = [], []
   for ir in which_resi:
      r = pose.residue(ir)
      if not r.is_protein():
         raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
      n, ca, c = r.xyz("N"), r.xyz("CA"), r.xyz("C")
      ros_stub = ros.core.kinematics.Stub(ca, n, ca, c)
      npstubs.append(numpy_stub_from_rosetta_stub(ros_stub))
      n_ca_c.append(np.array([[n.x, n.y, n.z], [ca.x, ca.y, ca.z], [c.x, c.y, c.z]]))
   return np.stack(npstubs).astype("f8"), np.stack(n_ca_c).astype("f8")

def get_bb_coords(pose, which_resi=None):
   if which_resi is None:
      which_resi = list(range(1, pose.size() + 1))
   n_ca_c = []
   for ir in which_resi:
      r = pose.residue(ir)
      if not r.is_protein():
         raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
      n, ca, c = r.xyz("N"), r.xyz("CA"), r.xyz("C")
      n_ca_c.append(np.array([[n.x, n.y, n.z, 1], [ca.x, ca.y, ca.z, 1], [c.x, c.y, c.z, 1]]))
   return np.stack(n_ca_c).astype("f8")

def get_cb_coords(pose, which_resi=None):
   if which_resi is None:
      which_resi = list(range(1, pose.size() + 1))
   cbs = []
   for ir in which_resi:
      r = pose.residue(ir)
      if not r.is_protein():
         raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
      if r.has("CB"):
         cb = r.xyz("CB")
      else:
         cb = r.xyz("CA")
      cbs.append(np.array([cb.x, cb.y, cb.z, 1]))
   return np.stack(cbs).astype("f8")

def get_chain_bounds(pose):
   ch = np.array([pose.chain(i + 1) for i in range(len(pose))])
   chains = list()
   for i in range(ch[-1]):
      chains.append((np.sum(ch <= i), np.sum(ch <= i + 1)))
   assert chains[0][0] == 0
   assert chains[-1][-1] == len(pose)
   return chains

def pose_bounds(pose, lb, ub):
   if ub < 0:
      ub = len(pose) + 1 + ub
   if lb < 1 or ub > len(pose):
      raise ValueError("lb/ub " + str(lb) + "/" + str(ub) + " out of bounds for pose with len " +
                       str(len(pose)))
   return lb, ub

def subpose(ros, pose, lb, ub=-1):
   lb, ub = pose_bounds(pose, lb, ub)
   p = ros.core.pose.Pose()
   ros.core.pose.append_subpose_to_pose(p, pose, lb, ub)
   return p

def xform_pose(ros, xform, pose, lb=1, ub=-1):
   lb, ub = pose_bounds(pose, lb, ub)
   xform = rosetta_stub_from_numpy_stub(xform.reshape(4, 4))
   ros.protocols.sic_dock.xform_pose(pose, xform, lb, ub)

def splice_poses(ros, pose_c, pose_n, ires_c, ires_n):
   new = subpose(pose_c, 1, ires_c)
   ros.core.pose.append_subpose_to_pose(new, pose_n, ires_n + 1, len(pose_n))

   stubs_ref, _ = get_bb_stubs(pose_c, [ires_c])
   stubs_move, _ = get_bb_stubs(pose_n, [ires_n])
   xalign = stubs_ref @ np.linalg.inv(stubs_move)
   xform_pose(xalign, new, ires_c + 1)

   return new

def worst_CN_connect(ros, p):
   for ir in range(1, len(p)):
      worst = 0
      if (p.residue(ir).is_protein() and p.residue(ir + 1).is_protein()
          and not (ros.core.pose.is_upper_terminus(p, ir)
                   or ros.core.pose.is_lower_terminus(p, ir + 1))):
         dist = p.residue(ir).xyz("C").distance(p.residue(ir + 1).xyz("N"))
         worst = max(abs(dist - 1.32), worst)
   return worst

def no_overlapping_adjacent_residues(p):
   for ir in range(1, len(p)):
      if p.residue(ir).is_protein() and p.residue(ir + 1).is_protein():
         dist = p.residue(ir).xyz("CA").distance(p.residue(ir + 1).xyz("CA"))
         if dist < 0.1:
            return False
   return True

def no_overlapping_residues(p):
   for ir in range(1, len(p) + 1):
      if not p.residue(ir).is_protein():
         continue
      for jr in range(1, ir):
         if not p.residue(jr).is_protein():
            continue
         dist = p.residue(ir).xyz("CA").distance(p.residue(jr).xyz("CA"))
         if dist < 0.5:
            return False
   return True

def trim_pose(ros, pose, resid, direction, pad=0):
   """trim end of pose from direction, leaving <=pad residues beyond resid

    Args:
        pose (TYPE): Description
        resid (TYPE): Description
        direction (TYPE): Description
        pad (int, optional): Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
   if direction not in "NC":
      raise ValueError("direction must be 'N' or 'C'")
   if not 0 < resid <= len(pose):
      raise ValueError("resid %i out of bounds %i" % (resid, len(pose)))
   p = ros.core.pose.Pose()
   if direction == "N":
      lb, ub = max(resid - pad, 1), len(pose)
   elif direction == "C":
      lb, ub = 1, min(resid + pad, len(pose))
   # print('_trim_pose lbub', lb, ub, 'len', len(pose), 'resid', resid)
   ros.core.pose.append_subpose_to_pose(p, pose, lb, ub)
   return p, lb, ub

# def fix_bb_h(pose, ires):
#     r = pose.residue(ires)
#     if r.name3() == 'PRO': return
#     ih = r.atom_index('H')
#     crd = r.build_atom_ideal(ih, pose.conformation())
#     pose.set_xyz(ros.core.id.AtomID(ih, ires), crd)

# def fix_bb_o(pose, ires):
#     r = pose.residue(ires)
#     io = r.atom_index('O')
#     crd = r.build_atom_ideal(io, pose.conformation())
#     pose.set_xyz(ros.core.id.AtomID(io, ires), crd)

def symfile_path(name):
   path, _ = os.path.split(__file__)
   return os.path.join(path, "rosetta_symdef", name + ".sym")

@functools.lru_cache()
def get_symfile_contents(name):
   print(f'reading symdef from {symfile_path(name)}')
   with open(symfile_path(name)) as f:
      return f.read()

@functools.lru_cache()
def get_symdata(ros, name):
   if name is None:
      return None
   ss = ros.std.stringstream(get_symfile_contents(name))
   d = ros.core.conformation.symmetry.SymmData()
   d.read_symmetry_data_from_stream(ss)
   return d

def get_symdata_modified(ros, name, string_substitutions=None, scale_positions=None):
   symfilestr = get_symfile_contents(name)
   if scale_positions is not None:
      if string_substitutions is None:
         string_substitutions = dict()
      for line in symfilestr.splitlines():
         if not line.startswith("xyz"):
            continue
         if isinstance(scale_positions, np.ndarray):
            for posstr in re.split("\s+", line)[-3:]:
               tmp = np.array([float(x) for x in posstr.split(",")])
               x, y, z = tmp * scale_positions
               string_substitutions[posstr] = "%f,%f,%f" % (x, y, z)
         else:
            posstr = re.split("\s+", line)[-1]
            x, y, z = [float(x) * scale_positions for x in posstr.split(",")]
            string_substitutions[posstr] = "%f,%f,%f" % (x, y, z)
   if string_substitutions is not None:
      for k, v in string_substitutions.items():
         symfilestr = symfilestr.replace(k, v)
   ss = ros.std.stringstream(symfilestr)
   d = ros.core.conformation.symmetry.SymmData()
   d.read_symmetry_data_from_stream(ss)
   return d, symfilestr

def infer_cyclic_symmetry(pose):
   raise NotImplementedError

def residue_coords(p, ir, n=3):
   crd = (p.residue(ir).xyz(i) for i in range(1, n + 1))
   return np.stack([c.x, c.y, c.z, 1] for c in crd)

def residue_sym_err(p, ang, ir, jr, n=1, axis=[0, 0, 1], verbose=0):
   mxdist = 0
   for i in range(n):
      xyz0 = residue_coords(p, ir + i)
      xyz1 = residue_coords(p, jr + i)
      xyz3 = hrot(axis, ang) @ xyz1.T
      xyz4 = hrot(axis, -ang) @ xyz1.T
      if verbose:
         print(i, xyz0)
         print(i, xyz1)
         print(i, xyz3.T)
         print(i, xyz4.T)
         print()
      mxdist = max(
         mxdist,
         min(
            np.max(np.sum((xyz0 - xyz3.T)**2, axis=1)),
            np.max(np.sum((xyz0 - xyz4.T)**2, axis=1)),
         ),
      )
   return np.sqrt(mxdist)
