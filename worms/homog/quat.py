import numpy as np
from worms.homog.util import jit, guvec, float32, float64

def is_valid_quat_rot(quat):
   assert quat.shape[-1] == 4
   return np.isclose(1, np.linalg.norm(quat, axis=-1))

def quat_to_upper_half(quat):
   ineg0 = (quat[..., 0] < 0)
   ineg1 = (quat[..., 0] == 0) * (quat[..., 1] < 0)
   ineg2 = (quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] < 0)
   ineg3 = ((quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] == 0) * (quat[..., 3] < 0))
   # print(ineg0.shape)
   # print(ineg1.shape)
   # print(ineg2.shape)
   # print(ineg3.shape)
   ineg = ineg0 + ineg1 + ineg2 + ineg3
   quat = quat.copy()
   quat[ineg] = -quat[ineg]
   return quat

@jit
def kernel_quat_to_upper_half(quat, ret):
   ineg0 = (quat[0] < 0)
   ineg1 = (quat[0] == 0) * (quat[1] < 0)
   ineg2 = (quat[0] == 0) * (quat[1] == 0) * (quat[2] < 0)
   ineg3 = ((quat[0] == 0) * (quat[1] == 0) * (quat[2] == 0) * (quat[3] < 0))
   mul = -1.0 if ineg0 or ineg1 or ineg2 or ineg3 else 1.0
   for i in range(4):
      ret[i] = mul * quat[i]

@jit
def numba_quat_to_upper_half(quat):
   ret = np.empty(4, dtype=quat.dtype)
   kernel_quat_to_upper_half(quat, ret)
   return ret

def rand_quat(shape=()):
   if isinstance(shape, int): shape = (shape, )
   q = np.random.randn(*shape, 4)
   q /= np.linalg.norm(q, axis=-1)[..., np.newaxis]
   return quat_to_upper_half(q)

def rot_to_quat(xform):
   x = np.asarray(xform)
   t0, t1, t2 = x[..., 0, 0], x[..., 1, 1], x[..., 2, 2]
   tr = t0 + t1 + t2
   quat = np.empty(x.shape[:-2] + (4, ))

   case0 = tr > 0
   S0 = np.sqrt(tr[case0] + 1) * 2
   quat[case0, 0] = 0.25 * S0
   quat[case0, 1] = (x[case0, 2, 1] - x[case0, 1, 2]) / S0
   quat[case0, 2] = (x[case0, 0, 2] - x[case0, 2, 0]) / S0
   quat[case0, 3] = (x[case0, 1, 0] - x[case0, 0, 1]) / S0

   case1 = ~case0 * (t0 >= t1) * (t0 >= t2)
   S1 = np.sqrt(1.0 + x[case1, 0, 0] - x[case1, 1, 1] - x[case1, 2, 2]) * 2
   quat[case1, 0] = (x[case1, 2, 1] - x[case1, 1, 2]) / S1
   quat[case1, 1] = 0.25 * S1
   quat[case1, 2] = (x[case1, 0, 1] + x[case1, 1, 0]) / S1
   quat[case1, 3] = (x[case1, 0, 2] + x[case1, 2, 0]) / S1

   case2 = ~case0 * (t1 > t0) * (t1 >= t2)
   S2 = np.sqrt(1.0 + x[case2, 1, 1] - x[case2, 0, 0] - x[case2, 2, 2]) * 2
   quat[case2, 0] = (x[case2, 0, 2] - x[case2, 2, 0]) / S2
   quat[case2, 1] = (x[case2, 0, 1] + x[case2, 1, 0]) / S2
   quat[case2, 2] = 0.25 * S2
   quat[case2, 3] = (x[case2, 1, 2] + x[case2, 2, 1]) / S2

   case3 = ~case0 * (t2 > t0) * (t2 > t1)
   S3 = np.sqrt(1.0 + x[case3, 2, 2] - x[case3, 0, 0] - x[case3, 1, 1]) * 2
   quat[case3, 0] = (x[case3, 1, 0] - x[case3, 0, 1]) / S3
   quat[case3, 1] = (x[case3, 0, 2] + x[case3, 2, 0]) / S3
   quat[case3, 2] = (x[case3, 1, 2] + x[case3, 2, 1]) / S3
   quat[case3, 3] = 0.25 * S3

   assert (np.sum(case0) + np.sum(case1) + np.sum(case2) + np.sum(case3) == np.prod(
      xform.shape[:-2]))

   return quat_to_upper_half(quat)

xform_to_quat = rot_to_quat

@jit
def kernel_rot_to_quat(xform, quat):
   t0, t1, t2 = xform[0, 0], xform[1, 1], xform[2, 2]
   tr = t0 + t1 + t2
   if tr > 0:
      S0 = np.sqrt(tr + 1) * 2
      quat[0] = 0.25 * S0
      quat[1] = (xform[2, 1] - xform[1, 2]) / S0
      quat[2] = (xform[0, 2] - xform[2, 0]) / S0
      quat[3] = (xform[1, 0] - xform[0, 1]) / S0
   elif t0 >= t1 and t0 >= t2:
      S1 = np.sqrt(1.0 + xform[0, 0] - xform[1, 1] - xform[2, 2]) * 2
      quat[0] = (xform[2, 1] - xform[1, 2]) / S1
      quat[1] = 0.25 * S1
      quat[2] = (xform[0, 1] + xform[1, 0]) / S1
      quat[3] = (xform[0, 2] + xform[2, 0]) / S1
   elif t1 > t0 and t1 >= t2:
      S2 = np.sqrt(1.0 + xform[1, 1] - xform[0, 0] - xform[2, 2]) * 2
      quat[0] = (xform[0, 2] - xform[2, 0]) / S2
      quat[1] = (xform[0, 1] + xform[1, 0]) / S2
      quat[2] = 0.25 * S2
      quat[3] = (xform[1, 2] + xform[2, 1]) / S2
   elif t2 > t0 and t2 > t1:
      S3 = np.sqrt(1.0 + xform[2, 2] - xform[0, 0] - xform[1, 1]) * 2
      quat[0] = (xform[1, 0] - xform[0, 1]) / S3
      quat[1] = (xform[0, 2] + xform[2, 0]) / S3
      quat[2] = (xform[1, 2] + xform[2, 1]) / S3
      quat[3] = 0.25 * S3
   kernel_quat_to_upper_half(quat, quat)

@jit
def numba_rot_to_quat(xform):
   quat = np.empty(4, dtype=xform.dtype)
   kernel_rot_to_quat(xform, quat)
   return quat

# update to numba 0.52 broke this
# gu_rot_to_quat = guvec([
# (float64[:, :], float64[:]),
# (float32[:, :], float32[:]),
# ], '(n,n)->(n)', kernel_rot_to_quat)
#
def quat_to_rot(quat, dtype='f8', shape=(3, 3)):
   quat = np.asarray(quat)
   assert quat.shape[-1] == 4
   qr = quat[..., 0]
   qi = quat[..., 1]
   qj = quat[..., 2]
   qk = quat[..., 3]
   outshape = quat.shape[:-1]
   rot = np.zeros(outshape + shape, dtype=dtype)
   rot[..., 0, 0] = 1 - 2 * (qj**2 + qk**2)
   rot[..., 0, 1] = 2 * (qi * qj - qk * qr)
   rot[..., 0, 2] = 2 * (qi * qk + qj * qr)
   rot[..., 1, 0] = 2 * (qi * qj + qk * qr)
   rot[..., 1, 1] = 1 - 2 * (qi**2 + qk**2)
   rot[..., 1, 2] = 2 * (qj * qk - qi * qr)
   rot[..., 2, 0] = 2 * (qi * qk - qj * qr)
   rot[..., 2, 1] = 2 * (qj * qk + qi * qr)
   rot[..., 2, 2] = 1 - 2 * (qi**2 + qj**2)
   return rot

def quat_to_xform(quat, dtype='f8'):
   r = quat_to_rot(quat, dtype, shape=(4, 4))
   r[..., 3, 3] = 1
   return r

def quat_multiply(q, r):
   q, r = np.broadcast_arrays(q, r)
   q0, q1, q2, q3 = np.moveaxis(q, -1, 0)
   r0, r1, r2, r3 = np.moveaxis(r, -1, 0)
   assert np.all(q1 == q[..., 1])
   t = np.empty_like(q)
   t[..., 0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
   t[..., 1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
   t[..., 2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
   t[..., 3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
   return t

# update to numba 0.52 broke this
# @jit
# def kernel_quat_multiply(q, r, out):
#    q0, q1, q2, q3 = q
#    r0, r1, r2, r3 = r
#    out[0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
#    out[1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
#    out[2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
#    out[3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0

# gu_quat_multiply = guvec([(float64[:], float64[:], float64[:])], '(n),(n)->(n)',
#                          kernel_quat_multiply)

@jit
def numba_quat_multiply(q, r):
   out = np.empty(4, dtype=q.dtype)
   kernel_quat_multiply(q, r, out)
   return out
