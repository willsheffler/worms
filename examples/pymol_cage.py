from pymol import cmd

c3axis = Vec(0.000000, 0.356822, 0.934172)
c5axis = Vec(0.525731, -0.000000, 0.850651)
xform("c3", alignvector(Uz, c3axis))
xform("c5", alignvector(Uz, c5axis))
trans("c3", 500 * c3axis)
trans("c5", 500 * c5axis)
