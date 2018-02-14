import tempfile
import numpy as np
from collections import defaultdict
import homog
from . import util
try:
    from pymol import cmd
    from pymol import cgo
except:
    print('pymol not available!')


def is_rosetta_pose(to_show):
    try:
        from pyrosetta import rosetta
        return isinstance(to_show, rosetta.core.pose.Pose)
    except ImportError:
        return False


def pymol_load_pose(pose, name):
    from pymol import cmd
    tmpdir = tempfile.mkdtemp()
    fname = tmpdir + '/' + name + '.pdb'
    pose.dump_pdb(fname)
    cmd.load(fname)


def pymol_xform(name, xform):
    from pymol import cmd
    assert name in cmd.get_object_list()
    cmd.transform_object(name, xform.flatten())


def pymol_load(to_show, state=None, name=None, **kw):
    if isinstance(to_show, list):
        for t in to_show:
            state = pymol_load(t, state)
    elif isinstance(to_show, dict):
        assert 'pose' in to_show
        state = pymol_load(to_show['pose'], state)
        pymol_xform(to_show['position'], state['last_obj'])
    elif is_rosetta_pose(to_show):
        name = name or 'rif_thing'
        state['seenit'][name] += 1
        name += '_%i' % state['seenit'][name]
        pymol_load_pose(to_show, name)
        state['last_obj'] = name
    elif isinstance(to_show, np.ndarray):
        showsegment(to_show, [0, 0, 0], **kw)
    else:
        raise NotImplementedError(
            "don't know how to show " + str(type(to_show)))
    return state

showme_state = dict(launched=0, seenit=defaultdict(lambda: -1))


def showme_pymol(what, headless=False, block=False, **kw):
    import pymol
    pymol.pymol_argv = ['pymol']
    if headless:
        pymol.pymol_argv = ['pymol', '-c']
    if not showme_state['launched']:
        pymol.finish_launching()
        showme_state['launched'] = 1
    from pymol import cmd
    r = pymol_load(what, showme_state, **kw)
    # cmd.set('internal_gui_width', '20')
    import time
    while block:
        time.sleep(1)
    return r


def showme(*args, how='pymol', **kw):
    if how == 'pymol':
        return showme_pymol(*args, **kw)
    else:
        raise NotImplementedError('showme how="%s" not implemented' % how)


numcom = 0
numvec = 0
numray = 0
numline = 0
numseg = 0


def showcom(sel="all"):
    global numcom
    c = com(sel)
    print("Center of mass: ", c)
    cgo = [pymol.cgo.COLOR, 1.0, 1.0, 1.0, cgo.SPHERE,
           c[0], c[1], c[2], 1.0]  # white sphere with 3A radius
    cmd.load_cgo(cgo, "com%i" % numcom)
    numcom += 1


def cgo_sphere(c, r=1, col=(1, 1, 1)):
    # white sphere with 3A radius
    return [cgo.COLOR, col[0], col[1], col[2], cgo.SPHERE, c[0], c[1], c[2], r]


def showsphere(c, r=1, col=(1, 1, 1), lbl=''):
    v = cmd.get_view()
    if not lbl:
        global numvec
        lbl = "sphere%i" % numvec
        numvec += 1
    mycgo = cgo_sphere(c=c, r=r, col=col)
    cmd.load_cgo(mycgo, lbl)
    cmd.set_view(v)


def showvecfrompoint(a, c, col=(1, 1, 1), lbl=''):
    if not lbl:
        global numray
        lbl = "ray%i" % numray
        numray += 1
    cmd.delete(lbl)
    v = cmd.get_view()
    OBJ = [
        cgo.BEGIN, cgo.LINES,
        cgo.COLOR, col[0], col[1], col[2],
        cgo.VERTEX, c[0], c[1], c[2],
        cgo.VERTEX, c[0] + a[0], c[1] + a[1], c[2] + a[2],
        cgo.END
    ]
    cmd.load_cgo(OBJ, lbl)
    # cmd.load_cgo([cgo.COLOR, col[0],col[1],col[2],
    #             cgo.SPHERE,   c[0],       c[1],       c[2],    0.08,
    #             cgo.CYLINDER, c[0],       c[1],       c[2],
    #                       c[0] + a[0], c[1] + a[1], c[2] + a[2], 0.02,
    #               col[0],col[1],col[2],col[0],col[1],col[2],], lbl)
    cmd.set_view(v)


def cgo_segment(c1, c2, col=(1, 1, 1)):
    OBJ = [
        cgo.BEGIN, cgo.LINES,
        cgo.COLOR, col[0], col[1], col[2],
        cgo.VERTEX, c1[0], c1[1], c1[2],
        cgo.VERTEX, c2[0], c2[1], c2[2],
        cgo.END
    ]
    # cmd.load_cgo([cgo.COLOR, col[0],col[1],col[2],
    #             cgo.CYLINDER, c1[0],     c1[1],     c1[2],
    #                           c2[0],     c2[1],     c2[2], 0.02,
    #               col[0],col[1],col[2],col[0],col[1],col[2],], lbl)
    return OBJ


def showsegment(c1, c2, col=(1, 1, 1), lbl=''):
    if not lbl:
        global numseg
        lbl = "seg%i" % numseg
        numseg += 1
    cmd.delete(lbl)
    v = cmd.get_view()
    cmd.load_cgo(cgo_segment(c1=c1, c2=c2, col=col), lbl)
    # cmd.load_cgo([cgo.COLOR, col[0],col[1],col[2],
    #             cgo.CYLINDER, c1[0],     c1[1],     c1[2],
    #                           c2[0],     c2[1],     c2[2], 0.02,
    #               col[0],col[1],col[2],col[0],col[1],col[2],], lbl)
    cmd.set_view(v)


def cgo_cyl(c1, c2, r, col=(1, 1, 1), col2=None):
    if not col2:
        col2 = col
    return [  # cgo.COLOR, col[0],col[1],col[2],
        cgo.CYLINDER, c1[0], c1[1], c1[2],
        c2[0], c2[1], c2[2], r,
        col[0], col[1], col[2], col2[0], col2[1], col2[2], ]


def showcyl(c1, c2, r, col=(1, 1, 1), col2=None, lbl=''):
    if not lbl:
        global numseg
        lbl = "seg%i" % numseg
        numseg += 1
    cmd.delete(lbl)
    v = cmd.get_view()
    cmd.load_cgo(cgo_cyl(c1=c1, c2=c2, r=r, col=col, col2=col2), lbl)
    cmd.set_view(v)


def showline(a, c, col=(1, 1, 1), lbl=''):
    if not lbl:
        global numline
        lbl = "line%i" % numline
        numline += 1
    cmd.delete(lbl)
    v = cmd.get_view()
    OBJ = [
        cgo.BEGIN, cgo.LINES,
        cgo.COLOR, col[0], col[1], col[2],
        cgo.VERTEX, c[0] - a[0], c[1] - a[1], c[2] - a[2],
        cgo.VERTEX, c[0] + a[0], c[1] + a[1], c[2] + a[2],
        cgo.END
    ]
    cmd.load_cgo(OBJ, lbl)
    cmd.set_view(v)


def cgo_lineabs(a, c, col=(1, 1, 1)):
    return [
        cgo.BEGIN, cgo.LINES,
        cgo.COLOR, col[0], col[1], col[2],
        cgo.VERTEX, c[0], c[1], c[2],
        cgo.VERTEX, a[0], a[1], a[2],
        cgo.END
    ]


def showlineabs(a, c, col=(1, 1, 1), lbl=''):
    if not lbl:
        global numline
        lbl = "line%i" % numline
        numline += 1
    cmd.delete(lbl)
    v = cmd.get_view()
    cgo = cgo_lineabs(a, c, col)
    cmd.load_cgo(cgo, lbl)
    cmd.set_view(v)


def show_with_axis(worms, idx=0):
    pose = worms.pose(idx, align=0, end=1)
    x_from = worms.positions[idx][worms.criteria.from_seg]
    x_to = worms.positions[idx][worms.criteria.to_seg]
    x = x_to @ np.linalg.inv(x_from)
    axis, ang, cen = homog.axis_ang_cen_of(x)
    np.set_printoptions(precision=20)
    print(x)
    print(axis)
    print(ang)
    print(cen)
    axis *= 100
    showme(pose, name='unit')
    util.xform_pose(x, pose)
    showme(pose, name='sym1')
    util.xform_pose(x, pose)
    showme(pose, name='sym2')
    showline(axis, cen)
    showsphere(cen)


def show_with_z_axes(worms, idx=0, only_connected=0, **kw):
    pose = worms.pose(idx, align=0, end=1, only_connected=only_connected, **kw)
    x_from = worms.positions[idx][worms.criteria.from_seg]
    x_to = worms.positions[idx][worms.criteria.to_seg]
    cen1 = x_from[..., :, 3]
    cen2 = x_to[..., :, 3]
    axis1 = x_from[..., :, 2] * 100
    axis2 = x_to[..., :, 2] * 100
    showme(pose)
    import pymol
    pymol.finish_launching()
    showline(axis1, cen1)
    showsphere(cen1)
    showline(axis2, cen2)
    showsphere(cen2)
