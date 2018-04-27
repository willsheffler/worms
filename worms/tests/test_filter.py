import pytest

try:
    import pyrosetta
    from pyrosetta import rosetta as ros
    from ..filter import *
    HAVE_PYROSETTA = True
    try:
        import pyrosetta.distributed
        HAVE_PYROSETTA_DISTRIBUTED = True
    except ImportError:
        HAVE_PYROSETTA_DISTRIBUTED = False
except ImportError:
    HAVE_PYROSETTA = HAVE_PYROSETTA_DISTRIBUTED = False

only_if_pyrosetta = pytest.mark.skipif('not HAVE_PYROSETTA')


@pytest.mark.skip
@only_if_pyrosetta
def test_BakerFilter(db_asu_pose, hetC2A_pose, hetC2B_pose):
    FW = BakerFilter(
        score0_cutoff=1.0,
        num_contact_threshold=40,
        num_contact_no_helix_threshold=3,
        n_helix_contacted_threshold=3,
        superimpose_length=9,
        superimpose_rmsd=0.7,
        pose_info_all=None)

    junct_res = 170
    src_pose_1_jct_res = 170  # 397 in actual numbering
    src_pose_2_jct_res = 54
    mode = ros.core.chemical.type_set_mode_from_string("centroid")
    ros.core.util.switch_to_residue_type_set(db_asu_pose, mode)
    db_asu_pose.update_residue_neighbors()
    ros.core.util.switch_to_residue_type_set(hetC2B_pose, mode)
    ros.core.util.switch_to_residue_type_set(hetC2A_pose, mode)
    result = FW.filter_worm(
        db_asu_pose,
        junct_res,
        src_pose_1_jct_res,
        src_pose_2_jct_res,
        src_pose_N=hetC2B_pose,
        src_pose_C=hetC2A_pose)
    assert result == 'AAAA'
