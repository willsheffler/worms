import sys
import itertools as it
import numpy as np
from worms.app import parse_args
from worms.edge import splice_metrics_pair, _ires_from_conn
from worms.bblock import bb_splice_res


def filter_audit():
    print('filter_audit')
    print(sys.argv)
    args = sys.argv[1:]
    if not args:
        args += '--geometry Null()'.split()
        args += '--bbconn _N het C_ het'.split()
        args += '--dbfiles worms/data/master_database_generation2.json'.split()
    crit, kw = parse_args(args)
    bbdb, spdb = kw['db']

    bbsN = bbdb.query('Het:C')
    bbsC = bbdb.query('Het:N')
    print('len(bbsN)', len(bbsN), 'len(bbsC)', len(bbsC))

    for bb1, bb2 in it.product(bbsN, bbsC):
        rms, nclash, ncontact = splice_metrics_pair(
            bb1,
            bb2,
            kw['splice_max_rms'],
            kw['splice_clash_d2'],
            kw['splice_contact_d2'],
            kw['splice_rms_range'],
            kw['splice_clash_contact_range'],
            skip_on_fail=False
        )
        print('splices shape', rms.shape)
        splice_res_c = bb_splice_res(bb1, dirn=1)
        splice_res_n = bb_splice_res(bb2, dirn=0)
        assert np.all(_ires_from_conn(bb1.connections, 1) == splice_res_c)
        assert np.all(_ires_from_conn(bb2.connections, 0) == splice_res_n)
        for i in range(rms.shape[0]):
            for j in range(rms.shape[1]):
                build pose and run_db_filters?
        break

    # (jstr, jstr1, filt, grade, sp, mc, mcnh, mhc, nc, ncnh,
    # nhc) = run_db_filters(
    # db, criteria, ssdag, iresult, result.idx[iresult], pose, prov, **kw
    # )


filter_audit()
