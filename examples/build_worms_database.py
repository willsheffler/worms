from worms.database import BBlockDB
from logging import info
import argparse
import pyrosetta
import os

if __name__ == '__main__':

    info('sent to info')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dbfiles', type=str, nargs='+', dest='dbfiles')
    parser.add_argument('--nprocs', type=int, dest='nprocs', default=1)
    parser.add_argument(
        '--read_new_pdbs', type=bool, dest='read_new_pdbs', default=False
    )
    args = parser.parse_args()
    pyrosetta.init('-mute all -ignore_unrecognized_res')

    try:
        pp = BBlockDB(
            dbfiles=args.dbfiles,
            nprocs=args.nprocs,
            read_new_pdbs=args.read_new_pdbs,
            lazy=False,
        )
        print('new entries', pp.n_new_entries)
        print('missing entries', pp.n_missing_entries)
        print('total entries', len(pp._bblock_cache))
    except AssertionError as e:
        print(e)
    except:
        if args.read_new_pdbs:
            os.remove(os.environ['HOME'] + '/.worms/cache/lock')
        raise
