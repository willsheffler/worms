from logging import info
import argparse

import os

from worms.cli import get_cli_args
from worms.database import CachingBBlockDB

import pyrosetta

if __name__ == "__main__":

   info("sent to info")

   args = get_cli_args(dbfiles=[""], cachedirs=[""], read_new_pdbs=False, parallel=0)
   if args.parallel == 0:
      args.parallel = 1

   pyrosetta.init("-mute all -ignore_unrecognized_res")

   try:
      pp = CachingBBlockDB(
         dbfiles=args.dbfiles,
         nprocs=args.parallel,
         cachedirs=args.cachedirs,
         read_new_pdbs=args.read_new_pdbs,
         lazy=False,
      )
      print("new entries", pp.n_new_entries)
      print("missing entries", pp.n_missing_entries)
      print("total entries", len(pp._bblock_cache))
   except AssertionError as e:
      print(e)
