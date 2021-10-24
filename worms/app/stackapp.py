import blosc, sys

from deferred_import import deferred_import

pyrosetta = deferred_import('worms.rosetta_init')
from worms.cli import build_worms_setup_from_cli_args
from worms.ssdag import simple_search_dag
from worms.search import grow_linear

def main():
   pyrosetta.init("-mute all -beta -preserve_crystinfo --prevent_repacking")
   blosc.set_releasegil(True)

   sys.argv.extend([
      '@/home/sheffler/debug/robby_stack2/new/input/OR_L5.flags',
      '--geometry',
      'Stack(2)',
      '--dbfiles',
      '/home/sheffler/debug/robby_stack2/new/test1/tmp_0.json',
      *('--bbconn _C fc_binder NN ds_flop_het_c2 CN Monomer C_ fc_binder'.split()),
   ], )

   criteria, kw = build_worms_setup_from_cli_args(sys.argv[1:])
   assert len(criteria) is 1
   criteria = criteria[0]

   ssdag, _ = simple_search_dag(criteria, print_edge_summary=True, **kw)

   result = grow_linear(ssdag, criteria.jit_lossfunc())

   print(result)

if __name__ == '__main__':
   main()
