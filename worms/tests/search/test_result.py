import pickle

def test_remove_dups():
   pass

if __name__ == '__main__':
   with open(
         '/home/sheffler/src/worms_unittests/worms/data/test_cases/test_cagextal_O_D3/2021_10_19_17_47_35/ResultTables.pickle',
         'rb') as inp:
      rt = pickle.load(inp)
   print(rt)