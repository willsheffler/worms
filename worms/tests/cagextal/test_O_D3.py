from worms.search.result import ResultTable
import worms as w
import os, pickle

os.nice(999)

def save_test_stuff(tag, stuff):
   fname = w.data.get_test_path(tag) + '.pickle'
   with open(fname, 'wb') as out:
      pickle.dump(stuff, out)

def get_test_stuff(tag):
   fname = w.data.get_test_path(tag) + '.pickle'
   with open(fname, 'rb') as inp:
      return pickle.load(inp)

def test_worms_main():
   argv = ['@' + w.data.get_test_path('8-11-21_restricted_cage/input/cagextal_O_D3.flags')]
   print('calling worms main', argv)
   criteria_list, kw = w.app.main.build_worms_setup_from_cli_args(argv)
   kw['return_raw_result'] = True
   log = w.app.main.worms_main2(criteria_list, kw)

   results = [x for x in log if isinstance(x, ResultTable)]

   # save_test_stuff('test_O_D3_results', results)
   refresults = get_test_stuff('test_O_D3_results')
   print('nresults', len(results), 'nrefresults', len(refresults))
   assert len(results) == len(refresults)

   for a, b in zip(results, refresults):
      assert a.close_without_stats(b)

   print('test_worms_main DONE')

   # for r in results:
   #    print(type(r), r.idx.shape)
   #    pickled_result = pickle.dumps(r)
   #    r2 = pickle.loads(pickled_result)
   #    assert r == r2

if __name__ == '__main__':
   test_worms_main()
