# sfrom worms import criteria
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

   criteria_list, kw = w.app.main.build_worms_setup_from_cli_args(argv)
   assert len(criteria_list) == 1
   criteria = criteria_list[0]
   print('calling worms main', criteria)

   kw.parallel = False
   kw.return_raw_result = True
   kw.print_splice_fail_summary = False
   kw.print_info_edges_with_no_splices = False
   kw.xtal_min_cell_size = 100
   kw.xtal_max_cell_size = 9e9
   rundata = w.app.main.construct_global_ssdag_and_run(criteria_list, kw)
   assert rundata.log
   results = [x for x in rundata.log if isinstance(x, ResultTable)]
   refresults = get_test_stuff('test_O_D3_results')

   if True:
      kw.output_from_pose = False
      kw.output_prefix = 'test_worms_main_new'
      outputfiles = list()
      for ijob, result in enumerate(results):
         print('resultset', ijob, len(result.idx))
         kw.merge_bblock = ijob
         outputresult = w.output.filter_and_output_results(
            criteria_list[0],
            rundata.ssdag,
            result,
            debug_log_traces=True,
            **kw,
         )
         outputfiles.extend(outputresult.files)

      print(outputfiles)
      print('!!!!!!!!!!!!!!!!!!!!!!!!')

      outputfiles = list()
      kw.output_prefix = 'test_worms_main_old'
      for ijob, result in enumerate(refresults):
         print('resultset', ijob, len(result.idx))
         kw.merge_bblock = ijob
         outputresult = w.output.filter_and_output_results(
            criteria_list[0],
            rundata.ssdag,
            result,
            debug_log_traces=True,
            **kw,
         )
         outputfiles.extend(outputresult.files)
      print(outputfiles)

   # save_test_stuff('test_O_D3_results', results, )

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
