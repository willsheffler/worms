import sys, os, pickle, inspect
import worms

# from worms.search.result import ResultTable

def setup_test_databases(criteria, **kw):
   kw = worms.Bunch(kw)

   bbdb = worms.data.get_testing_database('main_test_database')
   #      # kw.database = pickle.load(inp)
   #      # bbdb = worms.database.NoCacheBBlockDB(**kw)
   #      # bbdb.load_all_bblocks()
   #
   #      # pdb = list(bbdb._dictdb.keys())[0]
   #      # bb = bbdb.bblock(pdb)
   #      # print(bbdb)

   spdb = worms.database.NoCacheSpliceDB(**kw)
   print(spdb)
   worms.data.dump(spdb, 'spdb_before')

   dbpair = worms.database.Databases(bbdb, spdb)
   print(dbpair.splicedb)
   worms.ssdag.simple_search_dag(criteria, dbpair, precache_only=True, **kw)
   print(dbpair.splicedb)

   assert 0

   return dbpair

def setup_this_testfunc(testname='auto'):
   if testname == 'auto':
      testname = [fn[3] for fn in inspect.stack()[1:] if fn[3].startswith('test_')][0]
   print('SETTING UP TEST', testname)
   argv = ['@' + worms.data.get_test_path(f'{testname}/config/{testname}.flags')]

   criteria_list, kw = worms.cli.build_worms_setup_from_cli_args(argv, construct_databases=False)
   assert len(criteria_list) == 1
   criteria = criteria_list[0]

   kw.database = setup_test_databases(criteria, **kw)
   print(kw.database.bblockdb)
   print(kw.database.splicedb)

   # assert 0

   return testname, criteria_list, kw

def test_cagextal_O_D3():
   testname, criteria_list, kw = setup_this_testfunc()

   print('calling worms main', criteria_list)
   kw.return_raw_result = True
   rundata = worms.app.main.construct_global_ssdag_and_run(criteria_list, kw)

   print(kw.database.splicedb)
   assert 0

   assert rundata.log
   newresults = [x for x in rundata.log if isinstance(x, worms.search.result.ResultTable)]

   fail = False
   refpath, refresults = worms.data.get_latest_resulttables(testname, candidates_ok=False)
   if refresults is None:
      record_new_testresults = True
   else:
      print('found refresults at', refpath)
      print('nresults', len(newresults), 'nrefresults', len(refresults))
      fail = fail or len(newresults) != len(refresults)
      if not fail:
         for a, b in zip(newresults, refresults):
            fail = fail or a.close_without_stats(b)

   if fail:
      print('********************** FAIL! ********************************')
   else:
      print('********************** PASS! ********************************')

   if record_new_testresults:

      new_test_dir = worms.data.make_timestamped_test_dir(testname, candidate=True)
      new_resulttable_file = worms.data.get_latest_resulttables_path(testname, candidates_ok=True)

      print('new_test_dir', new_test_dir)
      print('INSPECT THIS NEW TEST DIR AND REMOVE IF INCORRECT')
      print('new_resulttable_file', new_resulttable_file)

      with open(new_resulttable_file, 'wb') as out:
         pickle.dump(newresults, out)

      for outfrompose in (False, True):
         kw.output_from_pose = outfrompose
         posetag = 'pose_' if outfrompose else 'nopose_'
         kw.output_prefix = new_test_dir + posetag + testname
         for ijob, result in enumerate(newresults):
            print('resultset', ijob, len(result.idx))
            kw.merge_bblock = ijob
            outputresult = worms.output.filter_and_output_results(
               criteria_list[0],
               rundata.ssdag,
               result,
               debug_log_traces=True,
               **kw,
            )

   print('test_O_D3 DONE')

if __name__ == '__main__':
   test_cagextal_O_D3()
