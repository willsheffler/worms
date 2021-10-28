import sys, os, pickle, inspect
import worms

# from worms.search.result import ResultTable

def setup_test_databases(criteria, **kw):
   kw = worms.Bunch(kw)

   bbdb = worms.data.get_testing_database('main_test_database')
   #      # kw.database = pickle.load(inp)
   #      # bbdb = worms.database.BBlockDB(**kw)
   #      # bbdb.load_all_bblocks()
   #
   #      # pdb = list(bbdb._dictdb.keys())[0]
   #      # bb = bbdb.bblock(pdb)
   #      # print(bbdb)

   spdb = worms.database.SpliceDB(**kw)
   print(spdb)
   worms.data.dump(spdb, 'spdb_before')

   dbpair = worms.database.Databases(bbdb, spdb)
   # print(dbpair.splicedb)
   # worms.ssdag.simple_search_dag(criteria, dbpair, precache_only=True, **kw)
   # print(dbpair.splicedb)

   return dbpair

def setup_this_testfunc(testname='auto'):
   if testname == 'auto':
      testname = [fn[3] for fn in inspect.stack()[1:] if fn[3].startswith('test_')][0]
   print('SETTING UP TEST', testname)
   argv = ['@' + worms.data.test_file_path(f'{testname}/config/{testname}.flags')]

   criteria_list, kw = worms.cli.build_worms_setup_from_cli_args(argv, construct_databases=True)
   assert len(criteria_list) == 1
   criteria = criteria_list[0]

   # kw.database = setup_test_databases(criteria, **kw)
   print(kw.database.bblockdb)
   print(kw.database.splicedb)

   # assert 0

   return testname, criteria_list, kw

def test_cagextal_O_D3():
   testname, criteria_list, kw = setup_this_testfunc()

   print('calling worms main', criteria_list)
   # kw.return_raw_result = True
   kw.return_raw_result = False
   # kw.only_merge_bblocks = [2]
   rundata = worms.app.main.construct_global_ssdag_and_run(criteria_list, kw)
   assert 0

   # print(kw.database.splicedb)
   # assert 0

   assert rundata.log
   newresults = [x for x in rundata.log if isinstance(x, worms.search.result.ResultTable)]

   # record_new_testresults = False
   record_new_testresults = True
   fail = False
   refpath, refresults = worms.data.get_latest_resulttables(testname, candidates_ok=False)
   if refresults is None:
      fail = True
   else:
      print('found refresults at', refpath)
      print('nresults', len(newresults), 'nrefresults', len(refresults))
      fail = fail or len(newresults) != len(refresults)
      if not fail:
         for a, b in zip(newresults, refresults):
            fail = fail or a.approx_equal(b)

   if fail:
      record_new_testresults = True
      print('********************** FAIL! ********************************')
   else:
      print('********************** PASS! ********************************')

   # if record_new_testresults:
   if True:

      new_test_dir = worms.data.make_timestamped_test_dir(testname, candidate=True)
      new_resulttable_file = worms.data.get_latest_resulttables_path(testname, candidates_ok=True)

      print('new_test_dir', new_test_dir)
      print('INSPECT THIS NEW TEST DIR AND REMOVE IF INCORRECT')
      print('new_resulttable_file', new_resulttable_file)

      with open(new_resulttable_file, 'wb') as out:
         pickle.dump(newresults, out)

      for outfrompose in (True, False):
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

def tmp_check_pdbout():
   testname = 'test_cagextal_O_D3'
   refpath, refresults = worms.data.get_latest_resulttables(testname, candidates_ok=True)
   tmp = worms.load(
      '/home/sheffler/src/worms_unittests/worms/data/test_cases/test_cagextal_O_D3/2021_10_22_18_47_51_CANDIDATE/reference_results.pickle'
   )

   print(refpath)
   print(refresults)

if __name__ == '__main__':
   test_cagextal_O_D3()
# tmp_check_pdbout()
