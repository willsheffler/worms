import pickle, os, sys

import worms

import willutil as wu

def generic_integration_test(testname):

   criteria, kw = setup_testfunc(testname)
   kw.verbosity = 0

   # print('!' * 500)
   # assert not 'pytest' in sys.modules
   # if os.path.exists('TEST.pickle'):
   #    ssdag, newresult = wu.load('TEST.pickle')
   # else:
   #    ssdag, newresult = worms.app.run_simple(criteria, **kw)
   #    wu.save((ssdag, newresult), 'TEST.pickle')

   ic('run_simple')
   ssdag, newresult = worms.app.run_simple(criteria, **kw)
   # newresult = worms.filters.prune_clashes(ssdag, criteria, newresult, **kw)
   ic('check_geometry')
   newresult = worms.filters.check_geometry(ssdag, criteria, newresult, **kw)
   ic('filter_and_output_results')
   worms.output.filter_and_output_results(
      criteria,
      ssdag,
      newresult,
      # use_simple_pose_construction=False,
      # **kw.sub(output_from_pose=True, merge_bblock=0, output_prefix='testout_orig/testout_orig',
      # ignore_recoverable_errors=False),
      # use_simple_pose_construction=False,
      use_simple_pose_construction=True,
      **kw.sub(
         output_from_pose=True,
         merge_bblock=0,
         output_prefix=f'{testname}/{testname}_',
         ignore_recoverable_errors=False,
         # only_outputs=[0],
      ),
   )
   assert 0

   ic(newresult)
   assert 0

   newresult.sort_on_idx()  # for repeatability

   print('load ref from', worms.data.get_latest_testresult_path(testname, candidates_ok=False))
   _, refdat = worms.data.get_latest_testresult(testname, candidates_ok=False)

   refresult = None
   if refdat is None:
      print('NO REF RESULT')
      fail = True
   else:
      refcrit, refssd, refresult = refdat
      assert refcrit == criteria
      assert refssd == ssdag
      fail = not newresult.approx_equal(refresult)

   if fail:
      if refresult:
         print(f'nresults orig {len(refresult.idx)} new {len(newresult.idx)}')
      make_candidate_test_results(
         testname,
         criteria,
         ssdag,
         newresult,
         refresult,
         **kw,
      )
   if fail:
      assert 0, f'TEST FAIL {testname}'
   print(f'TEST PASS {testname}')

def make_candidate_test_results(
   testname,
   criteria,
   ssdag,
   newresult,
   refresult,
   warn=True,
   **kw,
):
   kw = worms.Bunch(**kw)
   kw.max_output = 999
   new_test_dir = worms.data.make_timestamped_test_dir(testname, candidate=True)
   new_resulttable_file = worms.data.get_latest_testresult_path(testname, candidates_ok=True)

   if warn:
      print('!' * 33, 'TEST FAILED', '!' * 34)
      print('INSPECT THIS NEW TEST DIR AND REMOVE OR ACCEPT:')
      print(new_test_dir)
      print('!' * 80)

   with open(new_resulttable_file, 'wb') as out:
      pickle.dump((criteria, ssdag, newresult), out)

   kw.output_prefix = os.path.join(new_test_dir, testname)
   worms.app.output_simple(criteria, ssdag, newresult, extrachains=False, **kw)
   # assert refresult is not None
   if refresult is not None:
      kw.output_suffix = os.path.join('_REF')
      worms.app.output_simple(criteria, ssdag, refresult, extrachains=False, **kw)

def setup_test_databases(**kw):
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
   # print(spdb)
   worms.data.dump(spdb, 'spdb_before')

   dbpair = worms.database.Databases(bbdb, spdb)
   # print(dbpair.splicedb)
   # worms.ssdag.simple_search_dag(criteria, dbpair, precache_only=True, **kw)
   # print(dbpair.splicedb)

   return dbpair

def setup_testfunc(testname):

   t = worms.Timer()

   # if testname == 'auto':
   # testname = [fn[3] for fn in inspect.stack()[1:] if fn[3].startswith('test_')][0]

   # print('SETTING UP TEST', testname)
   argv = ['@' + worms.data.test_file_path(f'{testname}/config/{testname}.flags')]
   criteria_list, kw = worms.cli.build_worms_setup_from_cli_args(argv, construct_databases=True)

   kw.timer = t
   assert len(criteria_list) == 1
   criteria = criteria_list[0]

   # kw.database = setup_test_databases(criteria, **kw)
   # print(kw.database.bblockdb)
   # print(kw.database.splicedb)

   # assert 0
   t.checkpoint('setup_testfunc')

   return criteria, kw
