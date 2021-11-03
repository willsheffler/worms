import worms

def generic_integration_test(testname):

   criteria, kw = setup_testfunc(testname)
   kw.verbosity = 0

   ssdag, newresult = worms.app.run_simple(criteria, **kw)
   newresult.sort_on_idx()  # for repeatability

   refpath, refdat = worms.data.get_latest_testresult(testname, candidates_ok=False)

   if refdat is None:
      fail = True
   else:
      refcrit, refssd, refresult = refdat
      fail = not newresult.approx_equal(refresult)

   if fail:
      record_new_testresults(testname, criteria, ssdag, newresult, **kw)
      assert 0, f'TEST FAIL {testname}'
   print(f'TEST PASS {testname}')

def record_new_testresults(testname, criteria, ssdag, newresult, **kw):
   kw = worms.Bunch(**kw)
   new_test_dir = worms.data.make_timestamped_test_dir(testname, candidate=True)
   new_resulttable_file = worms.data.get_latest_testresult_path(testname, candidates_ok=True)

   print('===============================================')
   print('INSPECT THIS NEW TEST DIR AND REMOVE OR ACCEPT:')
   print(new_test_dir)
   print('===============================================')

   with open(new_resulttable_file, 'wb') as out:
      pickle.dump((criteria, ssdag, newresult), out)

   kw.output_prefix = os.path.join(new_test_dir, 'testname')
   worms.app.output_simple(criteria, ssdag, newresult, **kw)

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
