import worms

def run_simple(
   criteria,
   **kw,
):

   from worms.search.linear import grow_linear

   kw = worms.Bunch(kw)
   if hasattr(criteria, '__iter__'):
      assert len(criteria) == 1
      criteria = criteria[0]
   # kw.merge_segment = None
   kw.merge_bblock = None

   ssdag = worms.ssdag.simple_search_dag(criteria, lbl='all', **kw).ssdag

   worms.PING('call grow_linear')
   result = grow_linear(
      ssdag=ssdag,
      loss_function=criteria.jit_lossfunc(**kw),
      last_bb_same_as=criteria.from_seg if criteria.is_cyclic else -1,
      lbl='alltogether',
      **kw,
   )
   kw.timer.checkpoint('grow_linear')
   result = worms.search.result.ResultTable(result)

   return ssdag, result

def output_simple(
   criteria,
   ssdag,
   result,
   output_prefix='',
   output_suffix='',
   **kw,
):
   kw = worms.Bunch(kw)
   files_output = list()
   for iresult in range(min(kw.max_output, len(result.idx))):
      segpos = result.pos[iresult]
      xalign = criteria.alignment(segpos)
      if xalign is None: continue

      crystinfo = None
      if hasattr(criteria, "crystinfo"):
         crystinfo = criteria.crystinfo(segpos=result.pos[iresult])
      if crystinfo is not None:
         if crystinfo[0] < kw.xtal_min_cell_size: continue
         if crystinfo[0] > kw.xtal_max_cell_size: continue

      fname = f'{output_prefix}_{iresult:04}_{output_suffix}.pdb'
      # print('align_ax1', xalign @ segpos[0, :, 2])
      # print('align_ax2', xalign @ segpos[-1, :, 2])
      # print(fname)
      # print(result.err[iresult], fname)
      worms.output.dumppdb.graph_dump_pdb(
         fname,
         ssdag,
         result.idx[iresult],
         result.pos[iresult],
         join="bb",
         trim=True,
         xalign=xalign,
         crystinfo=crystinfo,
      )
      files_output.append(fname + '.pdb')

   return files_output
