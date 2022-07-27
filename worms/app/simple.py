from decimal import ExtendedContext
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
   max_output=10,
   output_indices=[],
   xtal_min_cell_size=100,
   xtal_max_cell_size=9e9,
   pos=None,
   **kw,
):
   kw = worms.Bunch(kw)
   files_output = list()
   if not output_indices:
      output_indices = range(min(max_output, len(result.idx)))
   print('output_indices', output_indices)
   for iresult in output_indices:
      segpos = result.pos[iresult] if pos is None else pos

      xalign = criteria.alignment(result.pos[iresult])
      if xalign is None: continue

      crystinfo = None
      if hasattr(criteria, "crystinfo"):
         crystinfo = criteria.crystinfo(segpos=result.pos[iresult])
      if crystinfo is not None:
         if crystinfo[0] < kw.xtal_min_cell_size: continue
         if crystinfo[0] > kw.xtal_max_cell_size: continue

      # print('align_ax1', xalign @ segpos[0, :, 2])
      # print('align_ax2', xalign @ segpos[-1, :, 2])
      # print(fname)
      # assert 0

      iseg = kw.repeat_add_to_segment
      # extensions = {iseg: nrepeat for nrepeat in kw.repeat_add_to_output}

      for iextend in kw.repeat_add_to_output:
         sep = '_' if output_suffix else ''
         fname = f'{output_prefix}_{iresult:04}{sep}{output_suffix}_ext{iextend:04}.pdb'
         print('------------ iextend -------------', iextend, fname, result.err[iresult])
         worms.output.dumppdb.graph_dump_pdb(
            fname,
            ssdag,
            result.idx[iresult],
            segpos,
            join="bb",
            trim=True,
            xalign=xalign,
            crystinfo=crystinfo,
            extensions={iseg: iextend},
            **kw,
         )
         files_output.append(fname + '.pdb')

   return files_output
