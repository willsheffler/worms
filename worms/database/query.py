def query_bblocks(bbdb, query, *, useclass=True, exclude_bases=None):
   '''query for names only
        match name, _type, _class
        if one match, use it
        if _type and _class match, check useclass option
        Het:NNCx/y require exact number or require extra
    '''
   if query.lower() == 'all':
      return [db['file'] for db in bbdb._alldb]
   query, subq = query.split(':') if query.count(':') else (query, None)
   if subq is None:
      c_hits = [db['file'] for db in bbdb._alldb if query in db['class']]
      n_hits = [db['file'] for db in bbdb._alldb if query == db['name']]
      t_hits = [db['file'] for db in bbdb._alldb if query == db['type']]
      if not c_hits and not n_hits:
         return t_hits
      if not c_hits and not t_hits:
         return n_hits
      if not t_hits and not n_hits:
         return c_hits
      if not n_hits:
         return c_hits if useclass else t_hits
      assert False, f'invalid database or query'
   else:
      excon = None
      if subq.lower().endswith('x'):
         excon = True
      if subq.lower().endswith('y'):
         excon = False
      hits = list()
      assert query == 'Het'
      for db in bbdb._alldb:
         if not query in db['class']:
            continue
         nc = [conn for conn in db['connections'] if conn['direction'] == 'C']
         nn = [conn for conn in db['connections'] if conn['direction'] == 'N']
         nc, tc = len(nc), subq.count('C')
         nn, tn = len(nn), subq.count('N')
         if nc >= tc and nn >= tn:
            if nc + nn == tc + tn and excon is not True:
               hits.append(db['file'])
            elif nc + nn > tc + tn and excon is not False:
               hits.append(db['file'])
   if exclude_bases is not None:
      hits0, hits = hits, []
      for h in hits0:
         base = bbdb._dictdb[h]['base']
         if base == '' or base not in exclude_bases:
            hits.append(h)
      print('exclude_bases', len(hits0), len(hits))
   return hits
