import worms
from worms.data import get_test_json
from worms.database import merge_json_databases

def test_json_database_merge():

   files = [
      'merge/worms_replicate_result__mbb0004_0003.json',
      'merge/worms_replicate_result__mbb0005_0001.json',
      'merge/worms_replicate_result__mbb0011_0000.json',
      'merge/worms_replicate_result__mbb0011_0002.json',
      'merge/worms_replicate_result__mbb0012_0000.json',
      'merge/worms_replicate_result__mbb0013_0002.json',
      'merge/worms_replicate_result__mbb0017_0001.json',
      'merge/worms_replicate_result__mbb0020_0003.json',
      'merge/worms_replicate_result__mbb0021_0001.json',
      'merge/worms_replicate_result__mbb0021_0009.json',
      'merge/worms_replicate_result__mbb0022_0000.json',
      'merge/worms_replicate_result__mbb0022_0001.json',
      'merge/worms_replicate_result__mbb0022_0004.json',
      'merge/worms_replicate_result__mbb0023_0002.json',
      'merge/worms_replicate_result__mbb0026_0001.json',
      'merge/worms_replicate_result__mbb0028_0011.json',
      'merge/worms_replicate_result__mbb0029_0001.json',
      'merge/worms_replicate_result__mbb0031_0005.json',
   ]

   jsondbs = [get_test_json(f) for f in files]
   merged = merge_json_databases(jsondbs,
                                 # dump_archive='cagextal_O_D3_minimal18.txz',
                                 # overwrite=True,
                                 )
   ref = get_test_json('merge/merged.json')
   if False:
      import json
      with open('NEW_JSON_MERGE_RESULT.json', 'w') as out:
         json.dump(merged, out)
      print('=====================')
      print(ref[0])
      print('=====================')
      print(merged[0])
      print('=====================')

   assert ref == merged

if __name__ == '__main__':
   test_json_database_merge()
