import json, os, sys

f = sys.argv[1]
fout = f + "_withnames.json"
if len(sys.argv) > 2:
    fout = sys.argv[2]

s = json.load(open(f))
for x in s:
    if not "name" in x:
        newname = os.path.basename(x["file"]).replace(".pdb", "")
        x["name"] = newname
json.dump(s, open(fout, "w"), sort_keys=True, indent=4)
