import sys, os, json, argparse


parser = argparse.ArgumentParser(description="move worms db around")
parser.add_argument("locations", nargs="+", type=str)
parser.add_argument("--destination", default=".", type=str)
args = parser.parse_args()
dest = args.destination


def get_files_from_dbfile(dbfile):
    with open(dbfile) as inp:
        j = json.load(inp)
    return [e["file"] for e in j]


for dbloc in args.locations:
    host, dbfile = dbloc.split(":")
    os.makedirs(dest, exist_ok=1)
    cmd = f"rsync -z {host}:{dbfile} {dest}"
    print(cmd)
    os.system(cmd)
    localdbfile = dest + "/" + os.path.basename(dbfile)
    files = get_files_from_dbfile(localdbfile)
    for f in files:
        os.makedirs(dest + "/" + os.path.dirname(f[1:]), exist_ok=1)
        cmd = f"rsync -z fw:{f} {dest}/{f[1:]}"
        print(cmd)
        os.system(cmd)
