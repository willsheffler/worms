import sys
import os
import psutil
import pickle
import pyrosetta


def report_memuse(tag):
    print(tag, psutil.Process(os.getpid()).memory_info().rss / 2 ** 20)


def main():
    print(len(sys.argv))
    pyrosetta.init()
    report_memuse("init")
    keep = list()
    for posefile in sys.argv[1:]:
        with open(posefile, "rb") as f:
            pose = pickle.load(f)
            # keep.append(pose)
            report_memuse("loaded pose")
        # report_memuse('pose out of scope')


if __name__ == "__main__":
    main()
