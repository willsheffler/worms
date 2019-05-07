import sys, os
from worms.app import worms_main

if __name__ == "__main__":
    print("RUNNING WORMS CWD", os.getcwd())
    print(sys.argv)
    worms_main(sys.argv[1:])
