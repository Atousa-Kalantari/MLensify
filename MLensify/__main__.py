import sys
from .core import run_prediction

def main():
    if len(sys.argv) != 4:
        print("Usage: MLensify <list_file.txt> <yes|no> <num_cores>")
        print("Example: MLensify sources.txt yes 16")
        sys.exit(1)
    list_file, stats_flag, cores_str = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        cores = int(cores_str)
        if cores < 1: raise ValueError
    except:
        print("Error: num_cores must be a positive integer")
        sys.exit(1)
    run_prediction(list_file, stats_flag, cores)

if __name__ == "__main__":
    main()
