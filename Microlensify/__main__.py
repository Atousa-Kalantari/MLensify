# Microlensify/__main__.py
import sys
from .core import run_prediction

def main():
    if len(sys.argv) != 4:
        print("Usage: microlensify <list_file.txt> <yes|no> <num_cores>")
        print("Example: microlensify urls_qlp_2cand.txt no 12")
        sys.exit(1)

    list_file = sys.argv[1]
    compute_stats = sys.argv[2].lower()
    if compute_stats not in ("yes", "no"):
        print("Second argument must be 'yes' or 'no'")
        sys.exit(1)

    try:
        num_cores = int(sys.argv[3])
        if num_cores < 1:
            raise ValueError
    except ValueError:
        print("num_cores must be a positive integer")
        sys.exit(1)

    run_prediction(list_file, compute_stats, num_cores)

if __name__ == "__main__":
    main()
