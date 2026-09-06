import importlib
import os
import sys

EXPERIMENTS = {
    "train-cnn": "experiments.train_cnn",
    "controllers": "experiments.controller_ladder",
    "order": "experiments.profile_order",
    "pso": "experiments.pso_scenarios",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in EXPERIMENTS:
        print(f"usage: python3 run.py {{{','.join(EXPERIMENTS)}}} [args...]")
        sys.exit(1)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    mod = importlib.import_module(EXPERIMENTS[sys.argv[1]])
    mod.main(sys.argv[2:])


if __name__ == "__main__":
    main()
