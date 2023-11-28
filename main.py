import os

from algorithms import get_trainer
from utils.config import get_config


def main():
    # Setting local simulation
    os.environ["SIMULATE_ON_THIS_MACHINE"] = "1"
    # sets number of threads spawned per worker by MKL/OpenMP solver in simframe. more than 1 thread per worker might cause heavy performance loss if not enough cpu cores are available
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    trainer = get_trainer(config = get_config())
    trainer.train()

if __name__ == "__main__":
    main()
