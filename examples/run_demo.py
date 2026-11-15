"""Run a small controller example."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from softcas.cli import demo
from softcas.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/demo")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/fixture.json")
    demo(config, args.output, stages=2, clients=2)


if __name__ == "__main__":
    main()
