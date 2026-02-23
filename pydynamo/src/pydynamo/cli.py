#!/usr/bin/env python3
"""pydynamo CLI entry point."""
# Avoid OMP Error #179 (SHM2) in sandboxed/restricted envs (macOS, etc.)
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import sys

from .runtime import write_error


def main():
    parser = argparse.ArgumentParser(
        prog="pydynamo",
        description="Python CLI for Dynamo subtomogram averaging: crop, reconstruction, alignment, classification (MRA).",
    )
    parser.add_argument(
        "command",
        choices=["crop", "reconstruction", "alignment", "classification", "gen_synthetic"],
        help="Command to run",
    )
    parser.add_argument("--i", "--config", dest="config", default=None, help="Path to config YAML (required except for gen_synthetic)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    parser.add_argument("--json-errors", action="store_true", help="Output errors as JSON")
    parser.add_argument("--log-file", help="Write logs to file")
    args, rest = parser.parse_known_args()

    config_path = args.config
    cmd = args.command
    if cmd != "gen_synthetic" and not config_path:
        write_error(f"--i <config.yaml> is required for {cmd}", args=args, config_path=config_path)
        print("Error: --i <config.yaml> is required for " + cmd, file=sys.stderr)
        sys.exit(1)

    # Dispatch to command
    if cmd == "crop":
        from .commands import crop
        return crop.run(config_path, rest, args)
    elif cmd == "reconstruction":
        from .commands import reconstruction
        return reconstruction.run(config_path, rest, args)
    elif cmd == "alignment":
        from .commands import alignment
        return alignment.run(config_path, rest, args)
    elif cmd == "classification":
        from .commands import classification
        return classification.run(config_path, rest, args)
    elif cmd == "gen_synthetic":
        from .scripts.generate_synthetic import run as gen_run
        try:
            gen_run(config_path=config_path, cli_args=args)
            return 0
        except Exception as e:
            write_error(str(e), args=args, config_path=config_path)
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
