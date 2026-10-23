"""Command line: `python -m laya_impute impute --checkpoint DIR --in in.csv --out out.csv`."""

import argparse
import sys

import pandas as pd


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laya_impute")
    sub = parser.add_subparsers(dest="cmd", required=True)

    imp = sub.add_parser("impute", help="fill missing target cells in a CSV")
    imp.add_argument("--checkpoint", required=True, help="directory written by the train step")
    imp.add_argument("--input", required=True, help="CSV with the same column names as training")
    imp.add_argument("--output", required=True)
    imp.add_argument("--audit", default=None, help="optional CSV of every decision")
    imp.add_argument("--min-confidence", type=float, default=0.5)
    imp.add_argument("--device", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "impute":
        from .impute import Imputer

        df = pd.read_csv(args.input, na_values=["?", ""], skipinitialspace=True)
        imputer = Imputer(args.checkpoint, min_confidence=args.min_confidence, device=args.device)
        filled, audit = imputer.impute(df)
        filled.to_csv(args.output, index=False)
        if args.audit:
            audit.to_csv(args.audit, index=False)
        counts = audit["status"].value_counts().to_dict() if len(audit) else {}
        print("imputed %d missing target cell(s): %s" % (len(audit), counts), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
