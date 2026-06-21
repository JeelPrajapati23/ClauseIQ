import argparse
import sys
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Extract rows with missing values in a given column (e.g. RAGAS faithfulness parsing failures)."
    )
    parser.add_argument("--input", default="ragas_eval_results_v3.csv", help="Input CSV path")
    parser.add_argument("--output", default="missing_faithfulness_rows.csv", help="Output CSV path")
    parser.add_argument("--column", default="faithfulness", help="Column to check for missing values")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: '{args.input}' not found in current directory.")
        sys.exit(1)

    if args.column not in df.columns:
        print(f"Error: column '{args.column}' not found. Available columns: {list(df.columns)}")
        sys.exit(1)

    missing_mask = df[args.column].isna()
    missing_rows = df[missing_mask]

    print(f"Found {len(missing_rows)} rows with missing '{args.column}' values (out of {len(df)} total).")

    if missing_rows.empty:
        print("Nothing to export.")
        return

    missing_rows.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    print(f"\nRow indices (0-indexed, matches original CSV order): {missing_rows.index.tolist()}")


if __name__ == "__main__":
    main()
