import argparse
import pandas as pd
import importlib
import sys
from typing import Callable, Any

def run_test(parser_module_name: str, pdf_path: str, csv_path: str):
    """
    Tests a dynamically loaded parser module by comparing its output
    with a ground-truth CSV file.
    """
    try:
        # Step 1: Dynamically import the generated parser module and get the 'parse' function
        module = importlib.import_module(parser_module_name)
        parse_function: Callable[[str], pd.DataFrame] = getattr(module, 'parse')
        print(f"✅ Successfully loaded 'parse' function from '{parser_module_name}'.")

    except (ImportError, AttributeError) as e:
        print(f"❌ ERROR: Could not load 'parse' function from '{parser_module_name}'. Details: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Step 2: Load the expected DataFrame from the ground-truth CSV
        expected_df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded ground-truth CSV from '{csv_path}'.")

        # Step 3: Get the actual DataFrame from the generated parser function
        actual_df = parse_function(pdf_path)
        print(f"✅ Successfully executed parser on '{pdf_path}'.")

        # Step 4: Clean up data for a more reliable comparison
        # (e.g., strip whitespace, standardize types)
        for col in expected_df.columns:
            if expected_df[col].dtype == 'object':
                expected_df[col] = expected_df[col].astype(str).str.strip()
            if col in actual_df.columns and actual_df[col].dtype == 'object':
                 actual_df[col] = actual_df[col].astype(str).str.strip()

        # Step 5: Assert that the parser's output equals the CSV's content
        pd.testing.assert_frame_equal(actual_df, expected_df)

        print("\n🎉 SUCCESS: DataFrame from parser matches the expected CSV.")

    except Exception as e:
        print(f"\n❌ FAILURE: Test failed. The parser's output does not match the CSV. Details:\n{e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for bank statement parsers.")
    parser.add_argument("--parser-module", required=True, help="Module name of the parser to test (e.g., custom_parsers.icici_parser)")
    parser.add_argument("--pdf-path", required=True, help="Path to the sample PDF file.")
    parser.add_argument("--csv-path", required=True, help="Path to the ground-truth CSV file.")

    args = parser.parse_args()
    run_test(args.parser_module, args.pdf_path, args.csv_path)