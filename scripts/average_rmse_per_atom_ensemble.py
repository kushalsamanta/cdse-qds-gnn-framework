#!/usr/bin/env python
"""
average_rmse_all_atoms.py

This script reads an input CSV file that contains ensemble RMSE values for each atom site.
The first column of the CSV is assumed to be "Atomsite" (the atom label) and the remaining columns
contain RMSE values from different ensemble predictions (some cells may be empty).

The script computes, for each row, the average RMSE (ignoring any missing/empty values),
multiplies the average by 1000 (to convert to meV), keeps only the "Atomsite" and "Average RMSE" columns,
sorts the rows in descending order by the average RMSE, and saves the results to an output CSV file.
"""

import os
import pandas as pd
import numpy as np

# ==============================================================================
# Configuration: adjust these paths as needed.
# ------------------------------------------------------------------------------
# Set BASE_DIR to the directory where your ensemble RMSE CSV file is located.
BASE_DIR = os.getcwd()  # Change if necessary

# Path to the input CSV file that contains ensemble RMSE values.
# The file should have rows like:
#   Atomsite,rmse_1,rmse_2,rmse_3,...
INPUT_CSV = os.path.join(BASE_DIR, "ensemble_rmse_results.csv")

# Path to the output CSV file that will contain only "Atomsite" and "Average RMSE".
OUTPUT_CSV = os.path.join(BASE_DIR, "average_rmse_results.csv")
# ==============================================================================

def read_ensemble_file(filepath):
    """
    Reads the CSV file with ensemble RMSE data.
    If the file does not have proper headers, it assigns a default header where the first column is
    "Atomsite" and the rest are named "rmse_1", "rmse_2", etc.
    Returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        # If the first column is not "Atomsite", we assume there is no header and assign one.
        if df.columns[0] != "Atomsite":
            df = pd.read_csv(filepath, header=None)
            ncols = df.shape[1]
            df.columns = ["Atomsite"] + [f"rmse_{i}" for i in range(1, ncols)]
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading {filepath}: {e}")

def compute_average_rmse(df, rmse_columns):
    """
    Computes the average RMSE for each row over the specified rmse_columns,
    ignoring empty or non-numeric cells.
    Returns a pandas Series with the average RMSE.
    """
    def row_average(row):
        values = []
        for col in rmse_columns:
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                try:
                    values.append(float(val))
                except ValueError:
                    continue
        return np.mean(values) if values else np.nan

    return df.apply(row_average, axis=1)

def main():
    # Read the ensemble RMSE CSV file.
    print(f"Reading ensemble file: {INPUT_CSV}")
    df = read_ensemble_file(INPUT_CSV)
    
    # Identify the columns containing RMSE values.
    rmse_columns = list(df.columns[1:])  # All columns except the first one.
    print("Detected RMSE columns:", rmse_columns)
    
    # Compute average RMSE for each atom site.
    # Multiply the average by 1000 to convert to meV.
    df["Average RMSE"] = compute_average_rmse(df, rmse_columns) * 1000
    
    # Subset the DataFrame to only include 'Atomsite' and 'Average RMSE'
    result_df = df[["Atomsite", "Average RMSE"]].copy()
    
    # Sort the results in descending order by Average RMSE.
    result_df = result_df.sort_values(by="Average RMSE", ascending=False)
    
    # Save the results to the output CSV file.
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Average RMSE results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

