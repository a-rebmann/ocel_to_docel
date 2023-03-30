import os
import sys
import time
import pandas as pd

# DIRECTORIES
# default input directory
from dynamic_attribute_extractor import DynamicAttributeExtractor

DEFAULT_INPUT_DIR = 'input/'
# default output directory
DEFAULT_OUTPUT_DIR = 'output/'
# default directory for resources
DEFAULT_RES_DIR = 'resources/'


def run(directory, name):
    if ".xlsx" in name:
        # Load Excel file using Pandas
        f = pd.ExcelFile(directory + "/" + name)
        # Define an empty list to store individual DataFrames
        event_df = None
        obj_dfs = dict()
        # Iterate through each worksheet
        for sheet in f.sheet_names:
            # Parse data from each worksheet as a Pandas DataFrame
            df = f.parse(sheet)
            if sheet == "Events":
                event_df = df
            else:
                obj_dfs[sheet] = df
        if event_df is None:
            raise RuntimeError("None of the sheet captures events, make sure to call that sheet 'Events'!")
        dae = DynamicAttributeExtractor(events=event_df, objects=obj_dfs)
        dae.parse_labels()
        dae.compute_candidate_dynamic_attributes()


def main():
    for (dir_path, dir_names, filenames) in os.walk(DEFAULT_INPUT_DIR):
        for filename in filenames:
            run(dir_path, filename)


if __name__ == '__main__':
    main_tic = time.perf_counter()
    main()
    main_toc = time.perf_counter()
    print(f"Program finished all operations in {main_toc - main_tic:0.4f} seconds")
    sys.exit()
