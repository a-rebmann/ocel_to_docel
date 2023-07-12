import os
import sys
import time
import pandas as pd
import ocel

# DIRECTORIES
# default input directory
import dynamic_attribute_extractor
from dynamic_attribute_extractor import DynamicAttributeExtractor

DEFAULT_INPUT_DIR = 'input/'
# default output directory
DEFAULT_OUTPUT_DIR = 'output/'
# default directory for resources
DEFAULT_RES_DIR = 'resources/'


def run(directory, name, transform=False, evaluate=False):
    if ".xlsx" in name and "~$" not in name:
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
        mapping = dae.compute_candidate_dynamic_attributes()
        if evaluate:
            dummy_df = event_df.rename(columns={"Customer_Adress": "attribute one",
                                     "Order_Weight": "attribute two",
                                     "Order_Price": "attribute three",
                                     "Resource": "attribute four",
                                     "Value": "attribute five",
                                     "Refund Value": "attribute six",
                                     "Method": "attribute seven",
                                     "OID": "attribute eight",
                                     }, inplace=False)
            dae = DynamicAttributeExtractor(events=dummy_df, objects=obj_dfs)
            dae.compute_candidate_dynamic_attributes()
        if transform:
            values_tables = dict()
            for key, value in mapping.items():
                print(key, value)
                vals_df = event_df[(~event_df[key].isna()) & (~event_df[key].isin(dynamic_attribute_extractor.TERMS_FOR_MISSING))][["EID", key, value]]
                vals_df = vals_df.assign(VID=range(len(vals_df)))
                vals_df["VID"] = vals_df["VID"].apply(lambda x: "V" + str(x))
                values_tables[key] = vals_df
                event_df.drop(columns=[key], inplace=True)

            #Export the DOCEL log
            with pd.ExcelWriter(
                    DEFAULT_OUTPUT_DIR + name + "_as_DOCEL.xlsx") as writer:
                # event table
                event_df.to_excel(writer, sheet_name='Events')
                # object tables
                for obj_type, obj_df in obj_dfs.items():
                    obj_df.to_excel(writer, sheet_name=obj_type)
                # value tables
                for key, value in values_tables.items():
                    if ":" in key:
                        key = key.replace(":", "_")
                    value.to_excel(writer, sheet_name=key)

    if ".jsonocel" in name:
        ocel = load_ocel(directory, name)
        print("ocel loaded")
        obj_dfs = dict()
        event_records = []
        for key, event in ocel["ocel:events"].items():
            event_record = {
                "EID": key,
                "Activity": event["ocel:activity"],
                "Timestamp": pd.to_datetime(event["ocel:timestamp"]).date(),
            }
            event_record |= {k: v for k, v in event["ocel:vmap"].items()}
            event_record |= {t: [] for t in ocel["ocel:global-log"]["ocel:object-types"]}
            for obj in event["ocel:omap"]:
                if obj in ocel["ocel:objects"]:
                    event_record[ocel["ocel:objects"][obj]["ocel:type"]].append(obj)
            event_records.append(event_record)
        events_df = pd.DataFrame(event_records)
        for obj_type in ocel["ocel:global-log"]["ocel:object-types"]:
            obj_records = []
            for obj, obj_body in ocel["ocel:objects"].items():
                if obj_body["ocel:type"] != obj_type:
                    continue
                obj_record = {
                    "OID": obj
                }
                obj_record |= {k: v for k, v in obj_body["ocel:ovmap"].items()}
                obj_records.append(obj_record)
            obj_dfs[obj_type] = pd.DataFrame(obj_records)
        # Export the OCEL log
        with pd.ExcelWriter(DEFAULT_INPUT_DIR + name.replace(".jsonocel", "").replace(".xes", "") + ".xlsx") as writer:
            # event table
            events_df.to_excel(writer, sheet_name='Events')
            # object tables
            for obj_type, obj_df in obj_dfs.items():
                obj_df.to_excel(writer, sheet_name=obj_type)


def main():
    for (dir_path, dir_names, filenames) in os.walk(DEFAULT_INPUT_DIR):
        for filename in filenames:
            run(dir_path, filename, transform=True, evaluate=False)


def comp_excel(df1, df2):
    comparison_values = df1.values == df2.values
    print(comparison_values)


def load_ocel(filepath, filename):
    return ocel.import_log(filepath + filename)


if __name__ == '__main__':
    main_tic = time.perf_counter()
    main()
    main_toc = time.perf_counter()
    print(f"Program finished all operations in {main_toc - main_tic:0.4f} seconds")
    sys.exit()