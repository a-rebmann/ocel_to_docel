from typing import Dict
from pandas import DataFrame

from bert_parser.bert_tagger import BertTagger
from bert_parser import label_utils
from const import TERMS_FOR_MISSING


class DynamicAttributeExtractor:

    def __init__(self, events: DataFrame, objects: Dict[str, DataFrame], activity_key="Activity"):
        self.events = events
        self.activity_key = activity_key
        self.objects = objects
        self.label_parser = BertTagger()
        self.parsed_labels = {}
        self.clean_objects = [label_utils.sanitize_label(obj) for obj in self.objects.keys()]

    def parse_labels(self):
        for idx, row in self.events.iterrows():
            if row[self.activity_key] not in self.parsed_labels:
                self.parsed_labels[row[self.activity_key]] = self.label_parser.parse_label(row[self.activity_key])

        print(self.parsed_labels)

    def compute_candidate_dynamic_attributes(self):
        candidates = {}
        for att in self.events.columns:
            print(self.events.dtypes[att])
            if self.events.dtypes[att] != object:
                continue
            if att not in self.objects.keys():
                print(att)
                # get rows for which this attribute has a value
                pattern = '|'.join(TERMS_FOR_MISSING)
                cols_with_val = self.events[(~(self.events[att].astype(str).str.contains(pattern)))]
                print(att)

            col_name = label_utils.sanitize_label(att)
            for obj in self.clean_objects:
                if self.label_parser.label_util.lemmatize_word(obj) in col_name and not obj == col_name:
                    print("found match ", col_name, obj)
        for obj_type in self.clean_objects:
            print(obj_type)
            pass
