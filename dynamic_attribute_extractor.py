from typing import Dict
from pandas import DataFrame
from sentence_transformers import SentenceTransformer, util

from bert_parser.bert_tagger import BertTagger
from bert_parser import label_utils
from bert_parser.label_utils import sanitize_label
from const import TERMS_FOR_MISSING

missing_pattern = '|'.join(TERMS_FOR_MISSING)


class DynamicAttributeExtractor:

    def __init__(self, events: DataFrame, objects: Dict[str, DataFrame], event_key="EID", activity_key="Activity", time_key="Timestamp"):
        self.events = events
        self.activity_key = activity_key
        self.event_key = event_key
        self.time_key = time_key
        self.objects = objects
        self.label_parser = BertTagger()
        self.clean_objects = {obj: label_utils.sanitize_label(obj) for obj in self.objects.keys()}
        self.sent_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cleaned_atts = {att: label_utils.sanitize_label(att) for att in self.events.columns
                             if att not in [self.event_key, self.activity_key, self.time_key, "Unnamed: 0"] + list(self.objects.keys())}
        self.parsed_labels = {}
        self.parse_labels()
        self.known_embeddings = {
            att: self.sent_model.encode(self.cleaned_atts[att]) for att in self.cleaned_atts.keys()
        }
        self.known_embeddings.update(
            {split_att: self.sent_model.encode(split_att) for att in self.cleaned_atts.keys()
                for split_att in self.cleaned_atts[att].split(' ')}
        )
        self.known_embeddings.update({
            obj: self.sent_model.encode(self.clean_objects[obj]) for obj in self.clean_objects.keys()
        })
        self.known_embeddings.update({
            split_obj: self.sent_model.encode(split_obj) for obj in self.clean_objects.keys()
            for split_obj in obj.split(' ')
        })
        self.known_embeddings.update({
            label: self.sent_model.encode(self.parsed_labels[label].main_object) for label in self.parsed_labels.keys()
        })

    def parse_labels(self):
        for idx, row in self.events.iterrows():
            if row[self.activity_key] not in self.parsed_labels:
                self.parsed_labels[row[self.activity_key]] = self.label_parser.parse_label(row[self.activity_key])
        print(self.parsed_labels)

    def compute_candidate_dynamic_attributes(self):
        candidates = {}
        event_attributes = set()
        name_matches = {}
        mapping = {}
        for att in self.cleaned_atts.keys():
            print(self.events.dtypes[att])

            if self.events.dtypes[att] != object:
                self.events[att] = self.events[att].astype(str)
            # cosine_scores = {obj: util.cos_sim(self.known_embeddings[att], self.known_embeddings[obj])[0][0]
            #                  for obj in self.objects.keys()}
            if att not in self.objects.keys() and att not in [self.event_key, self.activity_key]:
                for obj_type in self.objects.keys():
                    with_values_but_no_or_too_many_instances = self.events[
                        (~(self.events[att].isin(TERMS_FOR_MISSING))) &
                        ((self.events[obj_type].isin(TERMS_FOR_MISSING)) |
                         self.events[obj_type].astype(str).str.split(',').str.len().gt(1))]
                    print(att, obj_type, len(with_values_but_no_or_too_many_instances))
                    if len(with_values_but_no_or_too_many_instances) == 0:
                        if att not in candidates:
                            candidates[att] = set()
                        candidates[att].add(obj_type)
                max_values_per_type = {}
                for obj_type in self.objects.keys():
                    if obj_type not in candidates[att]:
                        continue
                    max_value_list_len = 0
                    with_values = self.events[
                        (~(self.events[att].isin(TERMS_FOR_MISSING))) &
                        (~(self.events[obj_type].isin(TERMS_FOR_MISSING)))]
                    for obj, group in with_values.groupby(obj_type):
                        if len(list(group[att].values)) > max_value_list_len:
                            max_value_list_len = len(list(group[att].values))
                    if max_value_list_len > 1:
                        max_values_per_type[obj_type] = max_value_list_len
                        #print("Counting", att, obj_type, obj, len(list(group[att].values)))
                print("Max value list len", att, max_values_per_type)
                # get minimum number of values
                if len(max_values_per_type) > 0:
                    min_values = min(max_values_per_type.values())
                    for obj_type, max_values in max_values_per_type.items():
                        if max_values != min_values:
                            if att in candidates and obj_type in candidates[att]:
                                candidates[att].remove(obj_type)
            col_name = label_utils.sanitize_label(att)
            for obj, clean_obj in self.clean_objects.items():
                if self.label_parser.label_util.lemmatize_word(clean_obj) in col_name and not clean_obj == col_name:
                    print("found match ", col_name, clean_obj, obj)
                    if att not in name_matches:
                        name_matches[att] = set()
                    name_matches[att].add(obj)
        print("Candidates", candidates)
        for att, obj_types in candidates.items():
            cosine_scores = {obj: util.cos_sim(self.known_embeddings[att], self.known_embeddings[obj])[0][0]
                             for obj in self.objects.keys()}
            if len(obj_types) == 1:
                mapping[att] = obj_types.pop()
            elif len(obj_types) > 1:
                if att in name_matches and len(name_matches[att]) == 1:
                    mapping[att] = name_matches[att].pop()
                elif att in name_matches and len(name_matches[att]) > 1:
                    sims = {obj: cosine_scores[obj] for obj in name_matches[att]}
                    print(sims)
                    split_sims = {split_obj: cosine_scores[split_obj.lower()] for obj in name_matches[att]
                                  for split_obj in obj.split(' ') if len(obj.split(' ')) > 1}
                    #print(split_sims)
                    mapping[att] = max(sims, key=sims.get)
                elif att not in name_matches:
                    print(att)
                    sims = {obj: cosine_scores[obj] for obj in obj_types}
                    # sims = {obj: util.cos_sim(self.known_embeddings[att], self.known_embeddings[obj])[0][0]
                    #                  for obj in obj_types}
                    print(sims)
                    split_sims = {split_obj: cosine_scores[split_obj.lower()] for obj in obj_types
                                  for split_obj in obj.split(' ') if len(obj.split(' ')) > 1}
                    #print(split_sims)
                    mapping[att] = max(sims, key=sims.get)
        print(mapping)
        return mapping


"""
Important for static attributes
# with_single_instance_but_different_value = self.events[
#     (~(self.events[att].isin(TERMS_FOR_MISSING))) &
#     (~(self.events[obj_type].isin(TERMS_FOR_MISSING)))
#     ]
# for group, group_df in with_single_instance_but_different_value.groupby(obj_type):
#     if len(group_df[att].unique()) == 1:
#         print("Static", att, obj_type, group, len(group_df[att].unique()))
"""

"""
# get rows for which this attribute has a value
if att not in self.objects.keys():
    with_values = self.events[~(self.events[att].isin(TERMS_FOR_MISSING))]
    print(att, len(with_values))
    for obj_type in self.objects.keys():
        with_instances = self.events[(~(self.events[obj_type].isin(TERMS_FOR_MISSING)))]
        print(obj_type, len(with_instances))
    # get rows for which this attribute has a value
"""
