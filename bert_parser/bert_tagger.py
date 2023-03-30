from bert_parser.label_utils import LabelUtil, split_label
from bert_parser.parsed_label import ParsedLabel
from bert_parser.bert_wrapper import BertWrapper
from bert_parser.bert_for_label_parsing import BertForLabelParsing

from const import MODEL_PATH, EN


class BertTagger:

    def __init__(self):
        self.model = BertWrapper.load_serialized(MODEL_PATH, BertForLabelParsing)
        self.label_util = LabelUtil()
        self.parse_map = {}

    def check_tok_for_object_type(self, split, pred):
        new_pred = []
        i = 1
        for tok, lab in zip(split, pred):
            if len(split) > i and tok == 'BO' and split[i] != 'BO' and self.label_util.check_propn(tok):
                new_pred.append('X')
            else:
                new_pred.append(lab)
            i += 1
        return new_pred

    def parse_label(self, label, lang=EN, print_outcome=False):
        if label in self.parse_map:
            return self.parse_map[label]
        split, tags = self.predict_single_label(label)
        result = ParsedLabel(label, split, tags, self.find_objects(split, tags), self.find_actions(split, tags, lemmatize=True), lang)
        if print_outcome:
            print(label, ", act:", result.actions, ", bos:", result.bos, ', tags:', tags)
        self.parse_map[label] = result
        return result

    def parse_labels(self, splits: list) -> list:
        processed = self.model.predict(splits)
        tagged_list = []
        for split, tagged in zip(splits, processed[0]):
            tagged_clean = self.check_tok_for_object_type(split, tagged)
            tagged_list.append(tagged_clean)
        return tagged_list

    def predict_single_label(self, label):
        split = split_label(label)
        return split, self.model.predict([split])[0][0]

    def predict_single_label_full(self, label):
        split = split_label(label)
        return split, self.model.predict([split])

    def find_objects(self, split, tags):
        bos_temp = " ".join([tok if bo == 'BO' or bo == 'REC' else '#+#' for tok, bo in zip(split, tags)])
        return bos_temp.split('#+#')

    def find_actor(self, split, tags):
        return [tok for tok, bo in zip(split, tags) if bo == 'ACTOR']

    def find_recipient(self, split, tags):
        return [tok for tok, bo in zip(split, tags) if bo == 'REC']

    def find_actions(self, split, tags, lemmatize=False):
        return [self.label_util.lemmatize_word(tok) if lemmatize else
                tok for tok, a in zip(split, tags) if a in ['A', 'ASTATE', 'BOSTATE']]



