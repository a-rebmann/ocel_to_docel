from nltk.corpus import stopwords

from const import TERMS_FOR_MISSING


def get_dummy(label, lang):
    return ParsedLabel(label, [label], ['X'], ['none'], ['none'], lang)


class ParsedLabel:

    def __init__(self, label, split, tags, bos, actions, lang):
        self._stopwords = stopwords.words(lang)
        self.lang = lang
        self.label = label
        self.split_label = split
        self.tags = tags
        self.bos = [bo.strip() for bo in bos if bo not in TERMS_FOR_MISSING]
        self.bos_plain = " ".join(bo.strip() for bo in self.bos if bo not in self._stopwords).strip()
        self.actions = actions
        self._act_bo_label = None
        self._main_action = None

    @property
    def act_bo_label(self):
        if self._act_bo_label is None:
            self._act_bo_label = self.main_action + " " + self.main_object
        return self._act_bo_label

    @property
    def main_action(self):
        if not self._main_action:
            self._main_action = self.actions[0].strip() if len(self.actions) > 0 else ""
        return self._main_action

    @property
    def main_object(self):
        return self.bos[0].strip() if len(self.bos) > 0 else ""

    def __repr__(self):
        return str((self.label, self.main_action, self.bos))

