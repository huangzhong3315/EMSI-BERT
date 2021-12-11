import random
import numpy as np
import codecs
from transformers import BertForMaskedLM, BertTokenizer

CLS = '[CLS]'
SEP = "[SEP]"
PAD = "[PAD]"
MASK = "[MASK]"
UNK = "[UNK]"
START1 = "[unused1]"
END1 = "[unused2]"

START2 = "[unused3]"
END2 = "[unused4]"


class Unigram(object):
    def __init__(self, path="../entity-mask", label_path="data/label.txt"):
        self.path = path
        self.label_path = label_path
        self.tokenizer = BertTokenizer.from_pretrained(self.path)
        self.cls, self.sep, self.pad, self.mask, self.unk = self.tokenizer.convert_tokens_to_ids(
            tokens=[CLS, SEP, PAD, MASK, UNK])
        self.vocab = list()

        for key in self.tokenizer.vocab.keys():
            self.vocab.append(key)

        self.label2index, self.index2label, self.labels = self.load_label()

        print(self.label2index, self.index2label)

    def load_label(self):
        label2index = dict()
        index2label = dict()
        labels = list()

        with codecs.open(self.label_path, "r", "utf8") as openfile:
            for line in openfile:
                line = line.strip("\r\n").split(" ")[0]
                ids = len(label2index)
                if line not in label2index:
                    label2index[line] = ids
                    index2label[ids] = line
                    labels.append(line)
        return label2index, index2label, labels

    def covert_ids_to_tokens(self, ids):
        return "".join(self.tokenizer.convert_ids_to_tokens(ids=ids)).replace(PAD, "").replace(CLS, "").replace(SEP, "")

    # return a id of a word
    def index_a_word(self, word):
        return self.tokenizer.convert_tokens_to_ids(tokens=[word])[0]

    def batch_instance(self, input_instance):

        tokens_list = [each["all_tokens"] for each in input_instance]
        labels = [each["type"] for each in input_instance]
        max_len = max([len(each) for each in tokens_list]) + 2  # CLS AND PAD

        ids_list = []
        attention_list = []
        label_ids = []

        for tokens, label in zip(tokens_list, labels):
            tokens = [CLS] + list(tokens) + [SEP]
            ids = self.tokenizer.convert_tokens_to_ids(tokens=tokens)
            attention = [1] * len(ids)
            # print(new_tokens, new_labels)

            while len(ids) != max_len:
                # print("get in here")
                ids.append(self.pad)
                attention.append(0)
            ids_list.append(ids)
            attention_list.append(attention)
            label_ids.append(self.label2index[label])

        # print(label_ids)

        ids_list = np.array(ids_list, dtype="int64")
        attention_list = np.array(attention_list, dtype="float32")
        label_ids = np.array(label_ids, dtype="int64")

        return ids_list, attention_list, label_ids


unigram = Unigram()
