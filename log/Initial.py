__author__ = 'Administrator'
from Tools import load_data
import numpy as np


class Initial(object):
    def __init__(self,
                 all_data_path,
                 vector_length=200,
                 pre_trained_embedding=".//data//wiki_pubmed"):
        self.all_data_path = all_data_path
        self.pre_trained_embedding = pre_trained_embedding

        # word2index and index2word
        self.word2index = dict()
        self.index2word = dict()
        self.word2index['</s>'] = 0
        self.word2index['DRUG1'] = 1
        self.word2index['DRUG2'] = 2
        self.word2index['DRUG0'] = 3
        self.index()

        # init the label
        self.label = dict()
        self.label['int'] = 0
        self.label['advise'] = 1
        self.label['effect'] = 2
        self.label['mechanism'] = 3
        self.label['other'] = 4

    def index(self):
        print("start index the data in ", self.all_data_path)
        sentences = load_data(path=self.all_data_path)
        current_index = 4
        for sentence in sentences:
            words = str(sentence.new_context).strip("\r").strip("\n").rstrip().split("@@")
            for word in words:
                if word not in self.word2index:
                    self.word2index[word] = current_index
                    current_index += 1
            # index the relations
            for relation in sentence.relation_list:
                sdps = str(relation.sdp).split("@@")
                for sdp in sdps:
                    if sdp not in self.word2index:
                        self.word2index[sdp] = current_index
                        current_index += 1

        for word in self.word2index:
            self.index2word[self.word2index[word]] = word

        # assert to make sure
        for current_index in self.index2word:
            assert self.word2index[self.index2word[current_index]] == current_index
