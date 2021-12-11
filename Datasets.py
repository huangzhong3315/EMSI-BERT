# coding=utf-8
__author__ = 'Administrator'

from Document import Sentence, RelationPair
from Tools import load_data
import numpy as np
import math
import yaml
import random
import json
from tqdm import tqdm
import copy
from Unigram import *


class Datasets(object):
    def __init__(self, path="data//test", batch=4, train=True):
        # print("start to load data from path", filename)
        self.filename = path
        self.train = train
        self.batch = batch
        self.instances = list()
        self.sentences = load_data(path)
        self.get_features()
        self.batch_ins = []
        self.batch_data()

    def get_features(self):
        for sentence in tqdm(self.sentences):
            f = Feature(sen=sentence)
            for instance in f.instances:
                self.instances.append(instance)

    def __len__(self):
        return len(self.batch_ins)

    def __getitem__(self, item):
        return self.batch_ins[item]

    def batch_data(self):
        print("batch the instance here, make sure")
        if self.train:
            random.shuffle(self.instances)

        batch_number = len(self.instances) // self.batch
        for batch_index in range(batch_number):
            start = batch_index * self.batch
            end = (batch_index + 1) * self.batch
            self.batch_ins.append(
                self.instances[start:end]
            )
        # print(self.batch_ins)


class Feature(object):
    def __init__(self, sen=Sentence(), padding=1):
        self.sen = sen
        self.words = self.sen.new_context.split("@@")
        self.instances = list()
        self.max_length = 150
        self.features()
        self.padding = padding

    def tokenizer_string(self, input_str):
        tokens = unigram.tokenizer.tokenize(text=input_str)
        return list(tokens)

    def solve_relation(self, input_realation=RelationPair()):
        all_tokens = []
        for index, word in enumerate(self.words):
            tokens = self.tokenizer_string(input_str=word)
            if index == input_realation.e1_position:
                all_tokens.append(START1)
                all_tokens += tokens
                all_tokens.append(END1)
            elif index == input_realation.e2_position:
                all_tokens.append(START2)
                all_tokens += tokens
                all_tokens.append(END2)
            else:
                all_tokens += tokens
        return all_tokens

    def features(self):
        for relation in self.sen.relation_list:
            instance = dict()
            e1_position = relation.e1_position
            e2_position = relation.e2_position
            padding_words = self.words[e1_position:e2_position + 1]
            padding_words[0] = "DRUG1"
            padding_words[-1] = "DRUG2"
            drug = set()
            for entity in self.sen.entity_list:
                drug.add(entity.text)
            for i in range(len(padding_words)):
                if padding_words[i] in drug:
                    padding_words[i] = 'DRUG0'
            # generate all the words
            sentence_words = copy.deepcopy(self.words)
            sentence_words[relation.e1_position] = "DRUG1"
            sentence_words[relation.e2_position] = "DRUG2"
            for i in range(len(sentence_words)):
                if sentence_words[i] in drug:
                    sentence_words[i] = "DRUG0"
            # ------->整个句子的单词
            instance['e1_pos'] = relation.e1_position
            instance['e2_pos'] = relation.e2_position
            instance["e1_name"] = relation.e1_name
            instance["e2_name"] = relation.e2_name
            instance['word_sequence'] = sentence_words
            instance["negative"] = self.filter(instance, relation=relation)
            instance["type"] = relation.type
            instance["label"] = unigram.label2index[relation.type]
            instance["all_tokens"] = self.solve_relation(input_realation=relation)
            # print(yaml.dump(instance, ))
            # print(instance)
            if instance["negative"] is True:
                continue
            self.instances.append(instance)

    # to decide whether the two entities are illegal
    def filter(self, instance, relation=RelationPair()):
        e1_name = str(relation.e1_name).lower()
        e2_name = str(relation.e2_name).lower()

        return self.filter_1(e1_name, e2_name) \
               or self.filter_2(e1_name, e2_name) \
               or self.filter_3(relation, instance) \
               or self.filter_4(instance=instance)

    # 判断名称是否一样
    def filter_1(self, e1_name, e2_name):
        return e1_name == e2_name

    # 判断一个名称是否是另一个名称的缩写
    def filter_2(self, e1_name, e2_name):
        if len(str(e1_name).split(" ")) > 1:
            if len(str(e2_name).split(" ")) == 1:
                split_words = str(e1_name).split(" ")
                line = "".join([word[0] for word in split_words if str(word).rstrip() != ""])
                return line == e2_name

        if len(str(e2_name).split(" ")) > 1:
            if len(str(e1_name).split(" ")) == 1:
                split_words = str(e2_name).split(" ")
                # print "split words", split_words, "\t", e1_name, e2_name
                line = "".join([word[0] for word in split_words if str(word).rstrip() != ""])
                return line == e1_name

    # 判断 A [and, or, ,, (,] B 的情况
    # 判断 A , or 这种情况
    def filter_3(self, relation=RelationPair(), instance=dict()):
        e1_pos = relation.e1_position
        e2_pos = relation.e2_position
        if math.fabs(e2_pos - e1_pos) == 1:
            return True

        if math.fabs(e2_pos - e1_pos) == 2:
            between = str(self.words[min(e1_pos, e2_pos) + 1]).lower()
            # if between == "and" \
            # or between == "or" \
            #         or between == "," \
            #         or between == "(" \
            #         or between == "-":
            if between == "or" \
                    or between == "," \
                    or between == "(" \
                    or between == "-":
                return True
                # if between == "and" and e1_pos - 1 >= 0:
                #     word = str(instance['word_sequence'][e1_pos - 1]).lower()
                #     if word not in ["of", "between", "with"]:
                #         return True

        if math.fabs(e2_pos - e1_pos) == 3:
            minvalue = min(e1_pos, e2_pos)
            word = str(" ".join(self.words[minvalue + 1: minvalue + 3])).lower()
            if word == ", or" or word == "such as":
                return True

    # filter 掉并列的结构，这个很重要
    # a,b,c, and d
    def filter_4(self, instance=None):
        except_words = [",", 'drug0', 'or', '(', '[', ')', ']', "and"]
        flags = False
        if not instance:
            instance = dict()
        e1_pos = instance['e1_pos']
        e2_pos = instance['e2_pos']
        sequence = instance['word_sequence']
        # print sequence
        for i in range(e1_pos + 1, e2_pos):
            word = str(sequence[i]).lower()
            if word not in except_words:
                return False
            else:
                if word == "and":
                    flags = True
        if flags is True:
            if e2_pos - e1_pos <= 4:
                return False
        return True

# d = Datasets(
#     path="xml/small"
# )
# print(len(d.instances))
#
# print(
#     unigram.batch_instance(
#         input_instance=d[0]
#     )
# )
