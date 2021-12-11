import transformers
from transformers import BertForMaskedLM
import torch
import sys
from Datasets import Datasets
import numpy as np
from sklearn.metrics import *
import time
import copy
from torch import nn
from tqdm import trange
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch
import tensorflow as tf
from transformers import BertForMaskedLM, BertTokenizer

from Unigram import *
import argparse
import os
from MicroCalculate import *

parser = argparse.ArgumentParser()
# parser.add_argument("--train-file", type=str, default="../filter-mont-date/")
# parser.add_argument("--train-file", type=str, default="xml/train/")
# parser.add_argument("--test-file", type=str, default="xml/test/")
parser.add_argument("--train-file", type=str, default="xml/train/")
parser.add_argument("--test-file", type=str, default="xml/test/")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--epoch", type=int, default=20)

parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--warmup-rate", type=float, default=0.1)
parser.add_argument("--use-cuda", type=bool, default=False)
parser.add_argument("--gpu", type=int, default=0)
# parser.add_argument("--count", type=int, default=298)
parser.add_argument("--count", type=int, default=18872)
parser.add_argument("--classes", type=int, default=len(unigram.label2index))

args = parser.parse_args()

# bert_model = BertForMaskedLM.from_pretrained("bert-base-chinese")
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


bert_model = BertForMaskedLM.from_pretrained("../entity-mask", from_tf=True)
bert_model.config.output_hidden_states = True
print(bert_model.config)
tokenizer = BertTokenizer.from_pretrained("../entity-mask")


class Classifier(nn.Module):
	def __init__(self, classes):
		super(Classifier, self).__init__()
		self.loss_fun = nn.CrossEntropyLoss()
		self.linear = nn.Linear(
			in_features=768,
			out_features=classes,
			bias=False
		)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, sequence_out, input_label, train):
		final = sequence_out[:, 0, :].squeeze(1)  # get the CLS  batch * 768
		linear_output = self.linear(final)  # batch * 23
		loss = self.loss_fun(
			linear_output,
			input_label
		)
		if train:
			return loss
		else:
			softmax = self.softmax(linear_output)
		return softmax


classifier = Classifier(
	classes=args.classes
)

print(bert_model.config)
print("load model")
# bert_model = torch.load("save/1_1900000_1.0.bin")
print("load model over")
# bert_model.forward()


if args.use_cuda:
	bert_model = bert_model.cuda(args.gpu)
	classifier = classifier.cuda(args.gpu)

t_total = args.count / args.batch * args.epoch
warmup_steps = t_total * args.warmup_rate

print(t_total, warmup_steps)
print(args)

start_time = time.time()

optimizer = AdamW(list(bert_model.parameters()) + list(classifier.parameters()), lr=args.lr)
scheduler = get_linear_schedule_with_warmup(
	optimizer,
	num_warmup_steps=warmup_steps,
	num_training_steps=t_total,
)

all_count = 0
max_acc = -1
max_epoch = -1
for it in range(args.epoch):
	print("---->train in it={0}<-----".format(it))

	avg_loss = 0
	count = 0

	bert_model.train()

	train_data = Datasets(
		path=args.train_file,
		batch=args.batch,
		train=True
	)
	for index in range(len(train_data)):
		ins = unigram.batch_instance(
			input_instance=train_data[index]
		)
		ids, mask, label = ins
		ids = torch.tensor(ids)
		mask = torch.tensor(mask)
		label = torch.tensor(label)

		if args.use_cuda:
			ids = ids.cuda(args.gpu)
			mask = mask.cuda(args.gpu)
			label = label.cuda(args.gpu)

		outputs = bert_model(ids, attention_mask=mask, masked_lm_labels=None)

		# add avg loss here
		count += 1
		loss = classifier(outputs[-1][-2], label, train=True)
		tmp_loss = loss.data.cpu().numpy()
		avg_loss += tmp_loss

		if index % 1 == 0:
			end_time = time.time()
			print("epoch={0}\tindex={1}/{2}\tavgloss={3}\tfilename={4}\ttime={5}".format(
				it,
				index,
				len(train_data),
				avg_loss / count,
				args.train_file,
				end_time - start_time
			))
			sys.stdout.flush()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()

	bert_model.eval()
	print("eval in it {0}".format(it))
	writefile = codecs.open("eval/it={0}_lr={1}_eval.txt".format(it, args.lr), "w", "utf8")
	test_data = Datasets(
		path=args.test_file,
		batch=args.batch,
		train=False
	)
	true_label = list()
	pred_label = list()
	for test_index in trange(len(test_data)):
		test_ins = unigram.batch_instance(
			input_instance=test_data[test_index]
		)
		ids, mask, label = test_ins

		ids_copy = copy.deepcopy(ids)
		label_copy = copy.deepcopy(label)

		for each in label:
			true_label.append(each)

		ids = torch.tensor(ids)
		mask = torch.tensor(mask)
		label = torch.tensor(label)

		if args.use_cuda:
			ids = ids.cuda(args.gpu)
			mask = mask.cuda(args.gpu)
			label = label.cuda(args.gpu)

		outputs = bert_model(ids, attention_mask=mask, masked_lm_labels=None)
		softmax_output = classifier(outputs[-1][-2], label, train=False)

		softmax_output = softmax_output.data.cpu().numpy()
		for each in np.argmax(softmax_output, axis=1):
			pred_label.append(each)

		argsort = np.argsort(softmax_output, axis=1)
		for ins_index in range(softmax_output.shape[0]):
			sort = argsort[ins_index][::-1]
			words = unigram.covert_ids_to_tokens(ids=ids_copy[ins_index])
			label_context = unigram.index2label[label_copy[ins_index]]
			top_3 = []
			for sort_index in sort[0:1]:
				top_3.append("{0}#{1}".format(unigram.index2label[sort_index], softmax_output[ins_index][sort_index]))
			flags = sort[0] == label_copy[ins_index]

			writefile.write(words + "\t" + label_context + "\t" + "\t".join(top_3) + "\t" + str(flags) + "\n")

	calculateMicroValue(
		y_pred=pred_label,
		y_true=true_label,
		labels=[0, 1, 2, 3]
	)
	torch.save(
		bert_model, "save/epoch={0}-lr={1}-bert.bin".format(it, args.lr)
	)
	# true_label = np.array(true_label)
	# pred_label = np.array(pred_label)
	#
	# print(
	# 	classification_report(
	# 		y_true=true_label,
	# 		y_pred=pred_label,
	# 		digits=4,
	# 		target_names=unigram.labels
	# 	)
	# )
	# acc = accuracy_score(
	# 	y_pred=pred_label,
	# 	y_true=true_label
	# )
	# if acc > max_acc:
	# 	max_epoch = it
	# 	max_acc = acc
	# 	torch.save(
	# 		bert_model, "save/epoch={0}-lr={1}-bert.bin".format(it, args.lr)
	# 	)
	# 	torch.save(
	# 		classifier, "save/epoch={0}-lr={1}-classifier.bin".format(it, args.lr)
	# 	)
	# print("it is in it {0}, max acc ={1} in it {2}".format(
	# 	it, max_acc, max_epoch
	# ))
	writefile.close()
