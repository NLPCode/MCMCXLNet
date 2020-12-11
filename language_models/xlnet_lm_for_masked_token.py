# -*- coding: utf-8 -*-
# @Time    : 2020/4/21 1:26 PM
# @Author  : He Xingwei

"""
This script uses the pretrained xlnet lm to predict the mask token.
"""
import torch
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import XLNetTokenizer, XLNetLMHeadModel, AdamW
import time
import os
import sys
import argparse
import random
sys.path.append('../')

from utils.log import Logger

class XLNetDataset(Dataset):
    def __init__(self, dataset, mode, tokenizer, max_sentence_length=50, is_forward = True):
        assert mode in ["test", 'dev']
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_sentence_length = max_sentence_length
        self.is_forward = is_forward
        self.sentences = []
        self.lengths = []
        self.input_tensors = []
        self.mask_ids = []
        data_dict_path = '../data/{}/{}_xlnet_maskedlm.pt'.format(dataset, mode)
        if os.path.exists(data_dict_path):
            print(f'''Loading data from {data_dict_path}''')
            data_dict = torch.load(data_dict_path)
            self.lengths = data_dict['length']
            self.input_tensors = data_dict['input_tensors']
            self.mask_ids = data_dict['mask_ids']
            num_ignored_sentence = data_dict['num_ignored_sentence']
            print("num_ignored_sentence",num_ignored_sentence)

        self.len = len(self.input_tensors)
        print('Max sentence length is {}'.format(np.max(self.lengths)))
        total_num =  float(self.len + num_ignored_sentence)
        print(f'''The number of sentences over the maximum sentence length {self.max_sentence_length} is 
            {num_ignored_sentence}/{total_num}={num_ignored_sentence/total_num:.3f}''')

        # construct perm_mask
        p_len = 1 # the length of prompt words, when training LM we use <s> as the prompt word.
        self.p_len = p_len
        t_len = self.max_sentence_length+1 # the maximum length of the target.
        max_batch = 200
        perm_mask = torch.zeros(max_batch, p_len + t_len, p_len + t_len)
        for i in range(p_len + t_len):
            if i < p_len:
                for j in range(p_len, p_len + t_len):
                    perm_mask[:, i, j] = 1
            else:
                for j in range(i, p_len + t_len):
                    perm_mask[:, i, j] = 1
        self.perm_mask = perm_mask
        # print(perm_mask)


    def __getitem__(self, idx):
        if self.is_forward:
            return self.input_tensors[idx], self.lengths[idx], self.mask_ids[idx]
        else:
            # construct input for backward language models
            input_ids = reversed(self.input_tensors[idx])
            input_ids[0] = self.tokenizer.bos_token_id
            input_ids[-1] = self.tokenizer.eos_token_id
            # print(input_ids)
            return input_ids, self.lengths[idx], self.lengths[idx]-self.mask_ids[idx]-1

    def __len__(self):
        return self.len


    def create_mini_batch(self, samples):
        input_ids = [s[0] for s in samples]
        mask_ids = [s[2] for s in samples]

        for i, _mask_id in enumerate(mask_ids):
            input_ids[i] = input_ids[i][:_mask_id+1]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        batch_size, max_length = input_ids.shape
        label_ids = torch.zeros(batch_size,1, dtype=torch.long)
        target_mapping = torch.zeros(batch_size, 1, max_length)
        perm_mask = self.perm_mask[:batch_size, :max_length, :max_length]
        for i, _mask_id in enumerate(mask_ids):
            label_ids[i,0] = input_ids[i, _mask_id]
            target_mapping[i,0,_mask_id]=1
            input_ids[i, _mask_id] = random.randint(0,20000)
        return input_ids, perm_mask, target_mapping, label_ids


class XLNetLM(torch.nn.Module):
    def __init__(self, model):
        super(XLNetLM, self).__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.model = model

    def forward(self,
        input_ids=None,
        attention_mask=None,
        perm_mask=None,
        target_mapping=None,
        labels=None):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, perm_mask=perm_mask, target_mapping=target_mapping)
        logits = outputs[0]
        if labels is not None:
            loss = self.loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            return (loss, logits)
        else:
            return (logits)

        return loss

    def save_pretrained(self, model_path):
        self.model.save_pretrained(model_path)


    @staticmethod
    def compute_accuracy(forward_dataloader=None,backward_dataloader=None, forward_model=None, backward_model=None,
                         topks=[1,5,10,20,50,100], mode=0):
        """
        compute the perplexity on dataloader with model.
        :param model:
        :param dataloader:
        :return:
        """
        total_loss = 0
        total_length = 0
        correct_dict = {}
        for e in topks:
            correct_dict[e] = 0.0
        forward_model.eval()
        backward_model.eval()
        print('topks',topks,'mode',mode)
        topk=int(np.max(topks))
        with torch.no_grad():
            start = time.time()
            data_length = len(forward_dataloader)
            assert len(forward_dataloader) == len(backward_dataloader)
            step=0
            for data,data2 in zip(forward_dataloader,backward_dataloader):
                step+=1
                if mode ==0: # only use forward
                    data = [t.to(device) for t in data]
                    input_ids, perm_mask, target_mapping, label_ids = data
                    # attention_mask can use the default None since it will not affect the sentence probability.
                    outputs = forward_model(input_ids=input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=label_ids)
                    loss, logits = outputs[:2]
                    total_loss += loss.item()*input_ids.shape[0]
                    total_length += input_ids.shape[0]
                elif mode==1: # only use backward
                    data = data2
                    data = [t.to(device) for t in data]
                    input_ids, perm_mask, target_mapping, label_ids = data
                    # attention_mask can use the default None since it will not affect the sentence probability.
                    outputs = backward_model(input_ids=input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=label_ids)
                    loss, logits = outputs[:2]
                    total_loss += loss.item()*input_ids.shape[0]
                    total_length += input_ids.shape[0]
                else:
                    data = [t.to(device) for t in data]
                    input_ids, perm_mask, target_mapping, label_ids = data
                    # attention_mask can use the default None since it will not affect the sentence probability.
                    outputs = forward_model(input_ids=input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=label_ids)
                    loss, logits = outputs[:2]
                    total_loss += loss.item()*input_ids.shape[0]
                    total_length += input_ids.shape[0]
                    data = data2
                    data = [t.to(device) for t in data]
                    input_ids, perm_mask, target_mapping, label_ids = data

                    # attention_mask can use the default None since it will not affect the sentence probability.
                    outputs = backward_model(input_ids=input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=label_ids)
                    _, logits2 = outputs[:2]
                    logits += logits2

                _, topk_token_ids = torch.topk(logits, topk)
                for i in range(topk):
                    predict_label = topk_token_ids[:,:,i]
                    for j in topks:
                        _sum = (predict_label == label_ids).sum().item()
                        if i<j:
                            correct_dict[j] += _sum

                print(f'''\r{step}/{data_length},{time.time()-start:.1f}''',end='')
                # if step>10:
                #     break

            average_loss = total_loss / total_length
            used_time = time.time() - start
            print(f'''Ave loss is {average_loss:.3f}, use {used_time:.3f} seconds.''')
            for i, j in correct_dict.items():
                print(f"""top{i} accuracy is {j/total_length:.3f}""")
        return average_loss, used_time



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use key words to generate sentence.")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--mode', type=int, default=2)
    parser.add_argument('--max_sentence_length', type=int, default=50,
                        help='the max length of sentences for training language models.')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='one-billion-words', choices=['yelp', 'amazon','one-billion-words'])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    forward_model_path = '../checkpoints/forward_xlnet/{}'.format(args.dataset)

    backward_model_path = '../checkpoints/backward_xlnet/{}'.format(args.dataset)

    forward_model = XLNetLMHeadModel.from_pretrained(forward_model_path)
    backward_model = XLNetLMHeadModel.from_pretrained(backward_model_path)

    forward_tokenizer = XLNetTokenizer.from_pretrained(forward_model_path)
    backward_tokenizer = XLNetTokenizer.from_pretrained(backward_model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    forward_model = forward_model.to(device)
    backward_model = backward_model.to(device)

    forward_testset = XLNetDataset(args.dataset, "test", tokenizer=forward_tokenizer,max_sentence_length = args.max_sentence_length, is_forward=1)
    backward_testset = XLNetDataset(args.dataset, "test", tokenizer=backward_tokenizer,max_sentence_length = args.max_sentence_length, is_forward=0)

    forward_testloader = DataLoader(forward_testset, batch_size=args.batch_size, shuffle=False, collate_fn=forward_testset.create_mini_batch)
    backward_testloader = DataLoader(backward_testset, batch_size=args.batch_size, shuffle=False, collate_fn=backward_testset.create_mini_batch)

    average_loss, used_time = XLNetLM.compute_accuracy(
        forward_dataloader=forward_testloader,backward_dataloader=backward_testloader,
        forward_model=forward_model, backward_model=backward_model, mode=args.mode)
    print(f'''The average loss of the test dataset is {average_loss:.3f}, uses {used_time:.2f} seconds.''')


