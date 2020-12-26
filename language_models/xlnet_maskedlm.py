# -*- coding: utf-8 -*-
# @Time    : 2020/4/21 1:26 PM
# @Author  : He Xingwei

"""
This scripts is used to fine-tune xlnet for language modeling on a specific dataset.
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
import codecs
import random
sys.path.append('../')

from utils.log import Logger

class XLNetDataset(Dataset):
    def __init__(self, dataset, mode, tokenizer, max_sentence_length=50):
        assert mode in ["train", "test", 'dev']
        self.mode = mode
        if self.mode=='test' or self.mode=='dev':
            self.is_train = False
        else:
            self.is_train = True
        self.tokenizer = tokenizer
        self.max_sentence_length = max_sentence_length
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
        else:
            filename_list = []
            filename = '../data/{}/{}.txt'.format(dataset, mode)
            filename_list.append(filename)

            for filename in filename_list:
                with codecs.open(filename, 'r', encoding='utf8') as fr:
                    for line in fr:
                        line = line.strip()
                        self.sentences.append(line)
            num_ignored_sentence = 0
            # convert sentences to ids
            i =0
            for sentence in self.sentences:
                if len(sentence.split())> self.max_sentence_length:
                    num_ignored_sentence+=1
                    continue
                input_ids = tokenizer.encode(sentence, add_special_tokens=False)
                _length = len(input_ids)

                # ignore longer sentence
                if _length > self.max_sentence_length:
                    num_ignored_sentence+=1
                    continue
                input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]

                i +=1
                if (i%10000==0):
                    print(f'''\r Constructing data in process {i} ''', end='')
                _length+=2
                self.lengths.append(_length)
                self.input_tensors.append(input_ids)
                if not self.is_train:
                    # randomly mask one position for a sentence.
                    mask_id = random.randint(1, _length - 2)
                    self.mask_ids.append(mask_id)


            data_dict = {'length':self.lengths, 'mask_ids': self.mask_ids,
                         'input_tensors':self.input_tensors,"num_ignored_sentence":num_ignored_sentence}
            torch.save(data_dict, data_dict_path)

        self.len = len(self.input_tensors)
        print('Max sentence length is {}'.format(np.max(self.lengths)))
        total_num =  float(self.len + num_ignored_sentence)
        print(f'''The number of sentences over the maximum sentence length {self.max_sentence_length} is 
            {num_ignored_sentence}/{total_num}={num_ignored_sentence/total_num:.3f}''')


    def __getitem__(self, idx):
        if self.is_train:
            return torch.tensor(self.input_tensors[idx], dtype=torch.long), self.lengths[idx]
        else:
            return torch.tensor(self.input_tensors[idx], dtype=torch.long), self.lengths[idx], self.mask_ids[idx]

    def __len__(self):
        return self.len

    def create_mini_batch(self, samples):
        input_ids = [s[0] for s in samples]
        lengths = [s[1] for s in samples]
        if self.is_train:
            mask_ids = []
            for i, _length in enumerate(lengths):
                mask_id = random.randint(1, _length - 2)
                mask_ids.append(mask_id)
        else:
            # construct labels
            mask_ids = [s[2] for s in samples]

        lengths_tensors = torch.tensor([s[1] for s in samples])
        _mask = pad_sequence(input_ids, batch_first=True, padding_value=-100)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = torch.zeros(input_ids.shape,dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)

        batch_size, max_length = input_ids.shape
        label_ids = torch.zeros(batch_size,1, dtype=torch.long)
        perm_mask = torch.zeros(batch_size, max_length, max_length)
        target_mapping = torch.zeros(batch_size, 1, max_length)
        for i, _mask_id in enumerate(mask_ids):
            label_ids[i,0] = input_ids[i, _mask_id]
            perm_mask[i,:,_mask_id] = 1
            target_mapping[i,0,_mask_id]=1
            input_ids[i,_mask_id] = 0
        return input_ids, perm_mask, target_mapping, label_ids, attention_mask, lengths_tensors


class XLNetMaskedLM(torch.nn.Module):
    def __init__(self, model):
        super(XLNetMaskedLM, self).__init__()
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

    def save_pretrained(self, model_path):
        self.model.save_pretrained(model_path)


    @staticmethod
    def compute_accuracy(model, dataloader, topks=[1,5,10,20,50,100],train=1):
        """
        compute the perplexity on dataloader with model.
        :param model:
        :param dataloader:
        :return:
        """
        # if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        #     model = torch.nn.DataParallel(model)
        total_loss = 0
        total_length = 0
        correct_dict = {}
        for e in topks:
            correct_dict[e] = 0.0
        model.eval()
        if train==0:
            topk = int(np.max(topks))
            print('topks',topks)
        else:
            topk = 1
            topks = [1]
        data_length = len(dataloader)
        with torch.no_grad():
            start = time.time()
            step = 0
            for data in dataloader:
                step+=1
                data = [t.to(device) for t in data]
                input_ids, perm_mask, target_mapping, label_ids, attention_mask, lengths_tensors = data
                # attention_mask can use the default None since it will not affect the sentence probability.
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, perm_mask=perm_mask,
                                target_mapping=target_mapping, labels=label_ids)
                loss, logits = outputs[:2]

                total_loss += loss.item()*input_ids.shape[0]
                total_length += input_ids.shape[0]

                _, topk_token_ids = torch.topk(logits, topk)
                for i in range(topk):
                    predict_label = topk_token_ids[:,:,i]
                    for j in topks:
                        _sum = (predict_label == label_ids).sum().item()
                        if i<j:
                            correct_dict[j] += _sum
                if train==0:
                    print(f'''\r{step}/{data_length},{time.time()-start:.1f}''',end='')
                if step>=100:
                    break

            average_loss = total_loss / total_length
            used_time = time.time() - start
            if train==0:
                print(f'''Ave loss is {average_loss:.3f}, use {used_time:.3f} seconds.''')
                for i, j in correct_dict.items():
                    print(f"""top{i} accuracy is {j/total_length:.3f}""")
            accuracy = correct_dict[topk]/total_length

        model.train()
        return average_loss, accuracy, used_time





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use key words to generate sentence.")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--max_sentence_length', type=int, default=50,
                        help='the max length of sentences for training language models.')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--convert_data', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default='one-billion-words')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.n_gpu = len(args.gpu.split(','))



    model_path = '../checkpoints/xlnet_maskedlm/{}'.format(args.dataset)
    log_path = '../logs/xlnet_maskedlm'

    args.model_path = model_path
    args.log_path = log_path

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = '{}/{}.log'.format(log_path, args.dataset)
    print('The log file is ', log_file)
    logger = Logger(log_file)
    logger.logger.info(args)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    try:
        # load the pre-trained model and tokenizer
        tokenizer = XLNetTokenizer.from_pretrained(args.model_path)
        model = XLNetLMHeadModel.from_pretrained(args.model_path)
        logger.logger.info('Initialize XLNet from checkpoint {}.'.format(args.model_path))
    except:
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')
        logger.logger.info('Initialize XLNet with default parameters.')

    model = XLNetMaskedLM(model)


    if args.local_rank == -1 or args.n_gpu<=1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print('local_rank:', args.local_rank)
    print("device:", device)
    model = model.to(device)

    if args.train:
        trainset = XLNetDataset(args.dataset, "train", tokenizer=tokenizer, max_sentence_length = args.max_sentence_length)
        logger.logger.info(f'''The size of the trainset is {len(trainset)}.''')

        if args.local_rank == -1 or args.n_gpu <= 1:
            train_sampler = torch.utils.data.RandomSampler(trainset)
        else:
            print('Use {} gpus to train the model.'.format(args.n_gpu))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              find_unused_parameters=True)
            # model = torch.nn.parallel.DistributedDataParallel(model)
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler,
                                 collate_fn=trainset.create_mini_batch)


    testset = XLNetDataset(args.dataset, "test", tokenizer=tokenizer,max_sentence_length = args.max_sentence_length)

    logger.logger.info(f'''The size of the testset is {len(testset)}.''')
    test_sampler =  torch.utils.data.SequentialSampler(testset)
    testloader = DataLoader(testset, batch_size=args.batch_size*2, sampler=test_sampler, collate_fn=testset.create_mini_batch)

    if args.convert_data == 1:
        exit(0)
    average_loss, accuracy, used_time= XLNetMaskedLM.compute_accuracy(model, testloader,train = args.train)
    if args.local_rank in [-1, 0]:
        logger.logger.info('The average loss of the test dataset is {:.3f}, accuracy is {:.3f}, uses {:.2f} seconds.'.
              format(average_loss, accuracy, used_time))
    if args.train==0:
        exit()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.1, patience=2, verbose=True, min_lr=1e-6)
    scheduler.step(accuracy)
    best_acc = accuracy

    evaluate_steps = 10000
    # evaluate_steps = 100
    print_steps = 10
    global_steps = 0

    start = time.time()
    total_loss = 0
    local_step = 0
    # fine-tune xlnet on the training dataset
    for epoch in range(args.epochs):
        for i, data in enumerate(trainloader):
            global_steps +=1
            local_step +=1
            data = [t.to(device) for t in data]
            input_ids, perm_mask, target_mapping, label_ids, attention_mask, lengths_tensors = data
            # print(input_ids, perm_mask, target_mapping, label_ids, lengths_tensors)
            # for e in  (input_ids, perm_mask, target_mapping, label_ids, lengths_tensors):
            #     print(e.shape)

            # attention_mask can use the default None since it will not affect the sentence probability.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            perm_mask=perm_mask, target_mapping=target_mapping,labels=label_ids)
            loss = outputs[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            # backward
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if global_steps%print_steps==0 and args.local_rank in [-1, 0]:
                print("\rEpoch {}/{} is in progress {}/{}, global steps {}, average loss is {:.3f},  {} steps uses {:.1f} seconds.".
                  format(epoch+1, args.epochs,i+1,len(trainloader),global_steps, total_loss/local_step,local_step, time.time()-start), end='')
            if global_steps%evaluate_steps==0 :
                average_loss, accuracy, used_time = XLNetMaskedLM.compute_accuracy(model, testloader,train=args.train)
                if accuracy > best_acc:
                    best_acc = accuracy
                    if args.local_rank in [-1, 0]:
                        model_to_save = model.module if hasattr(model, "module") else model
                        # Simple serialization for models and tokenizers
                        model_to_save.save_pretrained(args.model_path)
                        tokenizer.save_pretrained(args.model_path)
                if args.local_rank in [-1, 0]:
                    print()
                    logger.logger.info \
                        ('\tThe average loss of the test dataset is {:.3f}, best accuracy is {:.3f}, accuracy is {:.3f}, uses {:.2f} seconds.'.
                            format(average_loss, best_acc, accuracy, used_time))
                scheduler.step(accuracy)
                start = time.time()
                total_loss = 0
                local_step = 0
