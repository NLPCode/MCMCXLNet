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
sys.path.append('../')

from utils.log import Logger

class XLNetDataset(Dataset):
    def __init__(self, dataset, mode, tokenizer, max_sentence_length=50, is_forward = True):
        assert mode in ["train", "test", 'dev']
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
            num_ignored_sentence = data_dict['num_ignored_sentence']
            print("num_ignored_sentence",num_ignored_sentence)
        else:
            print(f'''Please create the synthetic dataset {data_dict_path}.''')

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
        # construct target_mapping
        target_mapping = torch.zeros(max_batch, t_len, p_len + t_len, dtype=torch.float)

        for i in range(t_len):
            target_mapping[:, i, i + p_len] = 1.0
        # print(target_mapping)
        self.target_mapping = target_mapping

    def __getitem__(self, idx):
        if self.is_forward:
            return torch.tensor(self.input_tensors[idx], dtype=torch.long), self.lengths[idx]
        else:
            # construct input for backward language models
            input_ids = list(reversed(self.input_tensors[idx]))
            input_ids[0] = self.tokenizer.bos_token_id
            input_ids[-1] = self.tokenizer.eos_token_id
            # print(input_ids)
            return torch.tensor(input_ids, dtype=torch.long), self.lengths[idx]

    def __len__(self):
        return self.len

    # @staticmethod
    def create_mini_batch(self, samples):
        input_ids = [s[0] for s in samples]
        label_ids = [s[0][1:].clone() for s in samples]
        lengths_tensors = torch.tensor([s[1] for s in samples])

        _mask = pad_sequence(input_ids, batch_first=True, padding_value=-100)
        attention_mask = torch.zeros(_mask.shape,dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)


        # pad input with 0 (the padded value can be arbitrary number.)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        # pad label with -100 (can not be other number.)
        label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)
        batch_size, max_length = input_ids.shape
        perm_mask = self.perm_mask[:batch_size,:max_length,:max_length]

        target_mapping = self.target_mapping[:batch_size,:max_length-self.p_len,:max_length]

        return input_ids, perm_mask, target_mapping, label_ids, lengths_tensors,attention_mask


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
    def compute_perplexity(model, dataloader):
        """
        compute the perplexity on dataloader with model.
        :param model:
        :param dataloader:
        :return:
        """
        log_ppl = 0
        total_length = 0
        model.eval()
        with torch.no_grad():
            start = time.time()
            for data in dataloader:
                data = [t.to(device) for t in data]
                input_ids, perm_mask, target_mapping, label_ids, lengths_tensors, attention_mask = data
                # attention_mask can use the default None since it will not affect the sentence probability.
                outputs = model(input_ids=input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=label_ids,
                                attention_mask=attention_mask)
                loss = outputs[0]
                log_ppl += loss*torch.sum(lengths_tensors)
                total_length += torch.sum(lengths_tensors)

            if torch.cuda.device_count() > 1:
                torch.distributed.all_reduce_multigpu([log_ppl])
                torch.distributed.all_reduce_multigpu([total_length])

            log_ppl /= total_length
            used_time = time.time() - start
        model.train()
        return log_ppl,used_time



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use key words to generate sentence.")
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--is_forward', type=int, default=1)
    parser.add_argument('--max_sentence_length', type=int, default=50,
                        help='the max length of sentences for training language models.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default='one-billion-words')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.n_gpu = torch.cuda.device_count()

    if args.is_forward:
        model_path = '../checkpoints/forward_xlnet/{}'.format(args.dataset)
        log_path = '../logs/forward_xlnet'
    else:
        model_path = '../checkpoints/backward_xlnet/{}'.format(args.dataset)
        log_path = '../logs/backward_xlnet'
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
        tokenizer = XLNetTokenizer.from_pretrained(model_path)
        model = XLNetLMHeadModel.from_pretrained(model_path)
        logger.logger.info('Initialize XLNet from checkpoint {}.'.format(model_path))
    except:
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')
        logger.logger.info('Initialize XLNet with default parameters.')

    model = XLNetLM(model)


    if args.local_rank == -1 or args.n_gpu<=1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print('local_rank:', args.local_rank)
    print("device:", device)
    model = model.to(device)



    if args.train:
        trainset = XLNetDataset(args.dataset, "train", tokenizer=tokenizer,
                                max_sentence_length=args.max_sentence_length, is_forward=args.is_forward)
        logger.logger.info(f'''The size of trainset is {len(trainset)}.''')
        if args.local_rank == -1 or args.n_gpu <= 1:
            train_sampler = torch.utils.data.RandomSampler(trainset)
        else:
            print('Use {} gpus to train the model.'.format(args.n_gpu))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              find_unused_parameters=True)
            # model = torch.nn.parallel.DistributedDataParallel(model)
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            # test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=trainset.create_mini_batch)

    testset = XLNetDataset(args.dataset, "test", tokenizer=tokenizer,max_sentence_length = args.max_sentence_length, is_forward=args.is_forward)
    logger.logger.info(f'''The size of testset is {len(testset)}.''')

    # if args.dataset!='one-billion-words':
    #     devset = XLNetDataset(args.dataset, "dev", tokenizer=tokenizer,max_sentence_length = args.max_sentence_length, is_forward=args.is_forward)
    #     # concat the devset and the testset
    #     testset = ConcatDataset([devset, testset])
    if args.local_rank == -1 or args.n_gpu <= 1:
        test_sampler =  torch.utils.data.SequentialSampler(testset)
    else:
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

    testloader = DataLoader(testset, batch_size=args.batch_size, sampler=test_sampler, collate_fn=testset.create_mini_batch)
    log_ppl, used_time = XLNetLM.compute_perplexity(model, testloader)
    if args.local_rank in [-1, 0]:
        logger.logger.info('log-perplexity of the dataset is {:.3f}, uses {:.2f} seconds.'.format(log_ppl, used_time))


    if args.train==0:
        exit()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=2, verbose=True, min_lr=1e-6)
    scheduler.step(log_ppl)
    best_log_ppl = log_ppl
    evaluate_steps = 10000
    # evaluate_steps = 100
    print_steps = 10
    global_steps = 0

    start = time.time()
    total_loss = 0
    local_step = 0
    # fine-tune xlnet on the training dataset
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for i, data in enumerate(trainloader):
            global_steps +=1
            local_step +=1
            data = [t.to(device) for t in data]
            input_ids, perm_mask, target_mapping, label_ids, lengths_tensors,attention_mask = data
            # print(input_ids, perm_mask, target_mapping, label_ids, lengths_tensors)
            # for e in  (input_ids, perm_mask, target_mapping, label_ids, lengths_tensors):
            #     print(e.shape)

            # attention_mask can use the default None since it will not affect the sentence probability.
            outputs = model(input_ids=input_ids, perm_mask=perm_mask, target_mapping=target_mapping,labels=label_ids,
                            attention_mask=attention_mask)
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
                log_ppl, used_time = XLNetLM.compute_perplexity(model, testloader)
                if args.local_rank in [-1, 0]:
                    print()
                    logger.logger.info(
                        '\tThe log-perplexity of the test dataset is {:.3f}, best log_ppl is {:.3f}, uses {:.2f} seconds.'.
                            format(log_ppl, best_log_ppl, used_time))
                if log_ppl < best_log_ppl:
                    best_log_ppl = log_ppl
                    if args.local_rank in [-1, 0]:
                        model_to_save = model.module if hasattr(model, "module") else model
                        # Simple serialization for models and tokenizers
                        logger.logger.info('Save the model at {}'.format(args.model_path))
                        model_to_save.save_pretrained(args.model_path)
                        tokenizer.save_pretrained(args.model_path)

                scheduler.step(log_ppl)
                start = time.time()
                total_loss = 0
                local_step = 0
