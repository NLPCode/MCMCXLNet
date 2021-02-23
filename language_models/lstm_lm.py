# -*- coding: utf-8 -*-
# @Time    : 2020/3/14 12:43 PM
# @Author  : He Xingwei
"""
This script is used to train a LSTM-based language model on a dataset.
"""
import torch
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import time
import os
import sys
import numpy as np
import argparse
sys.path.append('../')
from vocabulary.vocab import Vocab
from utils.log import Logger



class LSTMDataset(Dataset):
    def __init__(self, dataset, mode, tokenizer, max_sentence_length=50, is_forward = True):
        assert mode in ["train", "test", 'dev']
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_sentence_length=max_sentence_length
        self.is_forward = is_forward
        self.sentences = []
        self.lengths = []
        self.input_tensors = []
        self.label_tensors = []


        data_dict_path = '../data/{}/{}_lstm.pt'.format(dataset, mode)
        if os.path.exists(data_dict_path):
            print(f'''Loading data from {data_dict_path}''')
            data_dict = torch.load(data_dict_path)
            self.lengths = data_dict['length']
            self.input_tensors = data_dict['input_tensors']
            num_ignored_sentence = data_dict['num_ignored_sentence']
        else:
            filename_list = []
            if dataset == 'one-billion-words':
                filename = '../data/{}/{}.txt'.format(dataset, mode)
                filename_list.append(filename)
            else:
                for i in range(2):
                    filename = '../data/{}/sentiment.{}.{}'.format(dataset, mode, i)
                    filename_list.append(filename)

            for filename in filename_list:
                with open(filename) as fr:
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
                input_ids = tokenizer.encode(sentence)
                input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]

                i +=1
                if (i%10000==0):
                    print(f'''\r Constructing data in process {i} ''', end='' )
                self.lengths.append(len(input_ids)-1)
                self.input_tensors.append(input_ids)

            data_dict = {'length': self.lengths, 'input_tensors': self.input_tensors,
                          "num_ignored_sentence": num_ignored_sentence}
            torch.save(data_dict, data_dict_path)
        self.len = len(self.input_tensors)
        print('Max sentence length is {}'.format(np.max(self.lengths)))
        total_num =  float(self.len + num_ignored_sentence)
        print(f'''The number of sentences over the maximum sentence length {self.max_sentence_length} is 
            {num_ignored_sentence}/{total_num}={num_ignored_sentence/total_num:.3f}''')


    def __getitem__(self, idx):
        if self.is_forward:
            return torch.tensor(self.input_tensors[idx], dtype=torch.long),  self.lengths[idx]
        else:
            # construct input for backward language models
            input_ids = list(reversed(self.input_tensors[idx]))
            input_ids[0] = tokenizer.bos_token_id
            input_ids[-1] = tokenizer.eos_token_id
            return torch.tensor(input_ids, dtype=torch.long), self.lengths[idx]

    def __len__(self):
        return self.len
    @staticmethod
    def create_mini_batch(samples):
        input_ids = [s[0][:-1] for s in samples]
        lengths_tensors = torch.tensor([s[1] for s in samples])
        label_ids = [s[0][1:] for s in samples]

        # pad input with 0 (the padded value can be arbitrary number.)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        # pad label with -100 (can not be other number.)
        label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)
        # 1 for real tokens and 0 for padded tokens
        masks_tensors = torch.zeros(label_ids.shape,dtype=torch.float32)
        masks_tensors = masks_tensors.masked_fill(label_ids != -100, 1)
        return input_ids, masks_tensors,label_ids, lengths_tensors

class LSTMLanguageModel(torch.nn.Module):

    def __init__(self, vocab_size,  hidden_size, dropout=0.2, padding_idx=-100, num_layers=2):
        """

        :param vocab_size:
        :param hidden_size:
        :param padding_idx: the padding_idx should be consistent with padding_value of label_ids
        :param num_layers:
        :param is_training:
        """
        super(LSTMLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.num_layers = num_layers

        self.dropout = torch.nn.Dropout(dropout)
        self.embeddings = torch.nn.Embedding(self.vocab_size, self.hidden_size)
        self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, dropout = dropout,
                                  batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(self.hidden_size,self.vocab_size)
        # ignore padding_idx
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def forward(self, inputs, labels=None):
        """

        :param input: [batch_size, sequence_length]     [bos,a,b,c]
        :param label:  [batch_size, sequence_length]   [a,b,c,eos]
        :return:
        """
        inputs = self.embeddings(inputs)
        inputs  = self.dropout(inputs)
        hidden_states = self.lstm(inputs)[0]
        hidden_states  = self.dropout(hidden_states)
        logits = self.linear(hidden_states)
        if labels is not None:
            logits = logits.reshape(-1, logits.shape[-1])
            labels = labels.reshape(-1)
            loss = self.CrossEntropyLoss(logits, labels)
            return (loss, logits)
        else:
            return (logits,)

    def decode(self, inputs, hx=None):
        """

        :param input: [batch_size, 1]     [bos,a,b,c]
        :param label:  [batch_size, 1]   [a,b,c,eos]
        :return:
        """
        self.eval()
        with torch.no_grad():
            inputs = self.embeddings(inputs)
            output, hx = self.lstm(inputs,hx=hx)
            logits = self.linear(output)
            logits = logits[0,-1,:]
            log_probs = torch.log(torch.softmax(logits,-1))
            return (log_probs, hx)

    def from_pretrained(self, model_path):
        self.load_state_dict(torch.load(model_path))
    def save_pretrained(self, model_path):
        torch.save(self.state_dict(), model_path)


    def perplexity(self, tokenizer, input_ids=None, input_texts = None):
        if input_texts:
            input_ids = tokenizer.encode(input_texts)
        label_ids = input_ids + [tokenizer.eos_token_id]
        input_ids = [tokenizer.bos_token_id] + input_ids
        length = len(label_ids)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ids_tensors = torch.tensor([input_ids]).to(device)
        labels_tensors = torch.tensor([label_ids]).to(device)

        self.eval()
        with torch.no_grad():
            outputs = self.forward(ids_tensors, labels = labels_tensors)
            log_ppl = outputs[0]
        return log_ppl, length

    def perplexities(self, tokenizer, input_ids_list):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        label_ids = [torch.tensor(_input_ids + [tokenizer.eos_token_id]) for _input_ids in input_ids_list]
        input_ids = [torch.tensor([tokenizer.bos_token_id] + _input_ids) for _input_ids in input_ids_list]

        lengths = [len(e) for e in input_ids]
        lengths_tensors = torch.tensor(lengths).to(device)

        # pad input with 0 (the padded value can be arbitrary number.)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        # pad label with -100 (can not be other number.)
        label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)

        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)

        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)[0]

        loss_ = self.loss_func(logits.reshape(-1, logits.shape[-1]), label_ids.reshape(-1))
        loss_ = loss_.reshape(label_ids.shape)
        loss_ = torch.sum(loss_, dim=-1)
        log_ppls = (loss_/lengths_tensors).cpu().numpy()
        return log_ppls







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
            ids_tensors, masks_tensors, labels_tensors, lengths_tensors = data
            # outputs = model(ids_tensors, attention_mask = masks_tensors, labels = labels_tensors)
            outputs = model(ids_tensors, labels = labels_tensors)
            loss = outputs[0]
            log_ppl += loss*torch.sum(lengths_tensors)
            total_length += torch.sum(lengths_tensors)
        log_ppl /= total_length
        used_time = time.time() - start
    model.train()
    return log_ppl,used_time

if __name__ == "__main__":
    """
    Step 1: train the LSTMLanguageModel on one-billion-words.
    Step 2: fine-tune the pre-trained model on yelp or amazon.
    """

    parser = argparse.ArgumentParser(description="Use key words to generate sentence.")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--is_forward', type=int, default=1)
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--hidden_size', type=int, default=256,help='the hidden_size of LSTM.')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_layers', type=int, default=2,help='the number of layers for the LSTM-based language model.')
    parser.add_argument('--max_sentence_length', type=int, default=50,
                        help='the max length of sentences for training language models.')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='load the model from the given checkpoint, if checkpoint>0, else from the best checkpoint')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default='6')
    parser.add_argument('--dataset', type=str, default='one-billion-words', choices=['one-billion-words','yelp', 'amazon'])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.is_forward:
        model_path = '../checkpoints/forward_lstm_lm/{}'.format(args.dataset)
        log_path = '../logs/forward_lstm_lm'
    else:
        model_path = '../checkpoints/backward_lstm_lm/{}'.format(args.dataset)
        log_path = '../logs/backward_lstm_lm'
    args.model_path = model_path

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = '{}/{}.log'.format(log_path, args.dataset)
    print('The log file is ', log_file)
    logger = Logger(log_file)
    logger.logger.info(args)

    if args.dataset != 'one-billion-words':
        training_files = [f'../data/{args.dataset}/sentiment.train.0',
                          f'../data/{args.dataset}/sentiment.train.1']
    else:
        training_files = ['../data/one-billion-words/train.txt']
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    tokenizer = Vocab(training_files=training_files, vocab_size =args.vocab_size)
    model = LSTMLanguageModel(vocab_size=tokenizer.vocab_size, dropout=args.dropout, hidden_size=args.hidden_size,
                              num_layers=args.num_layers)
    try:
        if args.checkpoint==0:
            model_file = os.path.join(model_path,'best.pt')
        else:
            model_file = os.path.join(model_path, '{}.pt'.format(args.checkpoint))
        # load the pre-trained model and tokenizer
        model.from_pretrained(model_file)
        logger.logger.info('Initialize LSTMLanguageModel from checkpoint {}.'.format(args.model_path))
    except:
        logger.logger.info('Initialize LSTMLanguageModel with default paramerers.')
    logger.logger.info('Model architecture:')
    logger.logger.info('-'*100)
    logger.logger.info(model)
    logger.logger.info('-'*100)
    parameters_num = sum(p.numel() for p in model.parameters())
    logger.logger.info(f'''The model has {parameters_num} paramters''')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)

    testset = LSTMDataset(args.dataset, "test", tokenizer, max_sentence_length = args.max_sentence_length,is_forward=args.is_forward)
    if args.dataset!='one-billion-words':
        devset = LSTMDataset(args.dataset, "dev", tokenizer, max_sentence_length=args.max_sentence_length,
                             is_forward=args.is_forward)
        # concat the devset and the testset
        testset = devset
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, collate_fn=LSTMDataset.create_mini_batch)
    if args.train:
        trainset = LSTMDataset(args.dataset, "train", tokenizer, max_sentence_length = args.max_sentence_length,is_forward=args.is_forward)
        logger.logger.info(f'''The size of trainset is {len(trainset)}, the size of testset is {len(testset)}.''')
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=LSTMDataset.create_mini_batch)

    log_ppl, used_time = compute_perplexity(model, testloader)
    logger.logger.info('log-perplexity of the dataset is {:.3f}, uses {:.2f} seconds.'.format(log_ppl, used_time))

    if args.train==0:
        exit()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True,
                                                           min_lr=1e-8)
    scheduler.step(log_ppl)
    best_log_ppl = log_ppl

    evaluate_steps = int(len(trainset)/args.batch_size/5)
    total_steps = 0

    # fine-tune LSTMLanguageModel on the training dataset
    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        total_loss = 0
        for i, data in enumerate(trainloader):
            total_steps += 1
            data = [t.to(device) for t in data]
            ids_tensors, masks_tensors, labels_tensors, lengths_tensors = data
            outputs = model(ids_tensors,labels = labels_tensors)
            loss = outputs[0]

            # zero the parameter gradients
            optimizer.zero_grad()
            # backward
            loss.backward()
            # clip the gradient by global norm
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
            optimizer.step()
            total_loss += loss.item()
            print("\rEpoch {}/{} is in progress {}/{}, average loss is {:.3f}.".
                  format(epoch+1, args.epochs,i+1,len(trainloader),total_loss/(i+1)),end='')
            if total_steps % evaluate_steps == 0 and args.dataset == 'one-billion-words':
                log_ppl, used_time = compute_perplexity(model, testloader)
                if log_ppl < best_log_ppl:
                    best_log_ppl = log_ppl
                    # Simple serialization for models
                    model_file = os.path.join(model_path, 'best.pt')
                    print('Model weights saved in {}'.format(model_file))
                    model.save_pretrained(model_file)
                scheduler.step(log_ppl)
                logger.logger.info('The log-perplexity of the test dataset is {:.3f}, best log_ppl is {:.3f}, uses {:.2f} seconds.'.
                    format(log_ppl, best_log_ppl, used_time))

        print()
        used_time = time.time() - start
        logger.logger.info("Epoch {}/{}: the average loss of the train dataset is {:.3f}, uses {:.2f} seconds.".
                           format(epoch+1, args.epochs, total_loss/len(trainloader), used_time))

