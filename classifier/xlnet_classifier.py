# -*- coding: utf-8 -*-
# @Time    : 2020/4/26 5:04 PM
# @Author  : He Xingwei
"""
this script is used to train a xlnet for token classification model.
"""

import torch
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import XLNetTokenizer, XLNetForTokenClassification, AdamW
import time
import os
import sys
import argparse
import pickle
sys.path.append('../')

from utils.log import Logger

class XLNetDataset(Dataset):
    def __init__(self, dataset, mode, tokenizer=None, max_sentence_length=50):
        assert mode in ["train", "test", 'dev']
        self.mode = mode
        if self.mode=='test' or self.mode=='dev':
            self.is_train = False
        else:
            self.is_train = True
        self.tokenizer = tokenizer
        self.max_sentence_length = max_sentence_length
        self.input_tensors = []
        self.label_tensors = []
        # specify the training or test test
        data_dict_path1 = '../data/{}/{}_xlnet_synthetic_lm.pt'.format(dataset, mode)
        data_dict_path2 = '../data/{}/{}_xlnet_synthetic_random.pt'.format(dataset, mode)
        data_dict_paths = [data_dict_path1, data_dict_path2]
        # data_dict_paths = [data_dict_path2]
        for data_dict_path in data_dict_paths:
            if os.path.exists(data_dict_path):
                print(f'''Loading data from {data_dict_path}''')
                # data_dict = pickle.load(open(data_dict_path, 'rb'))
                data_dict = torch.load(data_dict_path)
                self.input_tensors += data_dict['incorrect_input_ids_list']
                self.label_tensors += data_dict['label_ids_list']
            else:
                print(f'''Please create the synthetic dataset {data_dict_path}.''')
        self.len = len(self.input_tensors)
    def __getitem__(self, idx):
            return torch.tensor(self.input_tensors[idx], dtype=torch.long), \
                   torch.tensor(self.label_tensors[idx], dtype=torch.long )

    def __len__(self):
        return self.len

    def create_mini_batch(self, samples):
        input_ids = [s[0] for s in samples]
        label_ids = [s[1] for s in samples]

        _mask = pad_sequence(input_ids, batch_first=True, padding_value=-100)
        attention_mask = torch.zeros(_mask.shape,dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)
        return input_ids, label_ids, attention_mask


class XLNetClassifier(torch.nn.Module):
    def __init__(self,model):
        super(XLNetClassifier, self).__init__()

        # self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1,1,1,1],dtype=torch.float))
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.model = model

    def forward(self, input_ids=None, attention_mask=None, labels=None):

        outputs = self.model(input_ids=input_ids, attention_mask = attention_mask)
        logits = outputs[0]
        if labels is not None:
            loss = self.loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            return (loss, logits)
        else:
            return (logits)

    def save_pretrained(self, model_path):
        self.model.save_pretrained(model_path)

    @staticmethod
    def compute_accuracy(model, dataloader, datasize, device,num_labels):
        """
        compute the perplexity on dataloader with model.
        :param model:
        :param dataloader:
        :return:
        """
        model.eval()
        total_loss = 0
        correct = {}
        recalls = {}
        precisions = {}
        f1s = {}
        for i in range(num_labels):
            recalls[i] = 0.0
            precisions[i] = 0.0
            correct[i] = 0.0
        length = 0
        with torch.no_grad():
            start = time.time()
            for data in dataloader:
                data = [t.to(device) for t in data]
                input_ids, label_ids,attention_mask = data
                # attention_mask can use the default None since it will not affect the sentence probability.
                outputs = model(input_ids=input_ids,attention_mask=attention_mask, labels=label_ids)
                loss, logits = outputs[:2]

                total_loss += loss*input_ids.shape[0]
                length += input_ids.shape[0]
                # compute accuracy
                predict_label = torch.argmax(logits, dim=-1)
                for i in range(num_labels):
                    correct[i] += ((predict_label==i) &  (label_ids== i)).sum()
                    recalls[i] += (label_ids== i).sum()
                    precisions[i] += ((predict_label== i)& (label_ids!=-100)).sum()
            if torch.cuda.device_count()>1:
                torch.distributed.all_reduce_multigpu([total_loss])
            total_loss = total_loss.item()
            for i in range(num_labels):
                if torch.cuda.device_count()>1:
                    torch.distributed.all_reduce_multigpu([correct[i]])
                    torch.distributed.all_reduce_multigpu([recalls[i]])
                    torch.distributed.all_reduce_multigpu([precisions[i]])
                correct[i] = correct[i].item()
                recalls[i] = recalls[i].item()
                precisions[i] = precisions[i].item()


            average_loss = total_loss / datasize
            for i in range(num_labels):
                recalls[i] = correct[i]/recalls[i]
                if precisions[i]!=0:
                    precisions[i] = correct[i]/precisions[i]
                    f1s[i] = 2*recalls[i]*precisions[i]/(recalls[i]+precisions[i])
                else:
                    precisions[i] = 0
                    f1s[i] = 0
            used_time = time.time() - start
        model.train()
        return average_loss,recalls, precisions, f1s ,used_time



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use key words to generate sentence.")
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--num_labels', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--max_sentence_length', type=int, default=50,
                        help='the max length of sentences for training language models.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default='one-billion-words')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # args.n_gpu = len(args.gpu.split(','))
    args.n_gpu = torch.cuda.device_count()

    model_path = f'../checkpoints/xlnet_classifier/{args.dataset}'
    log_path = '../logs/xlnet_classifier'

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
        model = XLNetForTokenClassification.from_pretrained(args.model_path,num_labels=args.num_labels)
        logger.logger.info('Initialize XLNet from checkpoint {}.'.format(args.model_path))
    except:
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = XLNetForTokenClassification.from_pretrained('xlnet-base-cased',num_labels=args.num_labels)
        logger.logger.info('Initialize XLNet with default parameters.')

    model = XLNetClassifier(model)
    """
    copy: 0
    replace: 1
    insert: 2
    delete: 3
    """

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
        trainset = XLNetDataset(args.dataset, "train", tokenizer=tokenizer,max_sentence_length = args.max_sentence_length)
        logger.logger.info(f'''The size of the trainset is {len(trainset)}.''')

        if args.local_rank == -1 or args.n_gpu <= 1:
            train_sampler = torch.utils.data.RandomSampler(trainset)
        else:
            print('Use {} gpus to train the model.'.format(args.n_gpu))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              find_unused_parameters=True)
            # model = torch.nn.parallel.DistributedDataParallel(model)
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = DataLoader(trainset, num_workers=0, batch_size=args.batch_size, sampler=train_sampler,
                                 collate_fn=trainset.create_mini_batch)

    testset = XLNetDataset(args.dataset, "test", tokenizer=tokenizer,max_sentence_length = args.max_sentence_length)


    logger.logger.info(f'''The size of the testset is {len(testset)}.''')
    # assert len(testset)%(args.test_batch_size*args.n_gpu) ==0
    if args.local_rank == -1 or args.n_gpu <= 1:
        test_sampler =  torch.utils.data.SequentialSampler(testset)
    else:
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

    testloader = DataLoader(testset, num_workers=0, batch_size=args.test_batch_size, sampler=test_sampler, collate_fn=testset.create_mini_batch)
    # average_loss, recalls, precisions, f1s , used_time= 0,0,0,0,0
    # Macro_F1 = 100
    average_loss, recalls, precisions, f1s , used_time= XLNetClassifier.compute_accuracy(model, testloader,len(testset),device,args.num_labels)

    Macro_P = np.mean(list(precisions.values()))
    Macro_R = np.mean(list(recalls.values()))
    Macro_F1 = np.mean(list(f1s.values()))
    if args.local_rank in [-1, 0]:
        print()
        logs = f'''    The average loss of the test dataset is {average_loss:.3f}, uses {used_time:.1f} seconds.\n'''
        for i in range(len(f1s)):
            logs += f'''    Label_{i}: Precision={precisions[i]:.3f},  Recall={recalls[i]:.3f}, F1:{f1s[i]:.3f};\n'''
        logs += f'''    Macro_P={Macro_P:.3f},  Macro_R={Macro_R:.3f}, Macro_F1={Macro_F1:.3f}.'''
        logger.logger.info(logs)

    if args.train==0:
        exit()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.1, patience=2, verbose=True, min_lr=1e-6)
    scheduler.step(Macro_F1)
    best_acc = Macro_F1

    # evaluate_steps = 10000
    evaluate_steps = 5000
    print_steps = 10
    global_steps = 0

    start = time.time()
    total_loss = 0
    local_step = 0
    # fine-tune xlnet on the training dataset
    for epoch in range(args.epochs):
        if args.n_gpu > 1:
            # shuffle the data for each epoch
            train_sampler.set_epoch(epoch)
        for i, data in enumerate(trainloader):
            global_steps +=1
            local_step +=1
            data = [t.to(device) for t in data]
            input_ids, label_ids,attention_mask = data
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,labels=label_ids)
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
                average_loss, recalls, precisions, f1s, used_time = XLNetClassifier.compute_accuracy(model, testloader,len(testset),device,num_labels=args.num_labels)
                Macro_P = np.mean(list(precisions.values()))
                Macro_R = np.mean(list(recalls.values()))
                Macro_F1 = np.mean(list(f1s.values()))
                if args.local_rank in [-1, 0]:
                    print()
                    logs = f'''    The average loss of the test dataset is {average_loss:.3f}, uses {used_time:.1f} seconds.\n'''
                    for i in range(len(f1s)):
                        logs += f'''    Label_{i}: Precision={precisions[i]:.3f},  Recall={recalls[i]:.3f}, F1:{f1s[i]:.3f};\n'''
                    logs += f'''    Macro_P={Macro_P:.3f},  Macro_R={Macro_R:.3f}, Macro_F1={Macro_F1:.3f}.'''
                    logger.logger.info(logs)

                if Macro_F1 > best_acc:
                    best_acc = Macro_F1
                    if args.local_rank in [-1, 0]:
                        model_to_save = model.module if hasattr(model, "module") else model
                        # Simple serialization for models and tokenizers
                        print('Save the model at {}'.format(args.model_path))
                        model_to_save.save_pretrained(args.model_path)
                        tokenizer.save_pretrained(args.model_path)
                scheduler.step(Macro_F1)
                start = time.time()
                total_loss = 0
                local_step = 0
