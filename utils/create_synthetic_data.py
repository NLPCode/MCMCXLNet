# -*- coding: utf-8 -*-
# @Time    : 2020/4/15 11:19 AM
# @Author  : He Xingwei

"""
this scripts uses the one-billion-words dataset to create the synthetic dataset.
And then we train a xlnet token classifier on the synthetic dataset.
There are four classes including
copy: 0
replace: 1
insert: 2
delete: 3
"""

import torch
import numpy as np
from transformers import XLNetTokenizer, XLNetLMHeadModel
import sys,os
import argparse
import random
import time
import codecs
from torch.nn.utils.rnn import pad_sequence
sys.path.append('../')


def create_replaced_samples(input_ids_list, length_list, model=None, tokenizer=None, generate_mode=1):
    """

    :param input_ids: list
    :param length: the length does not include the bos_token_id and the eos_token_id
    :return:
    """
    position_list = []

    incorrect_input_ids_list = []
    label_ids_list = []
    for input_ids, length in zip(input_ids_list,length_list):

        # randomly draw a segment from input_ids
        sublen = random.randint(6,length)
        start_id = random.randint(0,length-sublen)
        end_id = start_id+sublen
        incorrect_input_ids = input_ids[start_id:end_id][:]
        incorrect_input_ids[0] = tokenizer.bos_token_id
        incorrect_input_ids[-1] = tokenizer.eos_token_id

        incorrect_input_ids = torch.tensor(incorrect_input_ids)
        label_ids  = torch.zeros_like(incorrect_input_ids)

        if start_id!=0:
            label_ids[1] = 2
        if end_id!=length:
            label_ids[-1] = 2

        incorrect_input_ids_list.append(incorrect_input_ids)
        label_ids_list.append(label_ids)

        sublen = end_id - start_id
        position = random.randint(1, sublen - 2)
        position_list.append(position)

        # construct 3 sentences only with replacement
        for j in range(1):
            incorrect_input_ids = input_ids[:]
            incorrect_input_ids = torch.tensor(incorrect_input_ids)
            label_ids = torch.zeros_like(incorrect_input_ids)

            incorrect_input_ids_list.append(incorrect_input_ids)
            label_ids_list.append(label_ids)

            position = random.randint(1, length - 2)
            position_list.append(position)

    if generate_mode==1:
        for incorrect_input_ids, label_ids, position in zip(incorrect_input_ids_list, label_ids_list, position_list):
            # random generate the replaced token id
            replaced_token_id = random.randint(0, tokenizer.vocab_size-1)
            while replaced_token_id == incorrect_input_ids[position]:
                replaced_token_id = torch.tensor(random.randint(0, tokenizer.vocab_size - 1))

            incorrect_input_ids[position] = replaced_token_id
            label_ids[position] = 1
    else:
        _input_ids_list = []
        for e in incorrect_input_ids_list:
            _input_ids_list.append(e.clone())
        _mask = pad_sequence(_input_ids_list, batch_first=True, padding_value=-100)
        input_ids = pad_sequence(_input_ids_list, batch_first=True, padding_value=0)
        attention_mask = torch.zeros(input_ids.shape,dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)

        batch_size, max_length = input_ids.shape
        perm_mask = torch.zeros(batch_size, max_length, max_length)
        target_mapping = torch.zeros(batch_size, 1, max_length)
        for i, position in enumerate(position_list):
            perm_mask[i,:,position] = 1
            target_mapping[i,0,position]=1


        with torch.no_grad():
            # attention_mask can use the default None since it will not affect the sentence probability.
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            target_mapping = target_mapping.to('cuda')
            perm_mask = perm_mask.to('cuda')
            outputs = model(input_ids=input_ids,attention_mask=attention_mask, perm_mask=perm_mask, target_mapping=target_mapping)
            logits = outputs[0]
            logits=logits[:,0,:]
            topk=20
            top_k_conditional_probs, top_k_token_ids = torch.topk(logits, topk,dim=-1)
            top_k_token_ids = top_k_token_ids.cpu()

        for i, position in enumerate(position_list):
            sample_id = random.randint(0,topk-1)
            replaced_token_id = top_k_token_ids[i, sample_id]
            if replaced_token_id == input_ids[i, position]:
                sample_id = (sample_id + 1) % topk
                replaced_token_id =  top_k_token_ids[i,sample_id]

            incorrect_input_ids_list[i][position] = replaced_token_id
            label_ids_list[i][position] = 1

    for i in range(len(incorrect_input_ids_list)):
        incorrect_input_ids_list[i] = incorrect_input_ids_list[i].tolist()
        label_ids_list[i] = label_ids_list[i].tolist()

    return incorrect_input_ids_list, label_ids_list


def create_inserted_samples(input_ids_list, length_list, model=None, tokenizer=None, generate_mode=1, **kwargs):
    """
    :param input_ids:
    :return:
    """
    incorrect_input_ids_list = []
    label_ids_list = []
    for input_ids, length in zip(input_ids_list,length_list):

        # randomly draw a segment from input_ids and delete 15% word from the segment.
        sublen = random.randint(4,length)
        start_id = random.randint(0,length-sublen)
        end_id = start_id+sublen
        incorrect_input_ids = input_ids[start_id:end_id][:]
        incorrect_input_ids[0] = tokenizer.bos_token_id
        incorrect_input_ids[-1] = tokenizer.eos_token_id


        num_delete_words = max(1, int(0.15*sublen))

        # position = random.randint(1, sublen-2)
        positions = np.random.choice(sublen-2, num_delete_words, replace=False)+1
        positions = sorted(positions.tolist())
        label_ids = [0]*sublen
        for position in positions:
            label_ids[position+1] = 2

        for i, position in enumerate(positions):
            incorrect_input_ids.pop(position-i)
            label_ids.pop(position-i)

        if start_id!=0:
            label_ids[1] = 2
        if end_id!=length:
            label_ids[-1] = 2
        incorrect_input_ids_list.append(incorrect_input_ids)
        label_ids_list.append(label_ids)

        # construct 3 sentences only with insertion
        for j in range(1):
            # randomly draw a segment from input_ids and delete 15% word from the segment.
            incorrect_input_ids = input_ids[:]
            num_delete_words = max(1, int(0.15 * length))

            # position = random.randint(1, sublen-2)
            positions = np.random.choice(length - 2, num_delete_words, replace=False) + 1
            positions = sorted(positions.tolist())
            label_ids = [0] * length
            for position in positions:
                label_ids[position + 1] = 2

            for i, position in enumerate(positions):
                incorrect_input_ids.pop(position - i)
                label_ids.pop(position - i)

            incorrect_input_ids_list.append(incorrect_input_ids)
            label_ids_list.append(label_ids)


    return incorrect_input_ids_list, label_ids_list

def create_deleted_samples(input_ids_list, length_list, model=None, tokenizer=None, generate_mode=1):

    position_list = []

    incorrect_input_ids_list = []
    label_ids_list = []
    for input_ids, length in zip(input_ids_list,length_list):

        # randomly draw a segment from input_ids .
        sublen = random.randint(6,length)
        start_id = random.randint(0,length-sublen)
        end_id = start_id+sublen
        incorrect_input_ids = input_ids[start_id:end_id][:]
        incorrect_input_ids[0] = tokenizer.bos_token_id
        incorrect_input_ids[-1] = tokenizer.eos_token_id

        position = random.randint(1, sublen-1)
        position_list.append(position)
        incorrect_input_ids.insert(position, 0)

        incorrect_input_ids = torch.tensor(incorrect_input_ids)
        label_ids = torch.zeros_like(incorrect_input_ids)

        if start_id!=0:
            label_ids[1] = 2
        if end_id!=length:
            label_ids[-1] = 2


        incorrect_input_ids_list.append(incorrect_input_ids)
        label_ids_list.append(label_ids)

        # construct 3 sentences only with delete
        for j in range(1):
            incorrect_input_ids = input_ids[:]
            position = random.randint(1, length - 1)
            incorrect_input_ids.insert(position, 0)

            incorrect_input_ids = torch.tensor(incorrect_input_ids)
            label_ids = torch.zeros_like(incorrect_input_ids)

            incorrect_input_ids_list.append(incorrect_input_ids)
            label_ids_list.append(label_ids)
            position_list.append(position)

    if generate_mode==1:
        for incorrect_input_ids, label_ids, position in zip(incorrect_input_ids_list, label_ids_list, position_list):
            inserted_token_id = torch.tensor(random.randint(0, tokenizer.vocab_size-1))
            incorrect_input_ids[position] = inserted_token_id
            label_ids[position] = 3

    else:
        _input_ids_list = []
        for e in incorrect_input_ids_list:
            _input_ids_list.append(e.clone())
        _mask = pad_sequence(_input_ids_list, batch_first=True, padding_value=-100)
        input_ids = pad_sequence(_input_ids_list, batch_first=True, padding_value=0)
        attention_mask = torch.zeros(input_ids.shape,dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)

        batch_size, max_length = input_ids.shape
        perm_mask = torch.zeros(batch_size, max_length, max_length)
        target_mapping = torch.zeros(batch_size, 1, max_length)
        for i, position in enumerate(position_list):
            perm_mask[i,:,position] = 1
            target_mapping[i,0,position]=1

        with torch.no_grad():
            # attention_mask can use the default None since it will not affect the sentence probability.
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            target_mapping = target_mapping.to('cuda')
            perm_mask = perm_mask.to('cuda')
            outputs = model(input_ids=input_ids,attention_mask=attention_mask, perm_mask=perm_mask, target_mapping=target_mapping)
            logits = outputs[0]
            logits=logits[:,0,:]
            topk=20
            top_k_conditional_probs, top_k_token_ids = torch.topk(logits, topk,dim=-1)
            top_k_token_ids = top_k_token_ids.cpu()
        for i, position in enumerate(position_list):
            sample_id = random.randint(0,topk-1)
            inserted_token_id = top_k_token_ids[i, sample_id]
            if inserted_token_id == input_ids[i, position]:
                sample_id = (sample_id + 1) % topk
                inserted_token_id =  top_k_token_ids[i,sample_id]

            incorrect_input_ids_list[i][position] = inserted_token_id
            label_ids_list[i][position] = 3

    for i in range(len(incorrect_input_ids_list)):
        incorrect_input_ids_list[i] = incorrect_input_ids_list[i].tolist()
        label_ids_list[i] = label_ids_list[i].tolist()

    return incorrect_input_ids_list, label_ids_list


def create_synthetic_data(output_file, input_tensors,lengths, model, tokenizer, batch_size=1, dataset_size=100,generate_mode=1):

    incorrect_input_ids_list = []
    label_ids_list = []

    funcs = [create_replaced_samples,create_inserted_samples,create_deleted_samples]
    j = 0
    start = time.time()
    sub_inputs = []
    sub_lengths = []
    total_len = 0
    index = 0
    for input_ids, length in zip(input_tensors, lengths):
        index+=1
        if length>args.max_length+2 or length <6:
            continue
        sub_input_ids=input_ids[:]
        sub_inputs.append(sub_input_ids)
        sub_lengths.append(length)
        j+=1
        if j<batch_size:
            continue
        else:
            for f in funcs:
                incorrect_input_ids, label_ids = f(sub_inputs, sub_lengths, model, tokenizer, generate_mode)
                incorrect_input_ids_list += incorrect_input_ids
                label_ids_list+=label_ids
            total_len += j
            if total_len%1000==0:
                print(f'''\r{total_len}/{dataset_size}, {len(label_ids_list)}, use {time.time()-start:.1f} seconds.''',end='')
            if total_len>=dataset_size:
                print()
                break
            sub_inputs = []
            sub_lengths = []
            j = 0
    print(f"The size of the synthetic data set is {len(incorrect_input_ids_list)}.")
    data_dict = {'incorrect_input_ids_list': incorrect_input_ids_list,
                 'label_ids_list': label_ids_list}
    torch.save(data_dict, output_file)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use key words to generate sentence.")
    parser.add_argument('--dataset', type=str, default='one-billion-words')
    parser.add_argument('--max_length', type=int, default=50,
                        help='the maximum length of the input sentence.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--train_dataset_size', type=int, default=10,
                       help='the number of sentences used to create the synthetic training set.')
    parser.add_argument('--test_dataset_size', type=int, default=10,
                       help='the number of sentences used to create the synthetic test set.')
    parser.add_argument('--generate_mode', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--gpu', type=str, default='1')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model_path = f'../checkpoints/xlnet_maskedlm/{args.dataset}'
    args.model_path = model_path
    if args.generate_mode ==0:
        print('construct data with masked lm.')
    elif args.generate_mode ==1:
        print('construct data with random sampling.')
    else:
        print('construct data with masked lm and random sampling.')

    try:
        # load the pre-trained model and tokenizer
        tokenizer = XLNetTokenizer.from_pretrained(args.model_path)
        model = XLNetLMHeadModel.from_pretrained(args.model_path)
        print('Initialize XLNet from checkpoint {}.'.format(args.model_path))
    except:
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')
        print('Initialize XLNet with default parameters.')
    model.eval()

    model.to('cuda')

    for mode in ['test','train']:
        if mode =='train':
            dataset_size = args.train_dataset_size
        else:
            dataset_size = args.test_dataset_size

        input_file = '../data/{}/{}_xlnet_maskedlm.pt'.format(args.dataset, mode)
        print(f'''Loading data from {input_file}''')
        if not os.path.exists(input_file):
            sentences = []
            lengths = []
            input_tensors = []
            raw_input_file = '../data/{}/{}.txt'.format(args.dataset, mode)
            with codecs.open(raw_input_file, 'r', encoding='utf8') as fr:
                for line in fr:
                    line = line.strip()
                    sentences.append(line)
            # convert sentences to ids
            i =0
            for sentence in sentences:
                input_ids = tokenizer.encode(sentence, add_special_tokens=False)
                _length = len(input_ids)
                input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]

                i +=1
                if (i%10000==0):
                    print(f'''\r Constructing data in process {i} ''', end='')
                _length+=2
                lengths.append(_length)
                input_tensors.append(input_ids)
        else:
            data_dict = torch.load(input_file)
            # each element is the length of [tokenizer.bos_token_id] + input_ids
            lengths = data_dict['length']
            # each element is [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
            input_tensors = data_dict['input_tensors']
            
        dataset_size = np.min([len(input_tensors), dataset_size])
        print('Max sentence length is {}'.format(np.max(lengths)))
        max_sentence_length = np.max(lengths)
        print(f'''The size of the dataset is {len(input_tensors)}''')

        if args.generate_mode==2:
            dataset_size_1 = int(dataset_size/6)*5
            generate_mode = 0
            output_file = '../data/{}/{}_xlnet_synthetic_{}.pt'.format(args.dataset, mode, 'lm')
            print('The output file is ',output_file)
            index = create_synthetic_data(output_file, input_tensors,lengths, model, tokenizer, batch_size=args.batch_size,
                                  dataset_size=dataset_size_1, generate_mode=generate_mode)
            print('index',index)
            dataset_size_2 = dataset_size - dataset_size_1
            input_tensors = input_tensors[index:]
            lengths = lengths[index:]
            generate_mode = 1
            output_file = '../data/{}/{}_xlnet_synthetic_{}.pt'.format(args.dataset, mode,'random')
            print('The output file is ',output_file)
            create_synthetic_data(output_file, input_tensors,lengths, model, tokenizer, batch_size=args.batch_size,
                                  dataset_size=dataset_size_2, generate_mode=generate_mode)
        else:
            output_file = '../data/{}/{}_xlnet_synthetic_{}.pt'.format(args.dataset, mode, generate_mode)
            print('The output file is ',output_file)
            create_synthetic_data(output_file, input_tensors,lengths, model, tokenizer, batch_size=args.batch_size,
                                  dataset_size=dataset_size, generate_mode=generate_mode)






