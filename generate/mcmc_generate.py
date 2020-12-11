# -*- coding: utf-8 -*-
# @Time    : 2020/3/27 11:56 AM
# @Author  : He Xingwei
import numpy as np
from nltk.tokenize import word_tokenize
import torch
import re
import time
from torch.nn.utils.rnn import pad_sequence

class MCMCGenerate(object):
    def __init__(self):
        self.invalid_number = 0
        self.keywords_ids = None
    def clear(self):
        self.keywords_ids = None

    # @staticmethod
    # def normalize(probs, log_ppls=-1, candidates=None,tokenizer=None):
    #     """
    #     normalize the probabilities.
    #     :param probs:
    #     :return:
    #     """
    #     sum_probability = np.sum(probs)
    #     assert sum_probability>0, (sum_probability, probs,log_ppls, [tokenizer.convert_ids_to_tokens(e) for e in candidates])
    #     _probs= probs/sum_probability
    #     return _probs
    #
    @staticmethod
    def normalize(probs):
        """
        normalize the probabilities.
        :param probs:
        :return:
        """
        sum_probability = np.sum(probs)
        _probs= probs/sum_probability
        return _probs

    @staticmethod
    def sample(probs,top_k=-1):
        """
        draw a sample from probabilities
        :param probs: a list of probability
        :return:
        """
        if top_k!=-1:
            index = np.argsort(-probs)
            topk_probs = probs[index][:top_k]
            topk_probs = topk_probs/np.sum(topk_probs)
            list_len = len(topk_probs)
            sample = np.random.choice(list_len, 1, p=topk_probs)
            sample_index = sample[0]
            sample_index = index[sample_index]
        else:
            list_len = len(probs)
            sample = np.random.choice(list_len, 1, p=probs)
            sample_index = sample[0]
        return sample_index

    @staticmethod
    def acceptance_rate(p_old_state, p_new_state_given_old_state, p_new_state, p_old_state_given_new_state, prior_old=1,prior_new=1):
        """
        compute the acceptance rate
        :param p_old_state: the probability for the original sentence.
        :param p_new_state_given_old_state: the conditional probability given by the proposal distribution.
        :param p_new_state: the probability for the new sentence
        :param p_old_state_given_new_state: the conditional probability given by the proposal distribution.
        :param prior_old: prior to choose the proposal distribution for the old state
        :param prior_new:
        :return:
        """
        _acceptance_rate = prior_new * p_old_state_given_new_state * p_new_state / \
                           (prior_old * p_new_state_given_old_state * p_old_state)
        _acceptance_rate = min(1,_acceptance_rate)
        return _acceptance_rate

    @staticmethod
    def is_accept(acceptance_rate):
        u = np.random.random(1)[0]
        if u<=acceptance_rate:
            return 1
        else:
            return 0

    def word_end(self, next_subword, tokenizer_type):
        """
        this function means to check the previous subword is the end of the word.
        :param subword:
        :param tokenizer_type:
        :return:
        """
        if tokenizer_type == 0 or tokenizer_type == 'gpt2':
            if ord(next_subword[0]) == 288 or next_subword in [".", "?", "!", ",", "'", "\""]:
                return True
            else:
                return False
        elif tokenizer_type == 1 or tokenizer_type == 'bert':  # bert
            if next_subword.startswith('##'):
                return False
            else:
                return True
        elif tokenizer_type == 2 or tokenizer_type == 'xlnet':
            if ord(next_subword[0]) == 9601 or next_subword in [".", "?", "!", ",", "'", "\""]:
                return True
            else:
                return False
        else:
            raise ValueError('Invalid tokenizer_type: {tokenizer_type}.')

    def word_start(self, pre_subword, cur_subword, tokenizer_type):
        """
        check whether current subword is the start of a new word
        :param pre_subword:
        :param cur_subword:
        :param tokenizer_type:
        :return:
        """
        if tokenizer_type == 0 or tokenizer_type == 'gpt2':
            if ord(cur_subword[0]) == 288 or pre_subword in [".", "?", "!", ",", "'", "\""]:
                return True
            else:
                return False
        elif tokenizer_type == 1 or tokenizer_type == 'bert':  # bert
            if cur_subword.startswith('##'):
                return False
            else:
                return True
        elif tokenizer_type == 2 or tokenizer_type == 'xlnet':
            if ord(cur_subword[0]) == 9601 or pre_subword in [".", "?", "!", ",", "'", "\""] or \
                    (len(pre_subword) == 1 and ord(pre_subword) == 9601):
                return True
            else:
                return False
        else:
            raise ValueError('Invalid tokenizer_type: {tokenizer_type}.')

    def construct_keyword_vector(self, sentence, keywords, tokenizer):
        tokenizer_type = tokenizer.__class__.__name__
        if 'GPT2' in tokenizer_type:
            tokenizer_type = 0
        elif 'Bert' in tokenizer_type:  # 'bert'
            tokenizer_type = 1
        elif 'XLNet' in tokenizer_type:
            tokenizer_type = 2
        else:
            raise ValueError('Wrong tokenizer.')
        keyword_vector = []
        subwords = tokenizer.tokenize(sentence)
        print(subwords)
        subwords_ids = tokenizer.convert_tokens_to_ids(subwords)
        subwords_ids_length = len(subwords_ids)
        keywords = keywords
        if self.keywords_ids is None:
            print('Tokenize keywords. {}'.format(keywords))
            if tokenizer_type == 0:
                keywords_ids = []
                for keyword in keywords:
                    keywords_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(keyword)))
                    keywords_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + keyword)))
            elif tokenizer_type == 1:  # 'bert'
                keywords_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(keyword)) for keyword in keywords]
            elif tokenizer_type == 2:
                keywords_ids = []
                for keyword in keywords:
                    keysubwords = tokenizer.tokenize(keyword)
                    keywords_ids.append(tokenizer.convert_tokens_to_ids(keysubwords))
                    keysubwords = tokenizer.tokenize('.' + keyword)[2:]
                    keywords_ids.append(tokenizer.convert_tokens_to_ids(keysubwords))
            else:
                raise ValueError('Wrong tokenizer.')
            self.keywords_ids = keywords_ids

        keywords_ids =  [e[:] for e in self.keywords_ids]

        keywords_length = [len(keyword_ids) for keyword_ids in keywords_ids]
        keyword_index = 1
        subword_id = 0
        while subword_id < subwords_ids_length:
            is_keyword = False
            index = -1
            # check whether the current subword is the start of a new word
            if subword_id > 0 and not self.word_start(subwords[subword_id - 1], subwords[subword_id], tokenizer_type):
                # print(subwords[subword_id], len(subwords[subword_id]), 'not', tokenizer_type)
                pass
            else:
                # print(subwords[subword_id])

                for length, keyword_ids in zip(keywords_length, keywords_ids):
                    index += 1
                    if subword_id + length > subwords_ids_length:
                        continue
                    is_keyword = True
                    for j in range(subword_id, subword_id + length):
                        if subwords_ids[j] != keyword_ids[j - subword_id]:
                            is_keyword = False
                            break
                    if is_keyword:
                        # if subword_id + length < subwords_ids_length and subwords[subword_id + length].startswith('##'):
                        if subword_id + length < subwords_ids_length and not self.word_end(subwords[subword_id + length],
                                                                                      tokenizer_type):
                            is_keyword = False
                    if is_keyword:
                        # find a keyword
                        # pop current
                        length = keywords_length.pop(index)
                        keywords_ids.pop(index)
                        if tokenizer_type == 0 or tokenizer_type == 2:
                            if index % 2 == 0:
                                keywords_length.pop(index)
                                keywords_ids.pop(index)
                            else:
                                keywords_length.pop(index - 1)
                                keywords_ids.pop(index - 1)
                        for k in range(subword_id, subword_id + length):
                            keyword_vector.append(keyword_index)
                        keyword_index += 1
                        subword_id += length
                        break
            if is_keyword is False:
                keyword_vector.append(0)
                subword_id += 1
        assert len(keywords_ids)==0
        return keyword_vector, subwords, subwords_ids_length


    def check_sentence(self, sentence, keywords):
        """
        This function aims to check whether the new generated sentence contains all specific keywords.
        :param sentence: a new generated sentence
        :param keywords: a list of keywords
        :return: True if sentence contains all keywords, otherwise False
        """
        words = word_tokenize(sentence)
        # words = re.sub(sentence, r'\W', ' ').split()
        # words = sentence.split()
        d = {}
        for word in words:
            d[word] = 1
        for keyword in keywords:
            if d.get(keyword, -1) == -1:
                self.invalid_number += 1
                print(f'''{self.invalid_number}-th invalid sample is: {sentence}, which does not contain keywords {keyword}.''')
                return False
        return True

    def replace(self):
        raise NotImplementedError

    def insert(self):
        raise NotImplementedError

    def delete(self):
        raise NotImplementedError

class LanguageModel(object):

    def __init__(self, device, forward_lm, forward_lm_tokenizer, backward_lm=None, masked_lm=None):
        """

        :param device:
        :param forward_lm: an instance for LSTMLanguageModel, GPT2 LM or XLNet LM.
        :param forward_lm_tokenizer:
        :param backward_lm: an instance for LSTMLanguageModel, GPT2 LM or XLNet LM.
        :param masked_lm:  a XLNet Masked LM instance.

        """
        self.device = device
        self.forward_lm = forward_lm
        self.forward_lm_tokenizer = forward_lm_tokenizer
        self.backward_lm = backward_lm
        self.masked_lm = masked_lm
        self.forward_lm.to(self.device)
        self.forward_lm.eval()
        if self.backward_lm is not None:
            self.backward_lm.to(self.device)
            self.backward_lm.eval()
        if self.masked_lm is not None:
            self.masked_lm.to(self.device)
            self.masked_lm.eval()

        if 'XLNet' in self.forward_lm.__class__.__name__:
            max_sentence_length = 200
            # construct perm_mask
            p_len = 1  # the length of prompt words, when training LM we use <s> as the prompt word.
            self.p_len = p_len
            t_len = max_sentence_length + 1  # the maximum length of the target.
            max_batch = 200
            perm_mask = torch.zeros(max_batch, p_len + t_len, p_len + t_len)
            for i in range(p_len + t_len):
                if i < p_len:
                    for j in range(p_len, p_len + t_len):
                        perm_mask[:, i, j] = 1
                else:
                    for j in range(i, p_len + t_len):
                        perm_mask[:, i, j] = 1
            self.perm_mask = perm_mask.to(device)
            # construct target_mapping
            target_mapping = torch.zeros(max_batch, t_len, p_len + t_len, dtype=torch.float)

            for i in range(t_len):
                target_mapping[:, i, i + p_len] = 1.0
            # print(target_mapping)
            self.target_mapping = target_mapping.to(device)

        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def perplexity(self, input_texts):
        inputs_ids = []
        inputs_ids.append(self.forward_lm_tokenizer.encode(input_texts, add_special_tokens=False))
        length = len(inputs_ids[0])+1
        probs, log_ppls = self.sentences_probabilities(inputs_ids)
        return log_ppls[0], length

    def sentences_probabilities(self, inputs_ids=None, inputs_texts = None, subword_position=-1, prev_output_tokens=None,
                                repetition_penalty = 1, ignore_last = 0):
        """
        compute the probabilities with the model for the input sentences.
        :param model:
        :param inputs_ids: 2 dimensional list, each element is a list for a sentence.
        :return: probs:  a list of sentence probability
        :return: log_ppl: a list of log-perplexity value.
        """


        if inputs_texts:
           inputs_ids = []
           for input_text in inputs_texts:
               inputs_ids.append(self.forward_lm_tokenizer.encode(input_text, add_special_tokens=False))

        with torch.no_grad():
            # tokenizer.bos_token_id = tokenizer.eos_token_id = 50256
            ids_tensors = []
            lengths = []
            for input_ids in inputs_ids:
                input_ids = [self.forward_lm_tokenizer.bos_token_id] + input_ids[:] + [self.forward_lm_tokenizer.eos_token_id]
                # print(self.forward_lm_tokenizer.convert_ids_to_tokens(input_ids))
                lengths.append(len(input_ids) - 1)
                ids_tensors.append(torch.tensor(input_ids))

            # pad label with -100 (can not be other number.)
            labels_tensors = pad_sequence(ids_tensors, batch_first=True, padding_value=-100)

            # pad input with 0 (the padded value can be arbitrary number.)
            ids_tensors = pad_sequence(ids_tensors, batch_first=True, padding_value=0)
            ids_tensors = ids_tensors.to(self.device)
            labels_tensors = labels_tensors.to(self.device)
            labels_tensors = labels_tensors[:,1:]

            lengths_tensors = torch.tensor(lengths).to(self.device)
            if 'XLNet' in self.forward_lm.__class__.__name__:
                batch_size, max_length = ids_tensors.shape
                perm_mask = self.perm_mask[:batch_size, :max_length, :max_length]
                target_mapping = self.target_mapping[:batch_size, :max_length - self.p_len, :max_length]
                outputs = self.forward_lm(input_ids=ids_tensors, perm_mask=perm_mask, target_mapping=target_mapping)
                logits = outputs[0]
            else:
                outputs = self.forward_lm(ids_tensors)
                logits = outputs[0]
                logits = logits[:, :-1, :]


            # enforce repetition penalty
            if repetition_penalty!=1 and subword_position!=-1:

                if ignore_last!=0:
                    ignore_last*=-1
                    _logits = logits[:ignore_last,subword_position,:]
                else:
                    _logits = logits[:, subword_position, :]
                # construct mask tensors
                mask_tensor = torch.zeros(1, _logits.shape[-1]).to(self.device)
                for previous_token in prev_output_tokens:
                    if mask_tensor[0,previous_token]==0:
                        mask_tensor[:,previous_token] = repetition_penalty
                    else:
                        mask_tensor[:,previous_token] *= repetition_penalty
                mask_tensor=mask_tensor.repeat(_logits.shape[0],1)
                _mask_tensor = mask_tensor * _logits
                _mask_tensor_positive = _mask_tensor > 0
                _mask_tensor_negative = _mask_tensor < 0
                _logits[_mask_tensor_positive] /= mask_tensor[_mask_tensor_positive]
                _logits[_mask_tensor_negative] *= mask_tensor[_mask_tensor_negative]

            # labels_tensors[:,-1]=-100
            # compute sentence probs and log_ppls
            loss_ = self.loss_func(logits.reshape(-1, logits.shape[-1]), labels_tensors.reshape(-1))
            loss_ = loss_.reshape(labels_tensors.shape)
            # print(loss_)
            # if torch.sum(torch.isnan(loss_)):
            #     for i in range(loss_.shape[0]):
            #         for j in range(loss_.shape[1]):
            #             if torch.isnan(loss_[i,j]).item():
            #                 print(i,j)
            # if torch.sum(torch.isnan(logits)):
            #     for i in range(logits.shape[0]):
            #         for j in range(logits.shape[1]):
            #             for k in range(logits.shape[2]):
            #                 if torch.isnan(logits[i,j,k]).item():
            #                     print(i,j,k)
            loss_ = torch.sum(loss_, dim=-1).double()
            log_ppls = (loss_/lengths_tensors).cpu().numpy()
            probs = torch.exp(-loss_).cpu().numpy()
        return probs, log_ppls


    def construct_forward_and_backward_input_ids(self, input_ids, subword_position, include_subword_position=0):
        """
        This function is used to construct forward_input_ids and backward_input_ids for the forward language model
        and the backward language model, respectively.
        :param input_ids:
        :param subword_position:
        :return:
        """
        if include_subword_position==1:
            forward_input_ids = [self.forward_lm_tokenizer.bos_token_id] + input_ids[:subword_position+1]
            backward_input_ids = [self.forward_lm_tokenizer.bos_token_id] + list(reversed(input_ids[subword_position:]))
        else:
            forward_input_ids = [self.forward_lm_tokenizer.bos_token_id] + input_ids[:subword_position]
            backward_input_ids = [self.forward_lm_tokenizer.bos_token_id] + list(reversed(input_ids[subword_position + 1:]))

        return forward_input_ids, backward_input_ids

    def conditional_distribution(self, language_model, input_ids, subword_position=-1, keyword_vector=None, device=None,
                                 sub_tokens_tensor=None, stop_tokens_tensor=None):
        """
        this function is meant to get the distribution of p(x_n |x<n ).
        :param language_model: an instance for LSTMLanguageModel, GPT2 LM or XLNet LM.
        :param input_ids: one dimensional list. the input_ids will start with the bos_token_id.
        :param device: default None
        :return: the top_k probabilities and tokens
        """
        _input_ids = torch.tensor([input_ids])
        input_ids = _input_ids.to(device)

        with torch.no_grad():
            if 'XLNet' in language_model.__class__.__name__:
                batch_size, max_length = input_ids.shape
                perm_mask = self.perm_mask[:batch_size, :max_length, :max_length]
                target_mapping = torch.zeros(batch_size, 1, max_length)
                target_mapping[:, 0, -1] = 1
                perm_mask = perm_mask.to(device)
                target_mapping = target_mapping.to(device)
                outputs = language_model(input_ids=input_ids, attention_mask=None, perm_mask=perm_mask,
                                         target_mapping=target_mapping)

            else:
                outputs = language_model(input_ids)
            #  shape: [1, 1, vocab_size]
            logits = outputs[0]
            logits = logits[0, -1, :]

            # previous word is a keyword, then the following word can not be subword.
            if subword_position>0 and keyword_vector[subword_position-1]>0 and sub_tokens_tensor is not None:
                if stop_tokens_tensor is not None:
                    stop_tokens_tensor = stop_tokens_tensor + sub_tokens_tensor
                else:
                    stop_tokens_tensor = sub_tokens_tensor
            # set the probability of stop tokens to 0
            if stop_tokens_tensor is not None:
                logits = logits.masked_fill(stop_tokens_tensor >0 , -1e8)
            conditional_probs = torch.softmax(logits, -1)
        return conditional_probs


    def conditional_distribution_unidirectional_lm(self, input_ids, subword_position=-1, keyword_vector=None, isforward = 1, top_k = -1,
                                                   sub_tokens_tensor=None, stop_tokens_tensor=None):
        """
        this function is meant to get the distribution of p(x_n |x<n)
        :param input_ids:
        :param top_k:
        :return:
        """
        if isforward==1:
            language_model = self.forward_lm
        else:
            language_model = self.backward_lm

        conditional_probs = self.conditional_distribution(language_model, input_ids, subword_position, keyword_vector,
                                                          device=self.device,
                                                          sub_tokens_tensor = sub_tokens_tensor,
                                                          stop_tokens_tensor = stop_tokens_tensor)
        if top_k != -1:
            # select the top_k probabilities and tokens
            top_k_conditional_probs, top_k_token_ids = torch.topk(conditional_probs, top_k)
            top_k_conditional_probs = top_k_conditional_probs.cpu().numpy()
            top_k_token_ids = top_k_token_ids.cpu().numpy()
        else:
            top_k_conditional_probs = None
            top_k_token_ids = None
        conditional_probs = conditional_probs.cpu().numpy()
        return top_k_conditional_probs, top_k_token_ids, conditional_probs

    def conditional_distribution_bidirectional_lm(self, forward_input_ids, backward_input_ids, subword_position, keyword_vector,
                                                  top_k = -1, sub_tokens_tensor=None, stop_tokens_tensor=None):
        """
        this function is meant to get the distribution of p(x_n |x<n)* p(x_n |x>n).
        :param forward_input_ids:
        :param backward_input_ids:
        :param top_k:
        :return:
        """
        forward_conditional_probs = self.conditional_distribution(self.forward_lm, forward_input_ids, subword_position, keyword_vector,
                                                                  device=self.device,
                                                                  sub_tokens_tensor = sub_tokens_tensor,
                                                                  stop_tokens_tensor=stop_tokens_tensor)
        backward_conditional_probs = self.conditional_distribution(self.backward_lm, backward_input_ids,subword_position, keyword_vector,
                                                                  device=self.device,
                                                                  sub_tokens_tensor = sub_tokens_tensor,
                                                                  stop_tokens_tensor=stop_tokens_tensor)
        conditional_probs = forward_conditional_probs*backward_conditional_probs
        # conditional_probs = forward_conditional_probs
        if top_k != -1:
            # select the top_k probabilities and tokens
            top_k_conditional_probs, top_k_token_ids = torch.topk(conditional_probs, top_k)
            top_k_conditional_probs = top_k_conditional_probs.cpu().numpy()
            top_k_token_ids = top_k_token_ids.cpu().numpy()
        else:
            top_k_conditional_probs = None
            top_k_token_ids = None
        conditional_probs = conditional_probs.cpu().numpy()
        return top_k_conditional_probs, top_k_token_ids, conditional_probs

    def conditional_distribution_masked_lm(self, input_ids, masked_position, keyword_vector,
                                           top_k=-1, sub_tokens_tensor=None, stop_tokens_tensor=None):
        """
        this function is meant to get the distribution of p(x_n |x<n ,x>n).
        :param input_ids:
        :param top_k:
        :param stop_tokens_tensor:
        :return:
        """
        device = self.device
        # shape: [1, seq_len]
        input_ids = [self.forward_lm_tokenizer.bos_token_id] + input_ids[:] + [self.forward_lm_tokenizer.eos_token_id]
        # print(input_ids, keyword_vector, input_ids[masked_position+1])
        # print(self.forward_lm_tokenizer.convert_ids_to_tokens(input_ids))

        input_ids = torch.tensor([input_ids])

        batch_size, max_length = input_ids.shape
        perm_mask = torch.zeros(batch_size, max_length, max_length)
        target_mapping = torch.zeros(batch_size, 1, max_length)
        perm_mask[:,:,masked_position+1] = 1
        target_mapping[:,0,masked_position+1]=1

        with torch.no_grad():
            # attention_mask can use the default None since it will not affect the sentence probability.
            input_ids = input_ids.to(device)
            target_mapping = target_mapping.to(device)
            perm_mask = perm_mask.to(device)
            # since batch_size =1, so there is no padding for input_ids. Therefore, we do not need attention_mask.
            outputs = self.masked_lm(input_ids=input_ids,attention_mask=None, perm_mask=perm_mask, target_mapping=target_mapping)
            #  shape: [1, 1, vocab_size]
            logits = outputs[0]
            logits = logits[0, -1, :]

            # previous word is a keyword, then the following word can not be subword.
            if masked_position>0 and keyword_vector[masked_position-1]>0 and sub_tokens_tensor is not None:
                if stop_tokens_tensor is not None:
                    stop_tokens_tensor = stop_tokens_tensor + sub_tokens_tensor
                else:
                    stop_tokens_tensor = sub_tokens_tensor

            # set the probability of stop tokens to 0
            if stop_tokens_tensor is not None:
                logits = logits.masked_fill(stop_tokens_tensor>0 , -1e8)
            conditional_probs = torch.softmax(logits, -1)
            if top_k != -1:
                # select the top_k probabilities and tokens
                top_k_conditional_probs, top_k_token_ids = torch.topk(conditional_probs, top_k)
                top_k_conditional_probs = top_k_conditional_probs.cpu().numpy()
                top_k_token_ids = top_k_token_ids.cpu().numpy()
            else:
                top_k_conditional_probs = None
                top_k_token_ids = None
            # print(self.forward_lm_tokenizer.convert_ids_to_tokens(top_k_token_ids))
            # print(top_k_conditional_probs)
            conditional_probs = conditional_probs.cpu().numpy()
            return top_k_conditional_probs, top_k_token_ids, conditional_probs

    def generate_candidates(self, input_ids, subword_position, keyword_vector, top_k=None, stop_tokens_tensor=None,
                            sub_tokens_tensor = None, generate_candidate_method=None):
        """

        :param input_ids:
        :param subword_position:
        :param keyword_vector:
        :param top_k:
        :param stop_tokens_tensor:
        :param sub_tokens_tensor:
        :param generate_candidate_method:
        :return:
        """
        if generate_candidate_method<3:
            # Step 1: Construct the forward_input_ids and backward_input_ids for each action
            if 'XLNet' in self.forward_lm.__class__.__name__:
                include_subword_position=1
            else:
                include_subword_position = 0

            forward_input_ids, backward_input_ids = self.construct_forward_and_backward_input_ids\
                (input_ids, subword_position,include_subword_position=include_subword_position)
            # Step 2: Use language models to generate top_k candidate words.
            if generate_candidate_method==0: # use the forward language model
                top_k_conditional_probs, top_k_token_ids, conditional_probs = self.conditional_distribution_unidirectional_lm\
                        (forward_input_ids, subword_position, keyword_vector,  isforward=1, top_k=top_k,sub_tokens_tensor=sub_tokens_tensor,
                         stop_tokens_tensor=stop_tokens_tensor)
            elif generate_candidate_method==1:  # use the backward language model
                top_k_conditional_probs, top_k_token_ids, conditional_probs = self.conditional_distribution_unidirectional_lm\
                        (backward_input_ids, subword_position, keyword_vector, isforward=0,top_k=top_k, sub_tokens_tensor=sub_tokens_tensor,
                         stop_tokens_tensor=stop_tokens_tensor)
            elif generate_candidate_method==2: #use the bidirectional language models
                top_k_conditional_probs, top_k_token_ids, conditional_probs = self.conditional_distribution_bidirectional_lm\
                        (forward_input_ids, backward_input_ids, subword_position, keyword_vector, top_k=top_k,
                         sub_tokens_tensor=sub_tokens_tensor, stop_tokens_tensor=stop_tokens_tensor)
            else:
                raise ValueError('Please input the correct generate_candidate_method in [0,1,2,3].')

        elif generate_candidate_method == 3: # only XLNet supports this method
            # Step 1: Use the masked language model to generate top_k candidate words.
            top_k_conditional_probs, top_k_token_ids, conditional_probs = self.conditional_distribution_masked_lm\
                (input_ids, subword_position, keyword_vector, top_k=top_k,
                 sub_tokens_tensor=sub_tokens_tensor,stop_tokens_tensor=stop_tokens_tensor)
        else:
            raise ValueError('Please input the correct generate_candidate_method in [0,1,2,3].')

        # Generate candidates
        candidates = []
        for i in range(top_k):
            _candidate = input_ids[:]
            _candidate[subword_position] = top_k_token_ids[i]
            candidates.append(_candidate)

        return candidates

