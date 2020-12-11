#-*- coding: utf-8 -*-
# @Time    : 2020/4/16 4:53 PM
# @Author  : He Xingwei

import sys
sys.path.append('../')
from generate.mcmc_generate import MCMCGenerate, LanguageModel
import torch
import time
from torch.nn.utils.rnn import pad_sequence
class LMGenerate(MCMCGenerate):
    """

    Step 1: Generate candidate sentences.
    Step 2: Draw a new sample from the candidate sentences.
    Step 3: Use the language model to compute  probabilities for candidate sentence.
    Step 4: Compute the acceptance rate.
    """
    def __init__(self, device, forward_lm,  forward_lm_tokenizer, backward_lm=None, masked_lm=None,
                 generate_candidate_method=2, candidate_num = 100, top_k=50, repetition_penalty = 1,
                 classifier_model = None, classifier_model_tokenizer = None):
        super(LMGenerate, self).__init__()
        self.device = device
        self.forward_lm = forward_lm
        self.forward_lm_tokenizer = forward_lm_tokenizer
        self.backward_lm = backward_lm
        self.masked_lm = masked_lm
        self.generate_candidate_method = generate_candidate_method
        self.candidate_num = candidate_num
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        self.forward_lm.to(self.device)
        self.forward_lm.eval()
        if self.backward_lm is not None:
            self.backward_lm.to(self.device)
            self.backward_lm.eval()
        if self.masked_lm is not None:
            self.masked_lm.to(self.device)
            self.masked_lm.eval()
        self.classifier_model = classifier_model
        self.classifier_model_tokenizer = classifier_model_tokenizer

        self.language_model = LanguageModel(device, forward_lm, forward_lm_tokenizer, backward_lm=backward_lm,
                                             masked_lm = masked_lm)

        self.stop_tokens_tensor = torch.zeros(self.forward_lm_tokenizer.vocab_size).to(self.device)
        self.sub_tokens_tensor = torch.zeros(self.forward_lm_tokenizer.vocab_size).to(self.device)

        if 'GPT2' in self.forward_lm.__class__.__name__:
            filename = '../data/tokens/gpt2_stop_tokens.txt'
            index = 0
            with open(filename, 'r') as fr:
                for line in fr:
                    words  = line.strip().split()
                    token_id = int(words[0])
                    self.stop_tokens_tensor[token_id] = 1
                    index+=1
            print('Loading {} stop tokens from {} for {}.'.format(index, filename, self.forward_lm.__class__.__name__))

            # load sub tokens
            filename = '../data/tokens/gpt2_sub_tokens.txt'
            index = 0
            with open(filename, 'r') as fr:
                for line in fr:
                    words  = line.strip().split()
                    token_id = int(words[0])
                    self.sub_tokens_tensor[token_id] = 1
                    index+=1
            print('Loading {} sub tokens from {} for {}.'.format(index, filename, self.forward_lm.__class__.__name__))

        elif 'LSTM' in self.forward_lm.__class__.__name__:
            self.stop_tokens_tensor[50000] = 1
            self.stop_tokens_tensor[50001] = 1
            self.stop_tokens_tensor[50002] = 1
        elif 'XLNet' in self.forward_lm.__class__.__name__:
            for i in range(9):
                self.stop_tokens_tensor[i] = 1
            # load sub tokens
            filename = '../data/tokens/xlnet_sub_tokens.txt'
            index = 0
            with open(filename, 'r') as fr:
                for line in fr:
                    words  = line.strip().split()
                    token_id = int(words[0])
                    self.sub_tokens_tensor[token_id] = 1
                    index+=1
            print('Loading {} sub tokens from {} for {}.'.format(index, filename, self.forward_lm.__class__.__name__))
        else:
            pass


    def replace(self, input_text, input_ids, keyword_vector, keywords, subword_position, sentence_length, **kwargs):

        """
        This function serves as Gibbs sampling, where input_ids is the old state and return a new state.
        This function uses BERT to compute the conditional distribution for the masked token
        and draw a sample from the conditional distribution.
        :param input_text: a sentence
        :param keyword_vector: a list of int, where 1 denotes keywords and 0 denotes non-keywords.
        :param keywords: the keywords
        :param subword_position: replace the subword_position-th subword with other subword
        :param sentence_length: the length of the input sentence in terms of subwords.
        :param kwargs:
        :return:
        """
        # copy the input_ids avoiding the original list being changed
        old_input_text = input_text
        if input_ids is None:
            # input_ids does not contains bos_token_id and eos_token_id
            input_ids = self.forward_lm_tokenizer.encode(input_text,add_special_tokens=False)
        old_input_ids = input_ids

        candidates = self.language_model.generate_candidates(old_input_ids, subword_position, keyword_vector, top_k=self.candidate_num,
                                                             sub_tokens_tensor = self.sub_tokens_tensor,
                                                             stop_tokens_tensor=self.stop_tokens_tensor,
                                                             generate_candidate_method = self.generate_candidate_method)
        candidates.append(old_input_ids)

        prev_output_tokens = old_input_ids[:]
        prev_output_tokens.pop(subword_position)
        top_k_conditional_probs, log_ppls = self.language_model.sentences_probabilities(inputs_ids=candidates, subword_position = subword_position,
        prev_output_tokens = prev_output_tokens, repetition_penalty=self.repetition_penalty, ignore_last=1)

        # draw a sample from probs and return the sampled index
        normalized_probs = self.normalize(top_k_conditional_probs[:self.candidate_num])
        # _normalized_probs = self.normalize(top_k_conditional_probs[:5])
        # sampled_id = self.sample(_normalized_probs)
        sampled_id = self.sample(normalized_probs, top_k = self.top_k )

        new_input_ids = candidates[sampled_id]
        new_input_text = self.forward_lm_tokenizer.decode(new_input_ids,clean_up_tokenization_spaces=False)

        new_input_prob = top_k_conditional_probs[sampled_id]
        new_input_log_ppl = log_ppls[sampled_id]

        old_input_prob = top_k_conditional_probs[-1]
        old_input_log_ppl = log_ppls[-1]

        if input_ids[subword_position] == new_input_ids[subword_position]:
            _is_accept = 0
        else:
            if self.classifier_model is not None:
                _input_ids = []
                _input_ids.append([self.classifier_model_tokenizer.bos_token_id] + old_input_ids + [
                    self.classifier_model_tokenizer.eos_token_id])
                _input_ids.append([self.classifier_model_tokenizer.bos_token_id] + new_input_ids + [
                    self.classifier_model_tokenizer.eos_token_id])

                _input_ids = [torch.tensor(e) for e in _input_ids]
                _mask = pad_sequence(_input_ids, batch_first=True, padding_value=-100)
                attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
                attention_mask = attention_mask.masked_fill(_mask != -100, 1)
                _input_ids = pad_sequence(_input_ids, batch_first=True, padding_value=0)
                with torch.no_grad():

                    _input_ids = _input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    # print(_input_ids, attention_mask)
                    outputs = self.classifier_model(input_ids=_input_ids, attention_mask=attention_mask)
                    logits = outputs[0]
                    probabilities = torch.softmax(logits, -1)
                    probabilities = probabilities[:, 1:, 1:]
                    old_prior = probabilities[0, subword_position, 0]
                    new_prior = probabilities[1, subword_position, 0]
                    # print(new_prior / old_prior)

                acceptance = self.acceptance_rate(1, 1, 1, 1, prior_old=old_prior, prior_new=new_prior)
                _is_accept = self.is_accept(acceptance)
            else:
                _is_accept = 1
        if _is_accept and self.check_sentence(new_input_text, keywords):
            probability = new_input_prob
            perplexity = new_input_log_ppl

        else:
            new_input_text = old_input_text
            new_input_ids = old_input_ids
            probability = old_input_prob
            perplexity = old_input_log_ppl

        return new_input_text, new_input_ids, keyword_vector, _is_accept, probability, perplexity, sentence_length

    def insert(self, input_text, input_ids, keyword_vector, keywords, subword_position, sentence_length, **kwargs ):
        """
        This function serves as a proposal distribution for Metropolis-Hastings.
        This function is meant to insert a new token before the position of input_ids.
        The process of this function is as follows:
        Step 1: Insert the [MASK] token before the position of input_ids.
        Step 2: Use BERT to draw a new token with the masked language model.
        Step 3: Compute the acceptance rate for this proposal distribution.

        acceptance_rate =  [delete_prior * p(old|new) * p(new)] / [insert_prior * p(new|old) * p(old)]

        :param input_text: a sentence
        :param keyword_vector: a list of int, where 1 denotes keywords and 0 denotes non-keywords.
        :param keywords: the keywords
        :param subword_position: insert a new subword before the subword_position-th  subword.
        :param sentence_length: the length of the input sentence in terms of subwords.
        :param insert_prior: default 1
        :param delete_prior: default 1
        :param kwargs:
        :return:
        """

        # copy the input_ids avoiding the original list being changed
        old_input_text = input_text
        if input_ids is None:
            # input_ids does not contains bos_token_id and eos_token_id
            input_ids = self.forward_lm_tokenizer.encode(input_text,add_special_tokens=False)
        old_input_ids = input_ids

        new_input_ids = input_ids[:]
        new_input_ids.insert(subword_position,0)
        _keyword_vector = keyword_vector[:]
        _keyword_vector.insert(subword_position, 0)
        candidates = self.language_model.generate_candidates(new_input_ids, subword_position, _keyword_vector,top_k=self.candidate_num,
                                                              sub_tokens_tensor=self.sub_tokens_tensor,
                                                              stop_tokens_tensor=self.stop_tokens_tensor,
                                                              generate_candidate_method=self.generate_candidate_method)

        candidates.append(old_input_ids)

        prev_output_tokens = old_input_ids[:]
        top_k_conditional_probs, log_ppls = self.language_model.sentences_probabilities(inputs_ids=candidates, subword_position = subword_position,
        prev_output_tokens = prev_output_tokens, repetition_penalty=self.repetition_penalty, ignore_last=1)


        # draw a sample from probs and return the sampled index
        normalized_probs = self.normalize(top_k_conditional_probs[:self.candidate_num])
        # _normalized_probs = self.normalize(top_k_conditional_probs[:5])
        # sampled_id = self.sample(_normalized_probs)
        # sampled_id = self.sample(normalized_probs)
        sampled_id = self.sample(normalized_probs, top_k = self.top_k )

        new_input_ids = candidates[sampled_id]
        new_input_text = self.forward_lm_tokenizer.decode(new_input_ids,clean_up_tokenization_spaces=False)

        new_input_prob = top_k_conditional_probs[sampled_id]
        new_input_log_ppl = log_ppls[sampled_id]

        old_input_prob = top_k_conditional_probs[-1]
        old_input_log_ppl = log_ppls[-1]


        p_old_state = old_input_prob
        p_new_state = new_input_prob

        p_new_state_given_old_state = normalized_probs[sampled_id]
        p_old_state_given_new_state = 1

        if self.classifier_model is not None:
            _input_ids = []
            _input_ids.append([self.classifier_model_tokenizer.bos_token_id]+old_input_ids+[self.classifier_model_tokenizer.eos_token_id])
            _input_ids.append([self.classifier_model_tokenizer.bos_token_id]+new_input_ids+[self.classifier_model_tokenizer.eos_token_id])

            _input_ids = [torch.tensor(e) for e in _input_ids]
            _mask = pad_sequence(_input_ids, batch_first=True, padding_value=-100)
            attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
            attention_mask = attention_mask.masked_fill(_mask != -100, 1)
            _input_ids = pad_sequence(_input_ids, batch_first=True, padding_value=0)
            with torch.no_grad():
                _input_ids = _input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.classifier_model(input_ids=_input_ids, attention_mask=attention_mask)
                logits = outputs[0]
                probabilities = torch.softmax(logits, -1)
                probabilities = probabilities[:,1:,1:]
                insert_prior = probabilities[0,subword_position,1]
                delete_prior = probabilities[1,subword_position,2]
                insert_prior = 1
                delete_prior = 1
        else:
            insert_prior = 1
            delete_prior = 1

        acceptance = self.acceptance_rate(p_old_state, p_new_state_given_old_state, p_new_state, p_old_state_given_new_state,
                                          prior_old=insert_prior, prior_new=delete_prior)
        _is_accept = self.is_accept(acceptance)
        # print(_is_accept, old_input_text, self.check_sentence(new_input_text, keywords),new_input_text)
        if _is_accept and self.check_sentence(new_input_text, keywords):
            probability = new_input_prob
            perplexity = new_input_log_ppl

            sentence_length +=1
            keyword_vector.insert(subword_position, 0)

        else:
            new_input_text = old_input_text
            new_input_ids = old_input_ids
            probability = old_input_prob
            perplexity = old_input_log_ppl
        return new_input_text, new_input_ids, keyword_vector, _is_accept,probability, perplexity, sentence_length

    def delete(self, input_text, input_ids, keyword_vector, keywords, subword_position, sentence_length, **kwargs):
        """

        This function serves as a proposal distribution for Metropolis-Hastings.
        This function is meant to delete the token in the position of input_ids.
        The process of this function is as follows:
        Step 1: Delete the  token in the position of input_ids.
        Step 2: Compute the acceptance rate for this proposal distribution.

        acceptance_rate =  [insert_prior * p(old|new) * p(new)] / [delete_prior * p(new|old) * p(old)]

        :param input_text: a sentence
        :param keyword_vector: a list of int, where 1 denotes keywords and 0 denotes non-keywords.
        :param keywords: the keywords
        :param subword_position: insert a new subword before the subword_position-th  subword.
        :param sentence_length: the length of the input sentence in terms of subwords.
        :param insert_prior: default 1
        :param delete_prior: default 1
        :param kwargs:
        :return:
        """

        # copy the input_ids avoiding the original list being changed
        old_input_text = input_text
        if input_ids is None:
            # input_ids does not contains bos_token_id and eos_token_id
            input_ids = self.forward_lm_tokenizer.encode(input_text,add_special_tokens=False)
        old_input_ids = input_ids


        new_input_ids = input_ids[:]
        new_input_ids.pop(subword_position)
        new_input_text = self.forward_lm_tokenizer.decode(new_input_ids,clean_up_tokenization_spaces=False)

        candidates = self.language_model.generate_candidates(old_input_ids, subword_position, keyword_vector, top_k=self.candidate_num,
                                                             sub_tokens_tensor=self.sub_tokens_tensor,
                                                             stop_tokens_tensor=self.stop_tokens_tensor,
                                                             generate_candidate_method=self.generate_candidate_method)

        index = -1
        flag = 0
        deleted_subword_id = old_input_ids[subword_position]
        for i in range(self.candidate_num):
            if deleted_subword_id == candidates[i][subword_position]:
                index=i
                break
        if index ==-1:
            candidates.append(old_input_ids)
            index = self.candidate_num
            flag = 1
        candidates.append(new_input_ids)

        prev_output_tokens = new_input_ids[:]
        top_k_conditional_probs, log_ppls = self.language_model.sentences_probabilities(inputs_ids=candidates, subword_position = subword_position,
        prev_output_tokens = prev_output_tokens, repetition_penalty=self.repetition_penalty, ignore_last=1)

        # draw a sample from probs and return the sampled index
        normalized_probs = self.normalize(top_k_conditional_probs[:self.candidate_num+flag])

        new_input_prob = top_k_conditional_probs[-1]
        new_input_log_ppl = log_ppls[-1]

        old_input_prob = top_k_conditional_probs[index]
        old_input_log_ppl = log_ppls[index]

        p_old_state = old_input_prob
        p_new_state = new_input_prob

        p_new_state_given_old_state = 1
        p_old_state_given_new_state = normalized_probs[index]


        if self.classifier_model is not None:
            _input_ids = []
            _input_ids.append([self.classifier_model_tokenizer.bos_token_id]+old_input_ids+[self.classifier_model_tokenizer.eos_token_id])
            _input_ids.append([self.classifier_model_tokenizer.bos_token_id]+new_input_ids+[self.classifier_model_tokenizer.eos_token_id])

            _input_ids = [torch.tensor(e) for e in _input_ids]
            _mask = pad_sequence(_input_ids, batch_first=True, padding_value=-100)
            attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
            attention_mask = attention_mask.masked_fill(_mask != -100, 1)
            _input_ids = pad_sequence(_input_ids, batch_first=True, padding_value=0)
            with torch.no_grad():
                _input_ids = _input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.classifier_model(input_ids=_input_ids, attention_mask=attention_mask)
                logits = outputs[0]
                probabilities = torch.softmax(logits, -1)
                probabilities = probabilities[:,1:,1:]
                delete_prior = probabilities[0,subword_position,2]
                insert_prior = probabilities[1,subword_position,1]

        else:
            insert_prior = 1
            delete_prior = 1

        acceptance = self.acceptance_rate(p_old_state, p_new_state_given_old_state, p_new_state,
                                          p_old_state_given_new_state,
                                          prior_old=delete_prior, prior_new=insert_prior)

        # draw a sample from Uniform[0,1] to determine whether accept the new_state.
        _is_accept = self.is_accept(acceptance)

        if _is_accept and self.check_sentence(new_input_text, keywords):
            probability = new_input_prob
            perplexity = new_input_log_ppl
            sentence_length -=1
            keyword_vector.pop(subword_position)
        else:
            new_input_text = old_input_text
            new_input_ids = old_input_ids
            probability = old_input_prob
            perplexity = old_input_log_ppl
        return new_input_text,new_input_ids, keyword_vector, _is_accept,probability, perplexity, sentence_length



