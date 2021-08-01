from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import re
import math 
#from transformers import pipeline
import os
import random
import numpy as np
import pprint

from datasets import load_dataset

class Pretrain(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, args, print_text=False):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path
        self.ssm = False
        self.dataset_version = self.args.dataset_version
        if 't5' in args.model_name_or_path:
            self.model_type='T5'
        elif 'gpt2' in args.model_name_or_path:
            self.model_type='GPT2'
        dataset_v = ['debug', 'small', 'full']
        if not self.dataset_version in dataset_v:
            raise Exception(f'Provided the correct dataset version among {dataset_v}')
        if self.args.dataset == 'recentnews':
            if type_path=='train':
                if self.dataset_version=='debug':
                    self.dataset = pd.read_csv('data/recent_news_debug.csv')
                elif self.dataset_version=='small':
                    self.dataset = pd.read_csv('data/recent_news_small.csv')
                elif self.dataset_version=='full':
                    self.dataset = pd.read_csv('data/recent_news_full.csv')
            elif type_path =='pretrain':
                if self.dataset_version=='debug':
                    self.dataset = pd.read_csv('data/wikipedia_pretrain_debug.csv')
                else:
                    self.dataset = pd.read_csv('data/wikipedia_pretrain.csv')
            else:
                self.dataset = self.get_recent_val(-1,-1) #Getting validation data for both LAMA-entity and RecentProbe
        elif self.args.dataset == 'lama':
            if self.dataset_version=='debug':
                original = pd.read_csv('data/lama_template_debug.csv')
                if type_path =='train':
                    self.dataset = pd.read_csv('data/lama_template_debug.csv', nrows=int(len(original)*self.args.finetuning_ratio))
                else:
                    self.dataset = pd.read_csv('data/lama_template_debug.csv', skiprows=lambda i:i>0 and i<=int(len(original)*self.args.finetuning_ratio))
            else:
                original = pd.read_csv('data/lama_template.csv')
                if type_path =='train':
                    self.dataset = pd.read_csv('data/lama_template.csv', nrows=int(len(original)*self.args.finetuning_ratio))
                else:
                    self.dataset = pd.read_csv('data/lama_template.csv', skiprows=lambda i:i>0 and i<=int(len(original)*self.args.finetuning_ratio))
        elif self.args.dataset == 'recentprobe':
            if self.dataset_version=='debug':
                original = pd.read_csv('data/recentprobe_small.csv')
                if type_path =='train':
                    self.dataset = pd.read_csv('data/recentprobe_small.csv', nrows=int(len(original)*self.args.finetuning_ratio))
                else:
                    self.dataset = pd.read_csv('data/recentprobe_small.csv', skiprows=lambda i:i>0 and i<=int(len(original)*self.args.finetuning_ratio))
            else:
                original = pd.read_csv('data/recentprobe_small.csv')
                if type_path =='train':
                    self.dataset = pd.read_csv('data/recentprobe_small.csv', nrows=int(len(original)*self.args.finetuning_ratio))
                else:
                    self.dataset = pd.read_csv('data/recentprobe_small.csv', skiprows=lambda i:i>0 and i<=int(len(original)*self.args.finetuning_ratio))       
        elif self.args.dataset== 'TriviaQA':
            # Get the KILT task datasets
            kilt_triviaqa = load_dataset("kilt_tasks", name="triviaqa_support_only")

            # Most tasks in KILT already have all required data, but KILT-TriviaQA only provides the question IDs, not the questions themselves.
            # Thankfully, we can get the original TriviaQA data with:
            trivia_qa = load_dataset('trivia_qa', 'unfiltered.nocontext')

            # The KILT IDs can then be mapped to the TriviaQA questions with:
            triviaqa_map = {}

            def add_missing_data(x, trivia_qa_subset, triviaqa_map):
                i = triviaqa_map[x['id']]
                x['input'] = trivia_qa_subset[i]['question']
                #x['output']['original_answer'] = trivia_qa_subset[i]['answer']['value']
                return x

            for k in ['train', 'validation', 'test']:
                triviaqa_map = dict([(q_id, i) for i, q_id in enumerate(trivia_qa[k]['question_id'])])
                kilt_triviaqa[k] = kilt_triviaqa[k].filter(lambda x: x['id'] in triviaqa_map)
                kilt_triviaqa[k] = kilt_triviaqa[k].map(add_missing_data, fn_kwargs=dict(trivia_qa_subset=trivia_qa[k], triviaqa_map=triviaqa_map))
            self.dataset = kilt_triviaqa[type_path]    
            with open('data/tqa_val_answers.json') as f:
                ids_to_answers = json.load(f)
        elif self.args.dataset== 'fever':
            kilt_fever = load_dataset("kilt_tasks", name="fever")
            self.dataset = kilt_fever[type_path]
            ids_to_answers = None
        elif self.args.dataset== 'AY2':
            kilt_ay2 = load_dataset("kilt_tasks", name='aidayago2')
            self.dataset = kilt_ay2[type_path]
            ids_to_answers = None
        elif self.args.dataset== 'WNED':
            kilt_wned = load_dataset("kilt_tasks", name="wned")
            self.dataset = kilt_wned[type_path]
            ids_to_answers = None
        elif self.args.dataset== 'CWEB':
            kilt_cweb = load_dataset("kilt_tasks", name="cweb")
            self.dataset = kilt_cweb[type_path]
            ids_to_answers = None
        elif self.args.dataset== 'TREX':
            kilt_trex = load_dataset("kilt_tasks", name="trex")
            self.dataset = kilt_trex[type_path]
            with open('data/trex_val_answers.json') as f:
                ids_to_answers = json.load(f)  
        elif self.args.dataset== 'zsRE':
            kilt_zsre = load_dataset("kilt_tasks", name="structured_zeroshot")
            self.dataset = kilt_zsre[type_path]
            with open('data/zsre_val_answers.json') as f:
                ids_to_answers = json.load(f)  
        elif self.args.dataset== 'NQ':
            kilt_nq = load_dataset("kilt_tasks", name="nq")
            self.dataset = kilt_nq[type_path]
            with open('data/nq_val_answers.json') as f:
                ids_to_answers = json.load(f)  
        elif self.args.dataset== 'HotpotQA':
            kilt_hotqa = load_dataset("kilt_tasks", name="hotpotqa")
            self.dataset = kilt_hotqa[type_path]
            with open('data/hotpotqa_val_answers.json') as f:
                ids_to_answers = json.load(f)  
        elif self.args.dataset== 'ELI5':
            kilt_eli5 = load_dataset("kilt_tasks", name="eli5")
            self.dataset = kilt_eli5[type_path]
            with open('data/eli5_val_answers.json') as f:
                ids_to_answers = json.load(f)  
        elif self.args.dataset== 'WOW':
            kilt_wow = load_dataset("kilt_tasks", name="wow", ignore_verifications=True)
            self.dataset = kilt_wow[type_path]
            ids_to_answers = None
        else:
            raise NameError('Select the correct Dataset!')
        print(f'length of dataset: {len(self.dataset)}')
        if self.args.dataset == 'recentnews' and type_path=='validation':
            self.input_length = 50
            self.output_length = 10
        else:
            self.input_length = input_length
            self.output_length = output_length
        sentinels=[]
        for i in range(100):
            sentinels.append(f'<extra_id_{i}>')
        self.sentinels = sentinels
        self.ids_to_answers = ids_to_answers
        
    def get_recent_val(self, lama_num, recent_num):
        if self.dataset_version=='debug':
            recent = pd.read_csv('data/recentprobe_m_debug.csv')
            lama = pd.read_csv('data/lama_template_debug.csv')
        elif 'small' in self.dataset_version:
            if self.dataset_version=='small':
                recent = pd.read_csv('data/recentprobe_small.csv')
            elif self.dataset_version=='small_m':
                recent = pd.read_csv('data/recentprobe_m_small.csv')
            else:
                raise Exception('Select the proper dataset version.')
            lama = pd.read_csv('data/lama_template.csv')         
        elif self.dataset_version=='full':
            recent = pd.read_csv('data/recentprobe_m_full.csv')
            lama = pd.read_csv('data/lama_template.csv')
        dataset = []
        for index, row in lama.iterrows():
            dataset.append(row)
        for index, row in recent.iterrows():
            dataset.append(row)
        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')
        text = re.sub("\\[.*\\]",'',text)
        return text

    def convert_to_features(self, example_batch, index=None):
        if self.args.dataset == 'recentnews':
            if self.model_type == 'GPT2':
                if self.type_path=='train' or self.type_path=='pretrain':
                    input_ = example_batch['original']
                    target_= example_batch['original']
            elif self.model_type == 'T5':
                input_ = example_batch['input']
                target_ = example_batch['output']
                if type(target_)!=str:
                    target_=''
        elif self.args.dataset == 'lama':
            if self.model_type == 'GPT2':
                input_pre = example_batch['input']
                for index, word in enumerate(input_pre.split()):
                    if word == '<extra_id_0>':
                        input_pre = ' '.join(input_pre.split()[:index])
                        break
                if self.type_path == 'train':
                    input_ = input_pre + ' ' + example_batch['output'] 
                    target_= input_pre + ' ' + example_batch['output'] 
                else: 
                    input_ = input_pre
                    label_ = example_batch['output']
                    target_ = input_pre + ' ' + example_batch['output']
            elif self.model_type == 'T5':
                input_ = example_batch['input']
                target_ = example_batch['output']
        elif self.args.dataset == 'recentprobe':
            if self.model_type == 'GPT2':
                if self.type_path == 'train':
                    input_ = example_batch['question'] + ' ' + example_batch['output']
                    target_ = example_batch['question'] + ' ' + example_batch['output']
                else:
                    input_ = example_batch['question']
                    label_ = example_batch['output']
                    target_ = example_batch['question'] + ' ' + example_batch['output']
            elif self.model_type == 'T5':
                input_ = example_batch['question']
                target_ = example_batch['output']
        elif (self.args.dataset== 'TriviaQA' or self.args.dataset== 'fever' or self.args.dataset== 'AY2' or self.args.dataset== 'WNED' or self.args.dataset== 'CWEB' 
        or self.args.dataset== 'TREX' or self.args.dataset== 'zsRE' or self.args.dataset== 'NQ' or self.args.dataset== 'HotpotQA' or self.args.dataset== 'ELI5' or self.args.dataset== 'WOW'):
            input_ = example_batch['input']
            target_ = example_batch['output'][0]['answer']
        else:
            raise Exception('Select the correct dataset!')
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")     
        if self.type_path == 'validation' and self.model_type =='GPT2':
            labels = self.tokenizer.batch_encode_plus([label_], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")   
        elif (self.args.dataset== 'TriviaQA' or self.args.dataset== 'fever' or self.args.dataset== 'AY2' or self.args.dataset== 'WNED' or self.args.dataset== 'CWEB' 
        or self.args.dataset== 'TREX' or self.args.dataset== 'zsRE' or self.args.dataset== 'NQ' or self.args.dataset== 'HotpotQA' or self.args.dataset== 'ELI5' or self.args.dataset== 'WOW'):
            labels = example_batch['id']
            answer_lst = []
            for entry in example_batch['output']:
                answer_lst.append(entry['answer'])
        else:
            labels = None                         
        return source, targets, labels
  
    def __getitem__(self, index):
        if (self.args.dataset == 'recentnews' and self.type_path =='validation'):
            source, targets, labels = self.convert_to_features(self.dataset[index],index)
        elif (self.args.dataset== 'TriviaQA' or self.args.dataset== 'fever' or self.args.dataset== 'AY2' or self.args.dataset== 'WNED' or self.args.dataset== 'CWEB' 
        or self.args.dataset== 'TREX' or self.args.dataset== 'zsRE' or self.args.dataset== 'NQ' or self.args.dataset== 'HotpotQA' or self.args.dataset== 'ELI5' or self.args.dataset== 'WOW'):
            source, targets, labels = self.convert_to_features(self.dataset[index])
        else:
            source, targets, labels = self.convert_to_features(self.dataset.iloc[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        if labels is not None:
            if (self.args.dataset== 'TriviaQA' or self.args.dataset== 'fever' or self.args.dataset== 'AY2' or self.args.dataset== 'WNED' or self.args.dataset== 'CWEB' 
            or self.args.dataset== 'TREX' or self.args.dataset== 'zsRE' or self.args.dataset== 'NQ' or self.args.dataset== 'HotpotQA' or self.args.dataset== 'ELI5' or self.args.dataset== 'WOW'):
                label_ids = labels
            else:
                label_ids = labels["input_ids"].squeeze()
        else:
            label_ids = -1

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "label_ids": label_ids}