from torch.utils.data import Dataset
import pandas as pd
import json
import random

from datasets import load_dataset

class Pretrain(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, args, length=None):
        self.args = args
        print(f'split is {self.args.split}')
        self.tokenizer = tokenizer
        self.type_path = type_path
        self.ssm = False
        self.dataset_version = self.args.dataset_version
        if 't5' in args.model_name_or_path:
            self.model_type='T5'
        elif 'gpt2' in args.model_name_or_path:
            self.model_type='GPT2'
        dataset_v = ['small', 'full']
        ids_to_answers = None      
        if not self.dataset_version in dataset_v:
            raise Exception(f'Provided the correct dataset version among {dataset_v}')
        # dataset for continual training
        if self.args.dataset == 'recentnews':
            if type_path=='train':
                if self.dataset_version=='small':
                    self.dataset = pd.read_csv('data/recent_news_small.csv')
                elif self.dataset_version=='full':
                    self.dataset = pd.read_csv('data/recent_news_full.csv')
            elif type_path =='split':
                if self.args.split==1:
                    if self.dataset_version=='small':
                        self.dataset = pd.read_csv('data/split/recent_news_small1.csv')
                    else:
                        raise Exception('Not supporting split for full setting.')
                elif self.args.split==2:
                    if self.dataset_version=='small':
                        self.dataset = pd.read_csv('data/split/recent_news_small2.csv')
                    else:
                        raise Exception('Not supporting split for full setting.')
                else:
                    raise Exception('Currently only supporting two splits.')
            # for mixreview pretraining corpus
            elif type_path =='pretrain':
                if self.dataset_version=='small':
                    total_line = 802776
                    skip = sorted(random.sample(range(1,total_line+1),total_line-length))
                    self.dataset = pd.read_csv('data/wikipedia_pretrain_small.csv', usecols=['input', 'output', 'original'], skiprows=skip)
                elif self.dataset_version=='full':
                    total_line = 8021155
                    skip = sorted(random.sample(range(1,total_line+1),total_line-length))
                    self.dataset = pd.read_csv('data/wikipedia_pretrain_full.csv', usecols=['input', 'output'], skiprows=skip)
        # GPT-2  was  initially  pretrained  on  WebText  (Dec  2019),  which  consists  of  8  million  documents  withWikipedia  pages  excluded.
        # In  order  to  measure  the  performance  on  INVARIANTLAMA  constructed  from Wikipedia, we continually pretrain GPT-2 on a subset of Wikipedia (May 2020) for 14k global training stepsbefore CKL.
        elif self.args.dataset == 'wikitext103':
            self.dataset = pd.read_csv('data/wikipedia_pretrain_1G_final.csv')
        # dataset for evaluation
        else: 
            if self.args.dataset == 'invariantlama':
                # light tuning 5000 instances for GPT2 experiment
                if type_path =='train':
                    self.dataset = pd.read_csv('data/trex_5000.csv')
                else:
                    self.dataset = pd.read_csv('data/invariantLAMA.csv')
            elif self.args.dataset == 'updatedlama':
                if self.dataset_version == 'full':
                    rp_dir = 'data/updatedlama/updatedLAMA.csv'
                else: 
                    raise Exception('Not supporting small setting for updatedLAMA.')
                self.dataset = pd.read_csv(rp_dir)  
                with open('data/updatedlama_val_answers.json') as f:
                    ids_to_answers = json.load(f)  
            elif self.args.dataset == 'newlama':
                if self.dataset_version == 'full':
                    rp_dir = 'data/newlama/newLAMA.csv'
                else: 
                    raise Exception('Not supporting small setting for newLAMA.')
                self.dataset = pd.read_csv(rp_dir)
                with open('data/recentlama_h_val_answers.json') as f:
                    ids_to_answers = json.load(f) 
            elif self.args.dataset == 'newlama_easy' or self.args.dataset == 'newqa_easy':
                if self.dataset_version == 'small':
                    if self.args.split:
                        if self.args.split==1:
                            rp_dir = 'data/newlama/newLAMA_easy_small_split1.csv'
                        else:
                            rp_dir = 'data/newlama/newLAMA_easy_small_split2.csv'
                    else:
                        rp_dir = 'data/newlama/newLAMA_easy_small.csv'
                elif self.dataset_version == 'full':
                    rp_dir = 'data/newlama/newLAMA_easy.csv'
                # light tuning 5000 instances for GPT2 experiment
                if type_path =='train':
                    self.dataset = pd.read_csv('data/newlama/newLAMA_easy_5000.csv')
                else:
                    self.dataset = pd.read_csv(rp_dir) 
                with open('data/recentlama_val_answers.json') as f:
                    ids_to_answers = json.load(f) 
            # kilt finetuning + evaluation
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
            elif self.args.dataset== 'AY2':
                kilt_ay2 = load_dataset("kilt_tasks", name='aidayago2')
                self.dataset = kilt_ay2[type_path]
            elif self.args.dataset== 'WNED':
                kilt_wned = load_dataset("kilt_tasks", name="wned")
                self.dataset = kilt_wned[type_path]
            elif self.args.dataset== 'CWEB':
                kilt_cweb = load_dataset("kilt_tasks", name="cweb")
                self.dataset = kilt_cweb[type_path]
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
            else:
                raise NameError('Select the correct Dataset!')
        print(f'Length of dataset retrieving is.. {len(self.dataset)}')
        self.input_length = input_length
        self.output_length = output_length
        self.ids_to_answers = ids_to_answers

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        # continual pretraining
        if self.args.dataset == 'recentnews':
            if self.model_type == 'GPT2':
                input_ = example_batch['original']
                target_= example_batch['original']
            elif self.model_type == 'T5':
                input_ = example_batch['input']
                target_ = example_batch['output']
                if type(input_)!=str:
                    input_=''
                if type(target_)!=str:
                    target_=''
        elif self.args.dataset == 'wikitext103':
            input_ = example_batch['original']
            target_= example_batch['original']    
        # evaluation
        else: 
            if self.args.dataset == 'invariantlama':
                if self.model_type == 'GPT2':
                    input_pre = example_batch['input']
                    for index, word in enumerate(input_pre.split()):
                        if word == '<extra_id_0>':
                            input_pre = ' '.join(input_pre.split()[:index])
                            break
                    if self.type_path == 'train':
                        input_ = input_pre + ' ' + example_batch['output'] + '.'
                        target_= input_pre + ' ' + example_batch['output'] + '.'
                    else: 
                        input_ = input_pre
                        ground_truth_ = example_batch['output']
                        target_ = input_pre + ' ' + example_batch['output'] + '.'
                elif self.model_type == 'T5':
                    input_ = example_batch['input']
                    target_ = example_batch['output']
            elif self.args.dataset == 'updatedlama':
                input_ = example_batch['statement']
                target_ = example_batch['new_answer']
            elif self.args.dataset == 'newlama' or self.args.dataset == 'newlama_easy':
                input_ = example_batch['statement']
                target_ = example_batch['answer']
            elif self.args.dataset == 'newqa_easy':
                if self.model_type == 'GPT2':
                    if self.type_path == 'train':
                        input_ = example_batch['question'] + ' ' + example_batch['answer'].split(';')[0] + '.'
                        target_ = example_batch['question'] + ' ' + example_batch['answer'].split(';')[0] + '.'
                    else:
                        input_ = example_batch['question']
                        ground_truth_ = example_batch['answer'].split(';')[0] + '.'
                        target_ = str(example_batch['question']) + ' ' + str(example_batch['answer'])
                elif self.model_type == 'T5':
                    input_ = example_batch['question']
                    target_ = example_batch['answer'].split(';')[0]
            elif (self.args.dataset== 'TriviaQA' or self.args.dataset== 'fever' or self.args.dataset== 'AY2' or self.args.dataset== 'WNED' or self.args.dataset== 'CWEB' 
            or self.args.dataset== 'TREX' or self.args.dataset== 'zsRE' or self.args.dataset== 'NQ' or self.args.dataset== 'HotpotQA' or self.args.dataset== 'ELI5' or self.args.dataset== 'WOW'):
                input_ = example_batch['input']
                target_ = example_batch['output'][0]['answer'] 
            else:
                raise Exception('Select the correct dataset!')
        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")     
        if self.type_path == 'validation' and self.model_type =='GPT2':
            ground_truth = self.tokenizer.batch_encode_plus([str(ground_truth_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")  
        else: 
            ground_truth = None
        if (self.args.dataset == 'invariantlama' or self.args.dataset== 'TriviaQA' or self.args.dataset== 'fever' or self.args.dataset== 'AY2' or self.args.dataset== 'WNED' or self.args.dataset== 'CWEB' 
        or self.args.dataset== 'TREX' or self.args.dataset== 'zsRE' or self.args.dataset== 'NQ' or self.args.dataset== 'HotpotQA' or self.args.dataset== 'ELI5' or self.args.dataset== 'WOW'):
            labels = example_batch['id']
        elif (self.args.dataset == 'newlama' or self.args.dataset == 'updatedlama' or self.args.dataset == 'newlama_easy' or self.args.dataset == 'newqa_easy'):
            labels = example_batch['unique_id']
        else:
            labels = None                       
        return source, targets, labels, ground_truth
  
    def __getitem__(self, index):
        if (self.args.dataset== 'TriviaQA' or self.args.dataset== 'fever' or self.args.dataset== 'AY2' or self.args.dataset== 'WNED' or self.args.dataset== 'CWEB' 
        or self.args.dataset== 'TREX' or self.args.dataset== 'zsRE' or self.args.dataset== 'NQ' or self.args.dataset== 'HotpotQA' or self.args.dataset== 'ELI5' or self.args.dataset== 'WOW'):
            source, targets, labels, ground_truth = self.convert_to_features(self.dataset[index])
        else:
            source, targets, labels, ground_truth = self.convert_to_features(self.dataset.iloc[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        if labels is not None:
            label_ids = labels
        else:
            label_ids = -1
        
        if ground_truth is not None:
            ground_truth_ids = ground_truth["input_ids"].squeeze()
        else: 
            ground_truth_ids = -1

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "label_ids": label_ids, "ground_truth_ids": ground_truth_ids}