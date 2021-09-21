import pytorch_lightning as pl
from models.Modular_T5 import T5ForConditionalGeneration as T5_Modular
from models.Modular_Small_T5 import T5ForConditionalGeneration as T5_Modular_Small
from models.Modular_Small_T52 import T5ForConditionalGeneration as T5_Modular_Small2
from models.Kadapter_T5 import T5ForConditionalGeneration as T5_Kadapter
from models.Kadapter_T52 import T5ForConditionalGeneration as T5_Kadapter2
from models.Lora_T5 import T5ForConditionalGeneration as T5_Lora
from models.Lora_T52 import T5ForConditionalGeneration as T5_Lora2
from models.RecAdam import RecAdam, anneal_function
from transformers import T5Config
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from transformers import (
    AdamW,
    Adafactor,
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5EncoderModel,
    T5Config,
    get_linear_schedule_with_warmup
)
import torch
from Datasets import Pretrain
from torch.utils.data import RandomSampler, SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from rouge import Rouge
from collections import Counter

import random
import argparse
import time
import re
import numpy as np
import string
from string import punctuation
import os
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
import copy

class T5(pl.LightningModule):
    def __init__(self, hparams):
        super(T5, self).__init__()
        self.save_hyperparameters(hparams)

        self.mix_ratio = 4
        self.mix_decay = 0.7
        self.epoch = 0
        self.pruning_params = {}

        if hparams.method=='modular':
            self.model = T5_Modular.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='modular_small':
            self.model = T5_Modular_Small.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='modular_small2': 
            previous_model_dir = (hparams.output_dir)[:len(hparams.output_dir)-1]
            self.model = T5_Modular_Small2.from_pretrained(previous_model_dir)
        elif hparams.method=='kadapter':
            self.model = T5_Kadapter.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='kadapter2':
            previous_model_dir = (hparams.output_dir)[:len(hparams.output_dir)-1]
            self.model = T5_Kadapter2.from_pretrained(previous_model_dir)
        elif hparams.method=='lora':
            self.model = T5_Lora.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='lora2':
            previous_model_dir = (hparams.output_dir)[:len(hparams.output_dir)-1]
            self.model = T5_Lora2.from_pretrained(previous_model_dir)
        elif hparams.method=='recadam':
            self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
            self.pretrained_model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
            self.freeze_params(self.pretrained_model) #Freezing pretrained model
        elif hparams.method=='recadam2':
            self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
            self.pretrained_model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
            self.freeze_params(self.pretrained_model) #Freezing pretrained model
        elif hparams.method=='prune2':
            previous_model_dir = (hparams.output_dir)[:len(hparams.output_dir)-1]
            self.model = T5ForConditionalGeneration.from_pretrained(previous_model_dir)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        
        #Freezing only encoder or the whole model
        if hparams.freeze_level==0: # Do not freeze any parameters
            print('Not freezing any parameters!')
        elif hparams.freeze_level==1: # Freeze encoder only
            self.freeze_params(self.model.get_encoder())
        elif hparams.freeze_level==2: # Freeze encoder and decoder
            self.freeze_params(self.model) 

        if hparams.method=='modular_small':
            for name, param in self.model.named_parameters():
                if 'encoder_modular' in name:
                    param.requires_grad = True
        elif hparams.method=='modular_small2':
            for name, param in self.model.named_parameters():
                if 'encoder_modular2' in name or name=='encoder_modular_projection':
                    param.requires_grad = True
        elif hparams.method=='kadapter':
            # Unfreezing the parameters used for lora
            for name, param in self.model.named_parameters():
                if 'kadapter' in name:
                    param.requires_grad = True
        elif hparams.method=='lora':
            # Unfreezing the parameters used for lora
            for name, param in self.model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
        elif hparams.method=='kadapter2':
            # Unfreezing the parameters used for lora
            for name, param in self.model.named_parameters():
                if 'kadapter2' in name:
                    param.requires_grad = True
        elif hparams.method=='lora2':
            # Unfreezing the parameters used for lora
            for name, param in self.model.named_parameters():
                if 'lora2' in name:
                    param.requires_grad = True
        elif hparams.method=='prune' or hparams.method=='prune2':
            # Important: This property activates manual optimization.
            self.automatic_optimization = False
            trainable_param_cnt=0
            pruner = prune.L1Unstructured(amount=hparams.prune_ratio)
            for name, param in self.model.named_parameters():
                if 'SelfAttention' in name and not ('decoder' in name):
                    ones = torch.ones(param.data.size())
                    zeros = torch.zeros(param.data.size())
                    pruned = pruner.prune(param.data)
                    pruned = torch.where(pruned!=0, pruned, ones)
                    pruned = torch.where(pruned==1, pruned, zeros)
                    trainable_param_cnt+=torch.nonzero(pruned).size(0)
                    self.pruning_params[name] = pruned
            print(f'Trainable parameters count: {trainable_param_cnt}')
            self.log("trainable_param_count", trainable_param_cnt)
        elif hparams.method=='prune_new':
            self.automatic_optimization = False
            pruner = prune.L1Unstructured(amount=1-hparams.prune_ratio)
            for name, param in self.model.named_parameters():
                if 'SelfAttention' in name and not ('decoder' in name):
                    zeros = torch.zeros(param.data.size())
                    rec = torch.abs(1 / param.data)   
                    out = F.normalize(rec)
                    pruned = pruner.prune(out)
                    self.pruning_params[name] = pruned
        elif 'prune_lw' in hparams.method:
            self.automatic_optimization = False
            configs = T5Config(model_type=hparams.model_name_or_path)
            if "small" in hparams.model_name_or_path:
                num_enc_layers = 8
            else:
                num_enc_layers = 24
            for name, param in self.model.named_parameters():
                if 'SelfAttention' in name and not ('decoder' in name):
                    name_s = name.split('.')
                    layer_num = int(name_s[2]) + 1
                    if 'dec' in hparams.method:
                        importance = layer_num/num_enc_layers
                        p_ratio = 1 - ((hparams.prune_ratio * 2 ) * importance)
                    elif 'inc' in hparams.method:
                        importance = ( num_enc_layers - (layer_num - 1) ) / num_enc_layers
                        p_ratio = 1 - ((hparams.prune_ratio * 2) * importance)
                    elif 'thin' in hparams.method:
                        num_layers = num_enc_layers / 2
                        if layer_num <= num_layers:
                            importance = ( num_layers - (layer_num - 1) ) / num_layers
                        else:
                            layer_n = layer_num - num_layers
                            importance = layer_n/num_layers
                        p_ratio = 1 - ((hparams.prune_ratio * 2) * importance)
                    elif 'fat' in hparams.method:
                        num_layers = num_enc_layers / 2
                        if layer_num <= num_layers:
                            importance = layer_num/num_layers
                        else:
                            layer_n = layer_num - num_layers
                            importance = ( num_layers - (layer_n - 1) ) / num_layers
                        p_ratio = 1 - ((hparams.prune_ratio * 2) * importance)
                    pruner = prune.L1Unstructured(amount=p_ratio)
                    zeros = torch.zeros(param.data.size())
                    rec = torch.abs(1 / param.data)   
                    out = F.normalize(rec)
                    pruned = pruner.prune(out)
                    self.pruning_params[name] = pruned          
        elif 'layerwiselr' in hparams.method:
            self.automatic_optimization = False 
            configs = T5Config(model_type=hparams.model_name_or_path)
            if "small" in hparams.model_name_or_path:
                num_enc_layers = 8
            else:
                num_enc_layers = 24
            for name, param in self.model.named_parameters():
                if 'SelfAttention' in name and not ('decoder' in name):
                    name_s = name.split('.')
                    layer_num = int(name_s[2]) + 1
                    if 'dec' in hparams.method:
                        importance = layer_num/num_enc_layers
                    else:
                        importance = ( num_enc_layers - (layer_num - 1) ) / num_enc_layers
                    self.pruning_params[name] = importance
        elif 'prune_iter' in hparams.method:
            self.automatic_optimization = False
        self.output_dir = self.hparams.output_dir
            
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "validation": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
        self.em_score_list = []
        self.subset_score_list =[]

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        
        def rid_of_specials(text):
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            return text

        return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))

    def exact_match_score(self, prediction, ground_truth):
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def approx_match_score(self, prediction, ground_truth):
        answer = self.normalize_answer(prediction) 
        gt = self.normalize_answer(ground_truth)
        match = 0
        gt_words = gt.split(" ")
        for word in gt_words:
            if word in answer:
                match = 1
                return match
        return match
    
    def accuracy_match_score(self, prediction, ground_truth):
        return int(prediction.strip() == ground_truth.strip())

    def _rougel_score(self, prediction, ground_truth):
        rouge = Rouge()
        # no normalization
        try:
            scores = rouge.get_scores(prediction, ground_truth, avg=True)
        except ValueError:  # "Hypothesis is empty."
            return 0.0
        return scores["rouge-l"]["f"]
    
    def _f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def _f1_score_zeroshot(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        return num_same, len(prediction_tokens), len(ground_truth_tokens)

    def calculate_scores(self, predictions, ground_truths):
        em_score = 0
        subset_match_score = 0
        accuracy = 0
        
        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            em_score +=  self.exact_match_score(prediction, ground_truth)
            subset_match_score += self.approx_match_score(prediction, ground_truth)
            accuracy += self.accuracy_match_score(prediction, ground_truth)
        
        em_score /= len(predictions)
        subset_match_score /= len(predictions)
        accuracy /= len(predictions)
        return em_score*100, subset_match_score*100, accuracy*100
    
    def calculate_scores_multipleanswers(self, predictions, ground_truths, ids):
        em_score = 0
        subset_match_score = 0
        accuracy_score = 0
        
        for i in range(len(predictions)):
            unique_id = ids[i]
            answers = self.ids_to_answers[unique_id]
            #ground_truths = ground_truths[i]
            prediction = predictions[i]
            em_correct = False
            sm_correct = False
            accuracy_correct = False
            for answer in answers:
                accuracy = self.accuracy_match_score(prediction, answer)
                if accuracy == 1:
                    accuracy_correct = True
                em  = self.exact_match_score(prediction, answer)
                if em == 1:
                    em_correct = True
                sm = self.approx_match_score(prediction, answer)
                if sm == 1:
                    sm_correct = True
            if accuracy_correct:
                accuracy_score+=1
            if em_correct:
                em_score+=1
            if sm_correct:
                subset_match_score+=1
        
        accuracy_score /= len(predictions)
        em_score /= len(predictions)
        subset_match_score /= len(predictions)
        return em_score*100, subset_match_score*100, accuracy_score*100

    def calculate_rouge_multipleanswers(self, predictions, ground_truths, ids):
        rouge_score = 0 
        for i in range(len(predictions)):
            unique_id = ids[i]
            answers = self.ids_to_answers[unique_id]
            #ground_truths = ground_truths[i]
            prediction = predictions[i]
            rouge_local_score = 0
            for answer in answers:
                rouge = self._rougel_score(prediction, answer)
                if rouge > rouge_local_score:
                    rouge_local_score = rouge 
            rouge_score += rouge_local_score
        rouge_score /= len(predictions)
        return rouge_score*100

    def calculate_f1_scores(self, predictions, ground_truths, ids):
        f1_score = 0 
        for i in range(len(predictions)):
            unique_id = ids[i]
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            f1_score += self._f1_score(prediction, ground_truth)

        f1_score /= len(predictions)
        return f1_score*100


    def bleu(self, gen, ref):
        ''' 
        calculate pair wise bleu score. uses nltk implementation
        Args:
            references : a list of reference sentences 
            candidates : a list of candidate(generated) sentences
        Returns:
            bleu score(float)
        '''
        ref_bleu = []
        gen_bleu = []
        for l in gen:
            gen_bleu.append(l.split())
        for i,l in enumerate(ref):
            ref_bleu.append([l.split()])
        cc = SmoothingFunction()
        score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
        return score_bleu

    def get_dataset(self, tokenizer, type_path, num_samples, args, length=None):
        if args.mode == 'pretrain' or args.mode == 'finetune':
            dataset = Pretrain(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples,  input_length=args.max_input_length, 
                            output_length=args.max_output_length, args=args, length=length)
            self.ids_to_answers = dataset.ids_to_answers
            return dataset
        else:
            raise NameError('Select the correct mode please.')

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False
            
            
    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)
    
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))
    

    def is_logger(self):
        return self.trainer.global_rank <= 0
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
    )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss
    
    
    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)
    
     
    def _generative_step(self, batch, batch_idx):
        
        val_num = batch_idx * len(batch["source_ids"]) * self.hparams.n_gpu #For 2 val logs
        t0 = time.time()
        
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=10,
            num_beams=2,
            early_stopping=True
        )
        
        preds = self.ids_to_clean_text(generated_ids)
        targets = self.ids_to_clean_text(batch["target_ids"])
        ids = batch["label_ids"]
            
        gen_time = (time.time() - t0) / batch["source_ids"].shape[0]  
    
        loss = self._step(batch)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        summ_len = np.mean(self.lmap(len, generated_ids))
        if self.hparams.dataset == 'TriviaQA' or self.hparams.dataset == 'zsRE' or self.hparams.dataset == 'TREX' or self.hparams.dataset == 'NQ' or self.hparams.dataset == 'HotpotQA':
            em_score, subset_match_score, accuracy = self.calculate_scores_multipleanswers(preds, targets, ids)
            rouge_score = 0
            f1_score = 0
        elif self.hparams.dataset =='ELI5':
            rouge_score = self.calculate_rouge_multipleanswers(preds, targets, ids)
            em_score = 0
            subset_match_score = 0
            accuracy = 0
            f1_score = 0
        elif self.hparams.dataset =='WOW':
            f1_score = self.calculate_f1_scores(preds, targets, ids)
            rouge_score = 0
            em_score = 0
            subset_match_score = 0
            accuracy = 0
        else:
            em_score, subset_match_score, accuracy = self.calculate_scores(preds, targets)
            rouge_score = 0
            f1_score = 0
        #bleu_score = self.bleu(preds,targets)
        self.em_score_list.append(em_score)
        self.subset_score_list.append(subset_match_score)
        
        em_score = torch.tensor(em_score,dtype=torch.float32)
        subset_match_score = torch.tensor(subset_match_score,dtype=torch.float32)
        accuracy = torch.tensor(accuracy,dtype=torch.float32)
        rouge_score = torch.tensor(rouge_score, dtype=torch.float32)
        f1_score = torch.tensor(f1_score, dtype=torch.float32)
        #bleu_score = torch.tensor(bleu_score,dtype=torch.float32)
        if self.hparams.dataset_version=='debug':
            lama_len = 1202
        else:
            lama_len = 20725
        if self.hparams.dataset=='recentnews':
            if val_num < lama_len:
                self.log('lama_em_score', em_score, prog_bar=True, logger=True)
                self.log('lama_subset_match_score', subset_match_score, prog_bar=True, logger=True)
            else:
                self.log('recent_em_score', em_score, prog_bar=True, logger=True)
                self.log('recent_subset_match_score', subset_match_score, prog_bar=True, logger=True)
        elif self.hparams.dataset == 'ELI5':
            self.log('rouge_score', rouge_score, prog_bar=True, logger=True)
        elif self.hparams.dataset == 'WOW':
            self.log('f1_score', f1_score, prog_bar=True, logger=True)
        else:
            self.log('accuracy', accuracy, prog_bar=True, logger=True)
            self.log('em_score', em_score, prog_bar=True, logger=True)
            self.log('subset_match_score', subset_match_score, prog_bar=True, logger=True)
        #self.log('bleu_score', bleu_score, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        if 'prune' in self.hparams.method or 'layerwiselr' in self.hparams.method:
            sch = self.lr_schedulers()
            opt = self.optimizers()
            loss = self._step(batch)
            self.manual_backward(loss)
            if (batch_idx + 1) % self.hparams.gradient_accumulation_steps == 0:
                if self.hparams.method=='prune_iter':
                    self.iter_prune()
                elif self.hparams.method=='prune_iter_new':
                    self.iter_prune_new()
                else:
                    self.zero_grads()
                opt.step()
                sch.step()
                opt.zero_grad()
        else:
            loss = self._step(batch)
        self.log("loss", loss)
        return loss

    def iter_prune(self):
        pruner = prune.L1Unstructured(amount=self.hparams.prune_ratio)
        for name, param in self.model.named_parameters():
            if 'SelfAttention' in name and not ('decoder' in name):
                device = 'cuda:'+str(param.grad.get_device())
                ones = torch.ones(param.data.size()).to(device=device)
                zeros = torch.zeros(param.data.size()).to(device=device)
                pruned = pruner.prune(param.data)
                pruned = torch.where(pruned!=0, pruned, ones)
                pruned = torch.where(pruned==1, pruned, zeros)
                #pruned = pruned.to(device=device)
                param.grad = param.grad * pruned

    def iter_prune_new(self):
        pruner = prune.L1Unstructured(amount=1-self.hparams.prune_ratio)
        for name, param in self.model.named_parameters():
            if 'SelfAttention' in name and not ('decoder' in name):
                rec = torch.abs(1 / param.data)   
                out = F.normalize(rec)
                pruned = pruner.prune(out)
                param.grad = param.grad * pruned

    def zero_grads(self):
        for name, param in self.model.named_parameters():
            if name in self.pruning_params:
                pruned = self.pruning_params[name]        
                if not ('layerwiselr' in self.hparams.method):
                    device = 'cuda:'+str(param.grad.get_device())
                    pruned = pruned.to(device=device)
                param.grad = param.grad * pruned

    def on_train_epoch_start(self):
        if self.hparams.method=='mixreview':
            train_set = self.train_dataloader().dataset
        if self.hparams.method=='prune_iter_e':
            pruner = prune.L1Unstructured(amount=self.hparams.prune_ratio)
            for name, param in self.model.named_parameters():
                if not ('layer_norm' in name) and not ('decoder' in name):
                    device = 'cuda:'+str(param.get_device())
                    ones = torch.ones(param.data.size()).to(device=device)
                    zeros = torch.zeros(param.data.size()).to(device=device)
                    pruned = pruner.prune(param.data)
                    pruned = torch.where(pruned!=0, pruned, ones)
                    pruned = torch.where(pruned==1, pruned, zeros)
                    self.pruning_params[name] = pruned
        self.epoch+=1
    
    def on_train_end(self):
        if self.hparams.mode == 'pretrain':
            if self.hparams.method=='recadam':
                self.pretrained_model = self.model
            elif self.hparams.method=='kadapter' or self.hparams.method=='lora' or self.hparams.method=='prune' or self.hparams.method=='modular_small':
                self.model.save_pretrained(self.hparams.output_dir)

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch, batch_idx)

    def configure_optimizers(self, train_len=None):
        "Prepare optimizer and schedule (linear warmup and decay)"
        if self.hparams.method=='recadam':
            no_decay = ["bias", "LayerNorm.weight"]
            model_type = 't5'
            recadam_anneal_w = 1.0
            recadam_anneal_fun = 'sigmoid'
            recadam_anneal_k = 0.5
            recadam_anneal_t0 = 250
            recadam_pretrain_cof = 5000.0
            new_model = self.model
            pretrained_model = self.pretrained_model
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            not any(nd in n for nd in no_decay) and model_type in n],
                    "weight_decay": self.hparams.weight_decay,
                    "anneal_w": recadam_anneal_w,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        not any(nd in p_n for nd in no_decay) and model_type in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            not any(nd in n for nd in no_decay) and model_type not in n],
                    "weight_decay": self.hparams.weight_decay,
                    "anneal_w": 0.0,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        not any(nd in p_n for nd in no_decay) and model_type not in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            any(nd in n for nd in no_decay) and model_type in n],
                    "weight_decay": 0.0,
                    "anneal_w": recadam_anneal_w,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        any(nd in p_n for nd in no_decay) and model_type in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            any(nd in n for nd in no_decay) and model_type not in n],
                    "weight_decay": 0.0,
                    "anneal_w": 0.0,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        any(nd in p_n for nd in no_decay) and model_type not in p_n]
                }
            ]
            optimizer = RecAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,
                                anneal_fun=recadam_anneal_fun, anneal_k=recadam_anneal_k,
                                anneal_t0=recadam_anneal_t0, pretrain_cof=recadam_pretrain_cof)
        else:
            model = self.model
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            
            optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False)

        #self.optimizer = optimizer
        if self.hparams.use_lr_scheduling:
            len_data = len(self.train_dataloader())
            #denomniator = self.hparams.n_gpu * self.hparams.gradient_accumulation_steps
            denomniator = (self.hparams.n_gpu * self.hparams.gradient_accumulation_steps) // 3 # Do not decay learning rate to 0 for small set 
            if self.hparams.dataset_version=='full':
                denomniator = (self.hparams.n_gpu * self.hparams.gradient_accumulation_steps) // 2 # Do not decay learning rate to 0 for full set 
            steps_per_epoch = ( len_data // denomniator ) + 1
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=steps_per_epoch, pct_start=0.1, epochs=self.hparams.num_train_epochs, anneal_strategy='linear', cycle_momentum=False)
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "name": "learning rate"}]
        else:
            return [optimizer]

    def train_dataloader(self):
        n_samples = self.n_obs['train']
        if self.hparams.method=='mixreview':
            if self.hparams.split_num==2:
                train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="split", num_samples=n_samples, args=self.hparams)
            else:
                train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="train", num_samples=n_samples, args=self.hparams)
            train_len = len(train_dataset)
            mix_len = int(len(train_dataset) * self.mix_ratio * (self.mix_decay ** self.epoch))
            # mix_len=3000 #only for debug
            pretrain_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="pretrain", num_samples=n_samples, args=self.hparams, length=mix_len)  
            if self.hparams.split==2:
                args2 = copy.deepcopy(self.hparams)
                args2.split = 1
                previous_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="split", num_samples=n_samples, args=args2)  
                pretrain_dataset = ConcatDataset([previous_dataset,pretrain_dataset])
            mixed_dataset = ConcatDataset([train_dataset,pretrain_dataset])
            print("mix len is ", mix_len)
            sampler=RandomSampler(mixed_dataset)
            dataloader = DataLoader(mixed_dataset, sampler = sampler, batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
            print("dataset length is ", len(dataloader.dataset))
        elif self.hparams.split_num==2:
            train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="split", num_samples=n_samples, args=self.hparams)
            sampler = RandomSampler(train_dataset)
            dataloader = DataLoader(train_dataset, sampler=sampler,  batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)  
        else:     
            train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="train", num_samples=n_samples, args=self.hparams)
            sampler = RandomSampler(train_dataset)
            dataloader = DataLoader(train_dataset, sampler=sampler,  batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    # def val_dataloader(self):
    #     n_samples = self.n_obs['validation']
    #     validation_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="validation", num_samples=n_samples, args=self.hparams,)
    #     #sampler=RandomSampler(validation_dataset)
    #     dataloader = DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)
    #     return dataloader
    
    def test_dataloader(self):
        n_samples = self.n_obs['test']
        test_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="test", num_samples=n_samples, args=self.hparams)
        
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)