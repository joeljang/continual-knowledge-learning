import pytorch_lightning as pl
from transformers import (
    AdamW,
    Adafactor,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup
)
import torch
from datasets import Pretrain
from torch.utils.data import RandomSampler
from torch.utils.data import Dataset, DataLoader

import argparse
import time
import re
import numpy as np
import string
from string import punctuation
import os
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        #self.config = T5Config.from_pretrained(hparams.model_name_or_path)
        #self.model = T5ForConditionalGeneration(self.config)
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        
        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())
        
        self.step_count = 0
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

    def calculate_scores(self, predictions, ground_truths):
        em_score = 0
        subset_match_score = 0
        
        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            em_score +=  self.exact_match_score(prediction, ground_truth)
            subset_match_score += self.approx_match_score(prediction, ground_truth)
        
        em_score /= len(predictions)
        subset_match_score /= len(predictions)
        return em_score*100, subset_match_score*100

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

    def get_dataset(self, tokenizer, type_path, num_samples, args):
        if args.mode == 'pretrain':
            return Pretrain(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples,  input_length=args.max_input_length, 
                            output_length=args.max_output_length, args=args)
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
            
        gen_time = (time.time() - t0) / batch["source_ids"].shape[0]  
    
        loss = self._step(batch)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        summ_len = np.mean(self.lmap(len, generated_ids))
        em_score, subset_match_score = self.calculate_scores(preds, targets)
        #bleu_score = self.bleu(preds,targets)
        self.em_score_list.append(em_score)
        self.subset_score_list.append(subset_match_score)
        
        em_score = torch.tensor(em_score,dtype=torch.float32)
        subset_match_score = torch.tensor(subset_match_score,dtype=torch.float32)
        #bleu_score = torch.tensor(bleu_score,dtype=torch.float32)
        if self.hparams.dataset=='recentnews':
            if val_num < 20720:
                self.log('lama_em_score', em_score, prog_bar=True, logger=True)
                self.log('lama_subset_match_score', subset_match_score, prog_bar=True, logger=True)
            else:
                self.log('recent_em_score', em_score, prog_bar=True, logger=True)
                self.log('recent_subset_match_score', subset_match_score, prog_bar=True, logger=True)
        else:
            self.log('em_score', em_score, prog_bar=True, logger=True)
            self.log('subset_match_score', subset_match_score, prog_bar=True, logger=True)
        #self.log('bleu_score', bleu_score, prog_bar=True, logger=True)
    

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch, batch_idx)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

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
        
        #optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False,
                             relative_step=False)
        self.opt = optimizer
        len_data = len(self.train_dataloader())
        denomniator = self.hparams.n_gpu
        steps_per_epoch = len_data // denomniator
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=steps_per_epoch, pct_start=0.1, epochs=self.hparams.num_train_epochs, anneal_strategy='linear', cycle_momentum=False)

        if self.hparams.use_lr_scheduling:
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "name": "learning rate"}]
        else:
            return [optimizer]

    def train_dataloader(self):   
        n_samples = self.n_obs['train']
        train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="train", num_samples=n_samples, args=self.hparams)
        sampler=RandomSampler(train_dataset)
        dataloader = DataLoader(train_dataset, sampler=sampler,  batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        '''
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=int(0.1 * t_total) , num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        '''
        return dataloader

    def val_dataloader(self):
        n_samples = self.n_obs['validation']
        
        validation_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="validation", num_samples=n_samples, args=self.hparams)
        #sampler=RandomSampler(validation_dataset)
        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)
    
    
    def test_dataloader(self):
        n_samples = self.n_obs['test']
        test_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="test", num_samples=n_samples, args=self.hparams)
        
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)