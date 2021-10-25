import pytorch_lightning as pl
from models.Modular_T5 import T5ForConditionalGeneration as T5_Modular
from models.Modular_Small_T5 import T5ForConditionalGeneration as T5_Modular_Small
from models.Modular_Small_T52 import T5ForConditionalGeneration as T5_Modular_Small2
from models.Kadapter_T5 import T5ForConditionalGeneration as T5_Kadapter
from models.Kadapter_T52 import T5ForConditionalGeneration as T5_Kadapter2
from models.Lora_T5 import T5ForConditionalGeneration as T5_Lora
from models.Lora_T52 import T5ForConditionalGeneration as T5_Lora2
from models.RecAdam import RecAdam
import torch.nn.functional as F
from transformers import (
    Adafactor,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import torch
from Datasets import Pretrain
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader, ConcatDataset
from rouge import Rouge
from collections import Counter

import re
import string
import copy

class T5(pl.LightningModule):
    def __init__(self, hparams):
        super(T5, self).__init__()
        self.save_hyperparameters(hparams)

        self.mix_ratio = 4
        self.mix_decay = 0.7
        self.epoch = 0

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
 
        self.output_dir = self.hparams.output_dir
            
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "validation": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

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

    def calculate_scores(self, predictions, ground_truths):
        em_score = 0
        accuracy = 0
        
        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            em_score +=  self.exact_match_score(prediction, ground_truth)
            accuracy += self.accuracy_match_score(prediction, ground_truth)
        
        em_score /= len(predictions)
        accuracy /= len(predictions)
        return em_score*100, accuracy*100
    
    def calculate_scores_multipleanswers(self, predictions, ground_truths, ids):
        em_score = 0
        accuracy_score = 0
        
        for i in range(len(predictions)):
            unique_id = ids[i]
            answers = self.ids_to_answers[unique_id]
            prediction = predictions[i]
            em_correct = False
            accuracy_correct = False
            for answer in answers:
                accuracy = self.accuracy_match_score(prediction, answer)
                if accuracy == 1:
                    accuracy_correct = True
                em  = self.exact_match_score(prediction, answer)
                if em == 1:
                    em_correct = True
            if accuracy_correct:
                accuracy_score+=1
            if em_correct:
                em_score+=1
        
        accuracy_score /= len(predictions)
        em_score /= len(predictions)
        return em_score*100, accuracy_score*100

    def calculate_rouge_multipleanswers(self, predictions, ground_truths, ids):
        rouge_score = 0 
        for i in range(len(predictions)):
            unique_id = ids[i]
            answers = self.ids_to_answers[unique_id]
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
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            f1_score += self._f1_score(prediction, ground_truth)

        f1_score /= len(predictions)
        return f1_score*100

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
        source = self.ids_to_clean_text(batch["source_ids"])
        print("preds", preds)
        print("targets", targets)
            
        loss = self._step(batch)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        em_score = 0
        accuracy = 0
        rouge_score = 0
        f1_score = 0

        if self.hparams.dataset == 'TriviaQA' or self.hparams.dataset == 'zsRE' or self.hparams.dataset == 'TREX' or self.hparams.dataset == 'NQ' or self.hparams.dataset == 'HotpotQA':
            em_score, accuracy = self.calculate_scores_multipleanswers(preds, targets, ids)
        elif self.hparams.dataset =='ELI5':
            rouge_score = self.calculate_rouge_multipleanswers(preds, targets, ids)
        elif self.hparams.dataset =='WOW':
            f1_score = self.calculate_f1_scores(preds, targets, ids)
        else:
            em_score, accuracy = self.calculate_scores(preds, targets)

        em_score = torch.tensor(em_score,dtype=torch.float32)
        accuracy = torch.tensor(accuracy,dtype=torch.float32)
        rouge_score = torch.tensor(rouge_score, dtype=torch.float32)
        f1_score = torch.tensor(f1_score, dtype=torch.float32)
        if self.hparams.dataset == 'ELI5':
            self.log('rouge_score', rouge_score, prog_bar=True, logger=True)
        elif self.hparams.dataset == 'WOW':
            self.log('f1_score', f1_score, prog_bar=True, logger=True)
        elif self.hparams.dataset == 'fever' or self.hparams.dataset == 'AY2' or self.args.dataset== 'TREX' or self.args.dataset== 'zsRE':
            self.log('accuracy', accuracy, prog_bar=True, logger=True)
        else:
            self.log('em_score', em_score, prog_bar=True, logger=True)


    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss)
        return loss

    def on_train_epoch_start(self):
        if self.hparams.method=='mixreview':
            train_set = self.train_dataloader().dataset
        self.epoch+=1
    
    def on_train_end(self):
        if self.hparams.mode == 'pretrain':
            if self.hparams.method=='recadam':
                self.pretrained_model = self.model
            elif self.hparams.method=='kadapter' or self.hparams.method=='lora' or self.hparams.method=='modular_small':
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
            mix_len = int(len(train_dataset) * self.mix_ratio * (self.mix_decay ** self.epoch))
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

    def val_dataloader(self):
        if self.hparams.mode == 'pretrain':
            return None
        else: 
            n_samples = self.n_obs['validation']
            validation_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="validation", num_samples=n_samples, args=self.hparams,)
            return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)
    
    def test_dataloader(self):
        n_samples = self.n_obs['test']
        test_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="test", num_samples=n_samples, args=self.hparams)
        
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)