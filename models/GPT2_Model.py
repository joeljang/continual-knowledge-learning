import pytorch_lightning as pl
#from models.Modular_GPT2 import GPT2LMHeadModel as GPT2_Modular
#from models.Kadapter_GPT2 import GPT2LMHeadModel as GPT2_Kadapter
#from models.Lora_GPT2 import GPT2LMHeadModel as GPT2_Lora
#from models.RecAdam import RecAdam

from transformers import (
    Adafactor,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

import torch
from Datasets import Pretrain
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader, ConcatDataset

import re
import string

class GPT2(pl.LightningModule):
    def __init__(self, hparams):
        super(GPT2, self).__init__()
        self.save_hyperparameters(hparams)      

        self.mix_ratio = 4
        self.mix_decay = 0.7
        self.epoch = 0
        '''
        if hparams.method=='kadapter':
            self.model = GPT2_Kadapter.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='lora':
            self.model = GPT2_Lora.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='modular':
            self.model = GPT2_Modular.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='recadam':
            self.model = GPT2LMHeadModel.from_pretrained(hparams.model_name_or_path)
            self.pretrained_model = GPT2LMHeadModel.from_pretrained(hparams.model_name_or_path)
            self.freeze_params(self.pretrained_model) #Freezing pretrained model
        else:
        '''
        self.model = GPT2LMHeadModel.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(hparams.model_name_or_path)
        self.tokenizer.add_special_tokens({
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>"
            })

        if hparams.freeze_level==0: # Do not freeze any parameters
            print('Not freezing any parameters!')
        elif hparams.freeze_level==1: # Freeze model
            self.freeze_params(self.model) 

        if hparams.method=='kadapter':
            # Unfreezing the parameters used for kadapter
            for name, param in self.model.named_parameters():
                if 'kadapter' in name or 'lm_head' in name:
                    param.requires_grad = True
        elif hparams.method=='lora':
            # Unfreezing the parameters used for lora
            for name, param in self.model.named_parameters():
                if 'lora' in name or 'lm_head' in name:
                    param.requires_grad = True

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.padding_side = "left"

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

    def calculate_scores(self, predictions, ground_truths):
        em_score = 0
        
        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            em_score +=  self.exact_match_score(prediction, ground_truth)
        
        em_score /= len(predictions)
        return em_score*100


    def calculate_scores_multipleanswers(self, predictions, ground_truths, ids):
        em_score = 0
        
        for i in range(len(predictions)):
            unique_id = str(ids[i].item())
            answers = self.ids_to_answers[unique_id]
            prediction = predictions[i]
            em_correct = False
            for answer in answers:
                em = self.exact_match_score(prediction, answer)
                if em == 1:
                    em_correct = True
            if em_correct:
                em_score+=1

        em_score /= len(predictions)
        return em_score*100

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
            labels=lm_labels,
    )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
        )

        loss = outputs[0]
        return loss
    
    
    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)
    
     
    def _generative_step(self, batch, batch_idx):
        
        input_length = self.hparams.max_input_length
        if self.hparams.dataset=='invariantlama':
            max_length = 53
        elif self.hparams.dataset=='newqa_easy':
            max_length = 110
            
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            max_length=max_length,
            num_beams=2,
            early_stopping=True
        )

        generated_ids = torch.transpose(torch.transpose(generated_ids,0,1)[input_length:],0,1)
        preds = self.ids_to_clean_text(generated_ids)
        clean_preds = []
        for text in preds:
            if "." in text:
                clean_preds.append(text[:text.find(".")+1])
            else: 
                clean_preds.append(text)
        source = self.ids_to_clean_text(batch["source_ids"])
        print("clean_preds",clean_preds)
        targets = self.ids_to_clean_text(batch["ground_truth_ids"])
        ids = batch["label_ids"]
        print("targets",targets)
    
        loss = self._step(batch)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.hparams.dataset=='invariantlama':
            em_score = self.calculate_scores(clean_preds, targets)
        elif self.hparams.dataset=='newqa_easy':
            em_score = self.calculate_scores_multipleanswers(clean_preds, targets, ids)
        
        em_score = torch.tensor(em_score,dtype=torch.float32)
        if self.hparams.dataset=='invariantlama':
            self.log('invariant_em_score', em_score, prog_bar=True, logger=True)
        elif self.hparams.dataset=='newqa_easy':
            self.log('new_em_score', em_score, prog_bar=True, logger=True)
    
    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss)
        return loss

    
    def on_train_epoch_start(self):
        if self.hparams.method=='mixreview':
            train_set = self.train_dataloader().dataset
        self.epoch+=1

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch, batch_idx)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        if self.hparams.method=='recadam':
            no_decay = ["bias", "LayerNorm.weight"]
            model_type = 'gpt2'
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

        self.optimizer = optimizer
        len_data = len(self.train_dataloader())
        denomniator = self.hparams.n_gpu * self.hparams.gradient_accumulation_steps
        steps_per_epoch = ( len_data // denomniator ) + 1
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=steps_per_epoch, pct_start=0.1, epochs=self.hparams.num_train_epochs, anneal_strategy='linear', cycle_momentum=False)

        if self.hparams.use_lr_scheduling:
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "name": "learning rate"}]
        else:
            return [optimizer]

    def train_dataloader(self):   
        n_samples = self.n_obs['train']
        train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="train", num_samples=n_samples, args=self.hparams)
        if self.hparams.method=='mixreview':
            mix_len = int(len(train_dataset) * self.mix_ratio * (self.mix_decay ** self.epoch))
            pretrain_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="pretrain", num_samples=n_samples, args=self.hparams, length=mix_len)
            mixed_dataset = ConcatDataset([train_dataset,pretrain_dataset])
            print("mix len is ", mix_len)
            sampler=RandomSampler(mixed_dataset)
            dataloader = DataLoader(mixed_dataset, sampler = sampler, batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
            print("dataset length is ", len(dataloader.dataset))
        else:
            sampler=RandomSampler(train_dataset)
            dataloader = DataLoader(train_dataset, sampler=sampler,  batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        if self.hparams.mode == 'pretrain':
            return None
        else: 
            n_samples = self.n_obs['validation']
            validation_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="validation", num_samples=n_samples, args=self.hparams)
            return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)
    
    
    def test_dataloader(self):
        n_samples = self.n_obs['test']
        test_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="test", num_samples=n_samples, args=self.hparams)
        
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)