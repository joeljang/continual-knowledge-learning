import argparse
from argparse import ArgumentParser
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import textwrap
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import (
    T5Tokenizer,GPT2Tokenizer
)
from torch.utils.data import DataLoader
from models import load_model
import time

from Datasets import Pretrain
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")

    #Getting configurations
    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    #Setting GPUs to use
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=hparam.CUDA_VISIBLE_DEVICES

    #Logging into WANDB if needed
    if hparam.wandb_log:
        wandb_logger = WandbLogger(project=hparam.wandb_project, name=hparam.wandb_run_name, entity="lklab_kaist")
    else:
        wandb_logger = None

    #Init configs that are not given
    if 'finetuning_ratio' not in hparam:
        hparam.finetuning_ratio = 0.0
    if 'prune_ratio' not in hparam:
        hparam.prune_ratio = 0.0
    if 'split_num' not in hparam:
        hparam.split_num = 1
    if 'split' not in hparam:
        hparam.split = 0

    #If using pruning method, no grad_norm
    if hparam.method=='prune':
        grad_norm = None
    else:
        grad_norm = 0.5
        
    #Setting configurations
    args_dict = dict(
        output_dir=hparam.output_dir, # Path to save the checkpoints
        dataset=hparam.dataset,
        dataset_version = hparam.dataset_version,
        split_num = hparam.split_num,
        split = hparam.split,
        finetuning_ratio = hparam.finetuning_ratio,
        prune_ratio = hparam.prune_ratio,
        model_name_or_path=hparam.model,
        method=hparam.method,
        freeze_level=hparam.freeze_level,
        mode=hparam.mode,
        tokenizer_name_or_path=hparam.model,
        max_input_length=hparam.input_length,
        max_output_length=hparam.output_length,
        freeze_encoder=False,
        freeze_embeds=False,
        learning_rate=hparam.learning_rate,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=hparam.train_batch_size,
        num_train_epochs=hparam.num_train_epochs,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        n_gpu=hparam.ngpu,
        num_workers=hparam.num_workers,
        resume_from_checkpoint=hparam.resume_from_checkpoint, 
        use_lr_scheduling = hparam.use_lr_scheduling,
        val_check_interval = 1.0,
        n_val=-1,
        n_train=-1,
        n_test=-1,
        early_stop_callback=False,
        use_deepspeed=hparam.use_deepspeed,
        opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=grad_norm, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
        check_validation_only=hparam.check_validation,
        checkpoint_path=hparam.checkpoint_path,
        accelerator=hparam.accelerator
    )
    args = argparse.Namespace(**args_dict)

    # Defining how to save model checkpoints during training. Details: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.model_checkpoint.html 
    if args.dataset_version=='full':
        callbacks = [ModelCheckpoint(dirpath = args.output_dir, save_last=True, every_n_val_epochs=1)]
    else:
        if args.split_num==2:
            #callbacks = [ModelCheckpoint(dirpath = args.output_dir, save_last=True, every_n_val_epochs=args.num_train_epochs // 2)]
            callbacks = [ModelCheckpoint(dirpath = args.output_dir, save_last=True)]
        else:
            callbacks = [ModelCheckpoint(dirpath = args.output_dir, save_last=True)]
    checkpoint_callback = True

    if args.output_dir=="":
        checkpoint_callback = False # Do not save model checkpoints when output dir is empty
        callbacks=[]

    # Logging Learning Rate Scheduling
    if args.use_lr_scheduling and hparam.wandb_log:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    if args.use_deepspeed:
        plugins = 'deepspeed_stage_2'
        use_fp_16 = True
    else:
        plugins = []
        use_fp_16 = False

    # Setting Flags for pytorch lightning trainer. Details: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags
    if args.split_num==2:
        #num_train_epochs = args.num_train_epochs // 2
        num_train_epochs=1
    else:
        num_train_epochs = args.num_train_epochs
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        plugins=plugins,
        gpus=args.n_gpu,
        max_epochs=num_train_epochs,
        precision= 16 if use_fp_16 else 32,
        amp_level=args.opt_level,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=args.val_check_interval,
        logger=wandb_logger,
        callbacks = callbacks,
        accelerator=args.accelerator,
    )

    #Getting the Model type & Method
    if 't5' in args.model_name_or_path:
        model_type='T5'
    elif 'gpt2' in args.model_name_or_path:
        model_type='GPT2'
    else:
        raise Exception('Select the correct model. Supporting "t5" and "gpt2" only.')
    T5Model = load_model(type=model_type)
    
    if args.check_validation_only:
        model = T5Model(args)
        if args.checkpoint_path!="":
            model = T5Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args)

        model.eval()
        model.to('cuda')
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        #Get Validation Data
        if args.mode=='pretrain' or args.mode=='finetune':
            dataset = Pretrain(tokenizer, 'validation', None, input_length=args.max_input_length, 
                            output_length=args.max_output_length, args=args)
        else:
            raise Exception('Select the correct mode please.')
        print('Length of validation data: ',len(dataset))
        loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)
        
        total_cnt = 0
        rp_cnt = 0
        accuracy_correct_num = 0
        em_correct_num = 0
        subset_correct_num = 0
        rp_em_correct_num = 0
        rp_subset_correct_num = 0

        def clean_up(text):
            text =text.replace('<pad>', '')
            text = text.replace('</s>', '')
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            text = text.replace("<extra_id_2>", "")
            text = text.replace("<extra_id_3>", "")
            text = text.replace(".", '')
            text = text.replace(',', '')
            text = text.replace("'", '')
            text = text.replace('"', '')
            return text     
        for batch in iter(loader):
            outs = model.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=True,
                decoder_attention_mask=batch['target_mask'].cuda(),
                max_length=args.max_output_length,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=1
            )
            dec = [tokenizer.decode(ids) for ids in outs]
            texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
            targets = [tokenizer.decode(ids) for ids in batch['target_ids']]
  
            for i in range(len(batch['source_ids'])):
                total_cnt+=1
                lines = textwrap.wrap("\n%s\n" % texts[i], width=200)
                ground_truth = clean_up(targets[i])
                predicted = clean_up(dec[i])
                em = model.exact_match_score(predicted, ground_truth)
                subset = model.approx_match_score(predicted, ground_truth)         
                print(f'{total_cnt} INPUT : {lines}')
                print(f'GROUD TRUTH: {ground_truth}, MODEL OUTPUT: {predicted}')
                if args.dataset == 'recentnews':
                    if total_cnt < 20725:
                        if em == 1:
                            em_correct_num+=1
                        if subset == 1:
                            subset_correct_num+=1
                    else:
                        rp_cnt+=1
                        if em == 1:
                            rp_em_correct_num+=1
                        if subset == 1:
                            rp_subset_correct_num+=1
                else:
                    # zero-shot accuracy for WnED and CWEB
                    accuracy = model.accuracy_match_score(predicted, ground_truth)
                    if accuracy == 1:
                        accuracy_correct_num +=1
                    if em == 1:
                        em_correct_num+=1
                    if subset == 1:
                        subset_correct_num+=1  
        if args.dataset == 'recentnews':
            print(f'Number of total validation data: {total_cnt}')
            print(f'Number of correct lama predictions out of 20725 : {em_correct_num, subset_correct_num}. Percentage : {em_correct_num / 20725, subset_correct_num / 20725}')
            print(f'Number of correct recentprobe predictions out of {rp_cnt} : {rp_em_correct_num, rp_subset_correct_num}. Percentage : {rp_em_correct_num / rp_cnt, rp_subset_correct_num / rp_cnt}')
        else:
            print(f'Number of total validation data: {total_cnt}')
            print(f'Number of correct predictions: {accuracy_correct_num, em_correct_num, subset_correct_num}. Percentage : {accuracy_correct_num / total_cnt, em_correct_num / total_cnt, subset_correct_num / total_cnt}')
    else:
        set_seed(40)
        if args.checkpoint_path!="":
            model = T5Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args) 
        else:
            model = T5Model(args)
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)