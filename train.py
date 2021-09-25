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
import csv

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
    if 'prune' in hparam.method or 'layerwiselr' in hparam.method:
        grad_norm = None
    else:
        grad_norm = 0.5

    #If using pruning method, no grad_norm
    if 'weight_decay' not in hparam:
        hparam.weight_decay = 0.0

    if 'output_log' not in hparam:
        hparam.output_log = None
        
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
        weight_decay=hparam.weight_decay,
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
        accelerator=hparam.accelerator,
        output_log=hparam.output_log,
    )
    args = argparse.Namespace(**args_dict)

    # Defining how to save model checkpoints during training. Details: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.model_checkpoint.html 
    if args.dataset_version=='full':
        callbacks = [ModelCheckpoint(dirpath = args.output_dir, save_top_k=-1, period=1)]
    else:
        if args.split_num==2:
            #callbacks = [ModelCheckpoint(dirpath = args.output_dir, save_last=True, every_n_val_epochs=args.num_train_epochs // 2)]
            callbacks = [ModelCheckpoint(dirpath = args.output_dir, save_last=True)]
        else:
            callbacks = [ModelCheckpoint(dirpath = args.output_dir, save_top_k=-1, period=1)]
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
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        plugins=plugins,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
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
            model = T5Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False)

        model.eval()
        model.to('cuda')
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        #Get Validation Data
        if args.mode=='pretrain' or args.mode=='finetune':
            dataset = Pretrain(tokenizer, 'validation', None, input_length=args.max_input_length, 
                            output_length=args.max_output_length, args=args)
            ids_to_answers = dataset.ids_to_answers
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
        f1_correct_num = 0
        total_pred_len = 0
        total_target_len = 0
        f1_old_correct_num = 0
        total_old_pred_len = 0
        total_old_target_len = 0
        old_em_correct_num = 0
        old_subset_correct_num = 0
        f1_new_correct_num = 0
        total_new_pred_len = 0
        total_new_target_len = 0
        new_em_correct_num = 0
        new_subset_correct_num = 0

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
        
        with open(args.output_log, 'w', newline='') as writefile:  
            writer = csv.writer(writefile)
            for batch in iter(loader):
                outs = model.model.generate(
                    batch["source_ids"].cuda(),
                    attention_mask=batch["source_mask"].cuda(),
                    use_cache=True,
                    decoder_attention_mask=batch['target_mask'].cuda(),
                    max_length=args.max_output_length,
                    num_beams=2,
                    early_stopping=True,
                )
                # dec = [tokenizer.decode(ids) for ids in outs]
                # texts = [tokenizer.decode(ids,skip_special_tokens=True) for ids in batch['source_ids']]
                # targets = [tokenizer.decode(ids) for ids in batch['target_ids']]
                dec = model.ids_to_clean_text(outs)
                texts = model.ids_to_clean_text(batch['source_ids'])
                targets = model.ids_to_clean_text(batch['target_ids'])
                # if args.dataset == 'updatedqa' or args.dataset == 'updatedprobe'
                #     ids = batch["label_ids"]
                    
                for i in range(len(batch['source_ids'])):
                    total_cnt+=1
                    # lines = textwrap.wrap("\n%s\n" % texts[i], width=200)
                    # ground_truth = clean_up(targets[i])
                    # predicted = clean_up(dec[i])
                    lines = texts[i]
                    ground_truth = targets[i]
                    predicted = dec[i]
                    ids = batch['label_ids'][i].item()
                    if args.dataset == 'updatedqa' or args.dataset == 'updatedprobe':
                        old_answer_list = ids_to_answers[str(ids)][0]['old']
                        new_answer_list = ids_to_answers[str(ids)][0]['new']
                        old_em_correct = False
                        old_sm_correct = False
                        new_em_correct = False
                        new_sm_correct = False
                        old_global_answer = None
                        new_global_answer = None
                        max_f1 = 0
                        for answer in old_answer_list:
                            em = model.exact_match_score(predicted, answer)
                            if em == 1:
                                old_em_correct = True
                            sm = model.approx_match_score(predicted, answer)
                            if sm == 1:
                                old_sm_correct = True
                            f1 = model._f1_score(predicted, answer)
                            if f1 >= max_f1:
                                max_f1 = f1
                                old_global_answer = answer
                                old_num_same, old_pred_len, old_target_len = model._f1_score_zeroshot(predicted, answer)  

                        if old_em_correct:
                            old_em_correct_num+=1
                        if old_sm_correct:
                            old_subset_correct_num+=1
                        f1_old_correct_num+=old_num_same
                        total_old_pred_len+=old_pred_len
                        total_old_target_len+=old_target_len

                        for answer in new_answer_list:
                            em = model.exact_match_score(predicted, answer)
                            if em == 1:
                                new_em_correct = True
                            sm = model.approx_match_score(predicted, answer)
                            if sm == 1:
                                new_sm_correct = True
                            f1 = model._f1_score(predicted, answer)
                            if f1 >= max_f1:
                                max_f1 = f1
                                new_global_answer = answer
                                new_num_same, new_pred_len, new_target_len = model._f1_score_zeroshot(predicted, answer)  

                        if new_em_correct:
                            new_em_correct_num+=1
                        if new_sm_correct:
                            new_subset_correct_num+=1
                        f1_new_correct_num+=new_num_same
                        total_new_pred_len+=new_pred_len
                        total_new_target_len+=new_target_len
                        if model_type == 'T5':
                            writer.writerow([ids, lines, old_global_answer, new_global_answer, predicted])
                        else: 
                            writer.writerow([lines, old_global_answer, new_global_answer, predicted])
                            
                    else:
                    # lines = model.ids_to_clean_text(batch['source_ids'])
                    # ground_truth = model.ids_to_clean_text(batch['target_ids'])
                    # predicted = model.ids_to_clean_text(outs)
                        em = model.exact_match_score(predicted, ground_truth)
                        subset = model.approx_match_score(predicted, ground_truth)   
                        num_same, pred_len, target_len = model._f1_score_zeroshot(predicted, ground_truth)  
                        print(f'{total_cnt} INPUT : {lines}')
                        print(f'GROUD TRUTH: {ground_truth}, MODEL OUTPUT: {predicted}')
                        if model_type == 'T5':
                            writer.writerow([ids, lines, ground_truth, predicted])
                        else: 
                            writer.writerow([lines, ground_truth, predicted])
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
                        elif args.dataset == 'recentprobe':
                            if em == 1:
                                em_correct_num+=1
                            if subset == 1:
                                subset_correct_num+=1
                        else:
                            # zero-shot accuracy for WnED and CWEB
                            accuracy = model.accuracy_match_score(predicted, ground_truth)
                            if accuracy == 1:
                                accuracy_correct_num +=1
                            if em == 1:
                                em_correct_num+=1
                            if subset == 1:
                                subset_correct_num+=1  
                            f1_correct_num+=num_same
                            total_pred_len+=pred_len
                            total_target_len+=target_len
        if args.dataset == 'recentnews':
            print(f'Number of total validation data: {total_cnt}')
            print(f'Number of correct lama predictions out of 20725 : {em_correct_num, subset_correct_num}. Percentage : {em_correct_num / 20725, subset_correct_num / 20725}')
            print(f'Number of correct recentprobe predictions out of {rp_cnt} : {rp_em_correct_num, rp_subset_correct_num}. Percentage : {rp_em_correct_num / rp_cnt, rp_subset_correct_num / rp_cnt}')
        elif args.dataset == 'recentprobe':
            print(f'Number of total validation data: {total_cnt}')
            print(f'Number of correct predictions: {em_correct_num, subset_correct_num}. Percentage : {(em_correct_num / total_cnt)*100, (subset_correct_num / total_cnt)*100}')
        elif args.dataset == 'updatedqa' or args.dataset == 'updatedprobe':
            print(f'Number of total validation data: {total_cnt}')
            old_precision = 1.0 * f1_old_correct_num / total_old_pred_len
            old_recall = 1.0 * f1_old_correct_num / total_old_target_len
            old_f1_score = (2 * old_precision * old_recall) / (old_precision + old_recall)
            new_precision = 1.0 * f1_new_correct_num / total_new_pred_len
            new_recall = 1.0 * f1_new_correct_num / total_new_target_len
            new_f1_score = (2 * new_precision * new_recall) / (new_precision + new_recall)
            print(f'Number of old correct predictions: {old_em_correct_num, old_subset_correct_num}. Percentage : {old_f1_score, old_em_correct_num / total_cnt, old_subset_correct_num / total_cnt}')
            print(f'Number of new correct predictions: {new_em_correct_num, new_subset_correct_num}. Percentage : {new_f1_score, new_em_correct_num / total_cnt, new_subset_correct_num / total_cnt}')

        else:
            print(f'Number of total validation data: {total_cnt}')
            precision = 1.0 * f1_correct_num / total_pred_len
            recall = 1.0 * f1_correct_num / total_target_len
            f1_score = (2 * precision * recall) / (precision + recall)
            print(f'Number of correct predictions: {em_correct_num, subset_correct_num}. Percentage : {f1_score, em_correct_num / total_cnt, subset_correct_num / total_cnt}')
    else:
        set_seed(40)
        if args.checkpoint_path!="":
            model = T5Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False) 
        else:
            model = T5Model(args)
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)
