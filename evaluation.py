
from transformers import T5Tokenizer
from Datasets import Pretrain
from torch.utils.data import DataLoader
import csv
import os

def evaluate(args, Model):
    model = Model(args)
    if args.checkpoint_path!="":
        model = Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False)

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
    em_correct_num = 0
    old_em_correct_num = 0
    new_em_correct_num = 0
    accuracy_correct_num = 0
    def clean_up(text):
        text =text.replace('<pad>', '')
        text = text.replace('</s>', '')
        text = text.replace(".", '')
        text = text.replace(',', '')
        text = text.replace("'", '')
        text = text.replace('"', '')
        return text   
    # If folder doesn't exist, then create it.
    MYDIR = ("/".join((args.output_log.split('/'))[:-1]))
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print(MYDIR, "folder already exists.")

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
            dec = model.ids_to_clean_text(outs)
            texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
            targets = model.ids_to_clean_text(batch['target_ids'])
                
            for i in range(len(batch['source_ids'])):
                total_cnt+=1
                lines = clean_up(texts[i])
                ground_truth = targets[i]
                predicted = dec[i]
                # print("prediction:",total_cnt,predicted)
                ids = batch['label_ids'][i].item()

                if args.dataset == 'invariantlama':
                    em = model.exact_match_score(predicted, ground_truth)  
                    writer.writerow([ids, lines, ground_truth, predicted])
                    if em == 1:
                        em_correct_num+=1
                elif args.dataset == 'updatedlama':
                    old_answer_list = ids_to_answers[str(ids)][0]['old']
                    new_answer_list = ids_to_answers[str(ids)][0]['new']
                    old_em_correct = False
                    new_em_correct = False
                    old_global_answer = old_answer_list[0]
                    new_global_answer = new_answer_list[0]
                    for answer in old_answer_list:
                        em = model.exact_match_score(predicted, answer)
                        if em == 1:
                            old_em_correct = True
                            old_global_answer = answer
                    if old_em_correct:
                        old_em_correct_num+=1

                    for answer in new_answer_list:
                        em = model.exact_match_score(predicted, answer)
                        if em == 1:
                            new_em_correct = True
                            new_global_answer = answer
                    if new_em_correct:
                        new_em_correct_num+=1

                    writer.writerow([ids, lines, old_global_answer, new_global_answer, predicted])
                        
                elif args.dataset == 'newlama' or args.dataset == 'newlama_easy':
                    answer_list = ids_to_answers[str(ids)]
                    em_correct = False
                    global_answer = answer_list[0]
                    for answer in answer_list:
                        em = model.exact_match_score(predicted, answer)
                        if em == 1:
                            em_correct = True
                            global_answer = answer
                    if em_correct:
                        em_correct_num+=1
                    writer.writerow([ids, lines, global_answer, predicted])
                elif args.dataset == 'WNED' or args.dataset == 'CWEB':
                    accuracy = model.accuracy_match_score(predicted, ground_truth)
                    if accuracy == 1:
                        accuracy_correct_num +=1
                    writer.writerow([lines, ground_truth, predicted])
                else:
                    raise NameError('Select the correct Dataset for zeroshot evaluation!')
    print(f'Number of total validation data: {total_cnt}')
    if args.dataset == 'updatedlama':
        with open(args.output_log, 'a', newline='') as writefile:  
            writer = csv.writer(writefile)
            writer.writerow([old_em_correct_num, old_em_correct_num / total_cnt])
            writer.writerow([new_em_correct_num, new_em_correct_num / total_cnt])
        print(f'Number of old correct predictions: {old_em_correct_num}. Percentage : {old_em_correct_num / total_cnt}')
        print(f'Number of new correct predictions: {new_em_correct_num}. Percentage : {new_em_correct_num / total_cnt}')
    elif args.dataset == 'WNED' or args.dataset == 'CWEB':
        with open(args.output_log, 'a', newline='') as writefile:  
            writer = csv.writer(writefile)
            writer.writerow([accuracy_correct_num, accuracy_correct_num / total_cnt])
        print(f'Number of correct predictions: {accuracy_correct_num}. Percentage : {accuracy_correct_num / total_cnt}')
    else:
        with open(args.output_log, 'a', newline='') as writefile:  
            writer = csv.writer(writefile)
            writer.writerow([em_correct_num, em_correct_num / total_cnt])
        print(f'Number of correct predictions: {em_correct_num}. Percentage : {em_correct_num / total_cnt}')