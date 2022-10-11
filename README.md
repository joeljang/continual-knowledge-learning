# Towards Continual Knowledge Learning of Language Models

This is the official github repository for [Towards Continual Knowledge Learning of Language Models](https://arxiv.org/abs/2110.03215), accepted at ICLR 2022.

Use the following to cite our paper:
```
@inproceedings{jang2022towards,
  title={Towards Continual Knowledge Learning of Language Models},
  author={Jang, Joel and Ye, Seonghyeon and Yang, Sohee and Shin, Joongbo and Han, Janghoon and Kim, Gyeonghun and Choi, Stanley Jungkyu and Seo, Minjoon},
  booktitle={ICLR},
  year={2022}
}
```

In order to reproduce our results, take the following steps:
### 1. Create conda environment and install requirements
```
conda create -n ckl python=3.8 && conda activate ckl
pip install -r requirements.txt
```

Also, make sure to install the correct version of pytorch corresponding to the CUDA version and environment:
Refer to https://pytorch.org/
```
#For CUDA 10.x
pip3 install torch torchvision torchaudio
#For CUDA 11.x
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. Download the data used for the experiments.
To download only the CKL benchmark dataset:
```
wget https://continual.blob.core.windows.net/ckl/ckl_data.zip
```

To download ALL of the data used for the experiments (required to reproduce results):
```
wget https://continual.blob.core.windows.net/ckl/data.zip
```

To download the (continually pretrained) model checkpoints of the main experiment (required to reproduce results):
```
wget https://continual.blob.core.windows.net/ckl/modelcheckpoints_main.zip
```

For the other experimental settings such as multiple CKL phases, GPT-2, we do not separately provide the continually pretrained model checkpoints.

### 3. Reproducing Experimental Results
We provide all the configs in order to reproduce the zero-shot results of our paper. We only provide the model checkpoints for the main experimental setting (full_setting) which can be downloaded with the command above.

    configs
    ├── full_setting
    │   ├── evaluation
    │   |   ├── invariantLAMA
    │   |   |   ├── t5_baseline.json
    │   |   |   ├── t5_kadapters.json
    │   |   |   ├── ...
    │   |   ├── newLAMA
    │   |   ├── newLAMA_easy
    │   |   ├── updatedLAMA
    │   ├── training
    │   |   ├── t5_baseline.json
    │   |   ├── t5_kadapters.json
    │   |   ├── ...
    ├── GPT2
    │   ├── ...
    ├── kilt
    │   ├── ...
    ├── small_setting
    │   ├── ...
    ├── split
    │   ├── ...                    

#### Components in each configurations file
- input_length (int) : the input sequence length
- output_length (int) : the output sequence length
- num_train_epochs (int) : number of training epochs 
- output_dir (string) : the directory to save the model checkpoints
- dataset (string) : the dataset to perform zero-shot evaluation or continual pretraining
- dataset_version (string) : the version of the dataset ['full', 'small', 'debug']
- train_batch_size (int) : batch size used for training
- learning rate (float) : learning rate used for training
- model (string) : model name in huggingface models (https://huggingface.co/models)
- method (string) : method being used ['baseline', 'kadapter', 'lora', 'mixreview', 'modular_small', 'recadam']
- freeze_level (int) : how much of the model to freeze during traininig (0 for none, 1 for freezing only encoder, 2 for freezing all of the parameters)
- gradient_accumulation_steps (int) : gradient accumulation used to match the global training batch of each method
- ngpu (int) : number of gpus used for the run
- num_workers (int) : number of workers for the Dataloader
- resume_from_checkpoint (string) : null by default. directory to model checkpoint if resuming from checkpoint
- accelerator (string) : 'ddp' by default. the pytorch lightning accelerator to be used. 
- use_deepspeed (bool) : false by default. Currently not extensively tested.
- CUDA_VISIBLE_DEVICES (string) : gpu devices that are made available for this run (e.g. "0,1,2,3", "0")
- wandb_log (bool) : whether to log experiment through wandb
- wandb_project (string) : project name of wandb
- wandb_run_name (string) : the name of this training run
- mode (string) : 'pretrain' for all configs
- use_lr_scheduling (bool) : true if using learning rate scheduling
- check_validation (bool) : true for evaluation (no training)
- checkpoint_path (string) : path to the model checkpoint that is used for evaluation
- output_log (string) : directory to log evaluation results to
- split_num (int) : default is 1. more than 1 if there are multile CKL phases
- split (int) : which CKL phase it is

This is an example of getting the invariantLAMA zero-shot evaluation of continually pretrained t5_kadapters
```
python run.py --config configs/full_setting/evaluation/invariantLAMA/t5_kadapters.json
```

This is an example of performing continual pretraining on CC-RecentNews (main experiment) with t5_kadapters
```
python run.py --config configs/full_setting/training/t5_kadapters.json
```

## Reference
```
@article{jang2021towards,
  title={Towards Continual Knowledge Learning of Language Models},
  author={Jang, Joel and Ye, Seonghyeon and Yang, Sohee and Shin, Joongbo and Han, Janghoon and Kim, Gyeonghun and Choi, Stanley Jungkyu and Seo, Minjoon},
  journal={arXiv preprint arXiv:2110.03215},
  year={2021}
}
```
