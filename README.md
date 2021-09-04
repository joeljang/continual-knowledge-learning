# Continual Knowledge Learning

In order to set up environment, 
```
conda create -n ckl python=3.8
conda activate ckl
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

To download the data used for ALL of the experiments, 
```
python download_data.py

#if using azcopy
azcopy cp https://continual.blob.core.windows.net/recentnews/data ./ --recursive
azcopy cp https://continual.blob.core.windows.net/recentnews/model_checkpoints ./ --recursive
```

Azcopy docs : https://docs.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy-copy?toc=/azure/storage/blobs/toc.json

Log experiments with wandb: https://www.wandb.com/

To run the experiments,
```
python train.py --config configs/t5_baseline.json
```

For LG AI,
```
#To check if experiment works & getting logged to wandb,
python train.py --config configs/lgai/test.json #Using GPU num 0

VM instance #1 (~35 hours)
python train.py --config configs/lgai/1_baseline.json #Using GPU num 0,1,2,3
python train.py --config configs/lgai/1_kadapters.json #Using GPU num 4,5,6,7

VM instance #1 (~30 hours)
python train.py --config configs/lgai/2_lora.json #Using GPU num 0,1,2,3
python train.py --config configs/lgai/2_m_small.json #Using GPU num 4,5,6,7

VM instance #1 (~40 hours)
python train.py --config configs/lgai/3_m_large.json #Using GPU num 0,1,2,3
python train.py --config configs/lgai/3_recadam.json #Using GPU num 4,5,6,7

VM instance #2 (~100 hours)
python train.py --config configs/lgai/4_mixreview.json #Using GPU num 0,1,2,3,4,5,6,7
```
