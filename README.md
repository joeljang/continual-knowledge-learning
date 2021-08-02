# Continual Knowledge Learning

In order to set up environment, 
```
conda create -n ckl python=3.8
conda activate ckl
pip install -r requirements.txt
```

To download the data used for ALL of the experiments, 
```
python download_data.py
```

Log experiments with wandb: https://www.wandb.com/

To run the experiments,
```
python train.py --config configs/t5_baseline.json
```

For LG AI,
```
To check if experiment works & getting logged to wandb,
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
