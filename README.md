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
