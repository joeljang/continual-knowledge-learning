# Continual Knowledge Learning

To download the data, use azcopy:
```
#if azcopy version == 10.x
azcopy cp https://continual.blob.core.windows.net/recentnews/data ./ --recursive 
#if acopy == 7.x
azcopy --source https://continual.blob.core.windows.net/recentnews/data --destination ./data --recursive 
```
Azcopy Docs : docs: https://docs.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy-copy?toc=/azure/storage/blobs/toc.json

Log experiments with wandb: https://www.wandb.com/

To run the experiments, run the following command:
```
python train.py --config configs/recentnews.json
```
