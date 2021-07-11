# Continual Knowledge Learning

To download the data, use azcopy:
```
azcopy cp https://continual.blob.core.windows.net/recentnews/data ./ --recursive
```
Azcopy Docs : docs: https://docs.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy-copy?toc=/azure/storage/blobs/toc.json

Log experiments with wandb: https://www.wandb.com/

To run the experiments, run the following command:
```
python train.py --config configs/recentnews.json
```
