python train.py --config configs/kilt/t5_recadam/t5_fever.json 


python train.py --config configs/kilt/t5_recadam/t5_hotpotqa.json
python train.py --config configs/kilt/t5_recadam/t5_wow.json
python train.py --config configs/kilt/t5_recadam/t5_zsre.json
python train.py --config configs/kilt/t5_recadam/t5_nq.json
python train.py --config configs/kilt/t5_recadam/t5_tqa.json

python train.py --config configs/kilt/t5_recadam/t5_ay2.json
python -u train.py --config configs/kilt/t5_recadam/t5_wned.json 2>&1 | tee 't5_recadam_t5_wned.txt'
python -u train.py --config configs/kilt/t5_recadam/t5_cweb.json 2>&1 | tee 't5_recadam_t5_cweb.txt'

python train.py --config configs/kilt/t5_recadam/t5_trex.json
python train.py --config configs/kilt/t5_recadam/t5_eli5.json