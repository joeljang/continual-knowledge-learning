# Towards Continual Knowledge Learning of Language Models

This is the official github repository for [Towards Continual Knowledge Learning of Language Models](https://arxiv.org/abs/2110.03215).

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
To download only the data used for continued pretraining and zero shot evaluation:
```
python download_data.py
```

To download ALL of the data used for the experiments:
```
python download_data.py
```

### 3. Perform zero-shot evaluations of continually pretrained language models


### 4. Perform continual pretraining 

## Reference
```
@article{jang2021towards,
  title={Towards Continual Knowledge Learning of Language Models},
  author={Jang, Joel and Ye, Seonghyeon and Yang, Sohee and Shin, Joongbo and Han, Janghoon and Kim, Gyeonghun and Choi, Stanley Jungkyu and Seo, Minjoon},
  journal={arXiv preprint arXiv:2110.03215},
  year={2021}
}
```