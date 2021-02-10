# USAL
This is a pytorch implementation of the WWW 2021 paper "Unsupervised Semantic Association Learning with Latent Label Inference". 

# Requirements
python3.6.0+
pytorch 1.7.0+
transformers
tensorboard

# Getting start
download the data, and change the datapath in the configs file.
To train a new  model, run the following command:
python main.py --dataset='config name' --main=wsd.py/qa.py --do_train


