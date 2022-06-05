# USAL
This is a pytorch implementation of the WWW 2021 paper "Unsupervised Semantic Association Learning with Latent Label Inference". 

# Requirements
python3.6.0+

pytorch 1.7.0+

transformers

tensorboard

# Usage
Download the data, and change the datapath in the configs file.

To train a new  model, run the following command:
```
python main.py --dataset='config name' --main=wsd.py/qa.py --do_train
```

## Citation

If this work is helpful, please cite as:

```bibtex
@inproceedings{zhang2021unsupervised,
  title={Unsupervised Semantic Association Learning with Latent Label Inference},
  author={Zhang, Yanzhao and Zhang, Richong and Kim, Jaein and Liu, Xudong and Mao, Yongyi},
  booktitle={Proceedings of the Web Conference 2021},
  pages={4010--4019},
  year={2021}
}
```


## License

MIT

