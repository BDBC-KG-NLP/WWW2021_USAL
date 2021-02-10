import argparse
import importlib
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='wsd')
    parser.add_argument('--main',type=str,default='wsd')
    args,unparse = parser.parse_known_args()
    config = importlib.import_module('configs.'+args.dataset)
    main = importlib.import_module(args.main).main
    main(config.parse_args(unparse))
