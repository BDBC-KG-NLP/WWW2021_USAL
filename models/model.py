import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from transformers import BertTokenizer, BertModel
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_path)
        # fine-tuning the bert parameters
        unfreeze_layers = ['pooler'] + ['layer.'+str(11 - i) for i in range(args.layers)]
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        self.input_dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(768,1)
        # nn.init.xavier_normal_(self.classifier.weight)
        nn.init.xavier_uniform_(self.classifier.weight,gain=args.alpha)

    def forward(self, inputs):
        # unpack inputs
        tokens, token_ids,masks = inputs

        # ipnut embs
        _ , text_cls = self.bert(tokens, token_type_ids=token_ids, attention_mask=masks)
        text_cls = self.input_dropout(text_cls)
        # logits
        logits = self.classifier(text_cls).squeeze(-1)

        return logits
