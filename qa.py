import os
import pickle
import random
import argparse
import torch
import numpy as np
import json
import math
import uuid
import datetime
import subprocess
import sys
import time
import importlib
from collections import defaultdict

from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW
from torch.utils.tensorboard import SummaryWriter
import logging
import pytorch_warmup as warmup


'''
em4基础上 E步改为softmax
'''


def write_result(resultname, results):
    with open(resultname, 'w') as fout:
        for ele in results:
            fout.write(ele[0] + '\t' + str(ele[1]) + '\t' + str(ele[2]) + '\n')


def generate_bertinput(data, cand):
    doc = data['query']
    cand_doc = cand['subject'] + ' ' + cand['body']
    return doc, cand_doc


class Mydataset(object):
    def __init__(self, datas, tokenizer, neg_candidates, priors, args):
        self.datas = datas
        self.batch_size = args.mb
        self.select = list(range(len(datas)))
        self.index = 0
        random.shuffle(self.select)
        self.index_generater = self.get_batch()
        self.tokenizer = tokenizer
        self.neg_counts = {}
        self.neg_candidates = neg_candidates
        self.priors = {}
        for ele in datas:
            uid = ele['id']
            for cand, v in zip(ele['candidates'], priors[uid]):
                self.priors[cand['cid']] = v
    def __len__(self):
        return len(self.datas)

    def get_batch(self):
        while True:
            if self.index >= len(self.datas):
                self.index = 0
                random.shuffle(self.select)
            start = self.index
            end = start + self.batch_size
            self.index = end
            yield self.select[start:end]

    def generate_batch(self, preds, args):
        select = next(self.index_generater)
        docs = []
        labels = []
        pos_num = 0
        select_datas = []
        bias = []
        priors = []
        for index in select:
            data = self.datas[index]
            uid = data['id']
            for i, cand in enumerate(data['candidates']):
                doc = generate_bertinput(data, cand)
                docs.append(doc)
                labels.append(preds[uid][i])
                Pw = 1. / (len(data['candidates']))
                # if args.neg_sample == 'querys':
                #     Pw = 1. / (len(data['candidates']))
                # else:
                #     Pw = 1. / (len(self.neg_candidates))
                bias.append(math.log(args.sample_num * Pw))
                priors.append(self.priors[cand['cid']])
                pos_num += 1
        # neg sample
        for index in select:
            data = self.datas[index]
            if args.neg_sample == 'mix':
                neg_cands = list(np.random.choice(data['candidates'], int(args.sample_num * args.neg_ratio))) + list(
                    np.random.choice(
                        self.neg_candidates, int(args.sample_num * (1-args.neg_ratio))))
                Pw = 1. / len(self.neg_candidates)
            if args.neg_sample == 'querys':
                neg_candidates = data['candidates']
                neg_cands = np.random.choice(neg_candidates, args.sample_num)
                Pw = 1 / len(neg_candidates)
            elif args.neg_sample == 'keys':
                neg_candidates = self.neg_candidates
                neg_cands = np.random.choice(neg_candidates, args.sample_num)
                Pw = 1 / len(neg_candidates)
            for cand in neg_cands:
                self.neg_counts.setdefault(uid, {})
                self.neg_counts[uid].setdefault(cand['cid'], 0)
                self.neg_counts[uid][cand['cid']] += 1
                # sense = np.random.choice(neg_senses, 1)[0]
                # gloss = " ".join(Gloss[sense])
                # doc = '[CLS] ' + " ".join(data['context']) + '[SEP] ' + gloss
                doc = generate_bertinput(data, cand)
                docs.append(doc)
                # Pw = 1. / len(data['RelQuestions'])
                bias.append(math.log(args.sample_num * Pw))
                priors.append(self.priors[cand['cid']])
                labels.append(0)
        output = self.tokenizer(docs, padding=True, max_length=500)
        input_ids = torch.LongTensor(output['input_ids'])
        token_type_ids = torch.LongTensor(output['token_type_ids'])
        attention_mask = torch.LongTensor(output['attention_mask'])
        priors = torch.Tensor(priors)
        labels = torch.Tensor(labels)
        bias = torch.Tensor(bias)
        return input_ids, token_type_ids, attention_mask, labels, bias, pos_num, select_datas, priors


def test_iterator(model, data, tokenizer, prior_preds, args):
    MAP = 0
    uid = data['id']
    docs = []
    labels = []
    for i, cand in enumerate(data['candidates']):
        doc = generate_bertinput(data, cand)
        docs.append(doc)
        if int(cand['label']) == 0:
            labels.append((cand['cid'], 0))
        else:
            labels.append((cand['cid'], 1))
    output = tokenizer(docs, padding=True, max_length=500)
    input_ids = torch.LongTensor(output['input_ids'])
    token_type_ids = torch.LongTensor(output['token_type_ids'])
    attention_mask = torch.LongTensor(output['attention_mask'])
    pred = prior_preds[uid]
    pred = torch.Tensor(pred)
    if args.cuda:
        input_ids = input_ids.cuda()
        token_type_ids = token_type_ids.cuda()
        attention_mask = attention_mask.cuda()
        pred = pred.cuda()
    with torch.no_grad():
        scores = model((input_ids, token_type_ids, attention_mask))
        scores = torch.softmax(scores, 0)
        if not args.wo_prior:
            scores = scores * pred
            # scores = pred
            scores = scores / torch.sum(scores)
        scores = scores.cpu().detach().numpy().tolist()
        # scores = [round(s,4) for s in scores]
    result = [(v[0], s, v[1]) for s, v in zip(scores, labels)]
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    result = [(v, round(s, 4), l) for v, s, l in result]
    if args.metric == 'MAP':
        tot, n_right = 0.0, 0.0
        ar = 0
        for index, res in enumerate(sorted_result):
            if res[2] == 1:
                n_right += 1.0
                tot += n_right / (index + 1.0)
                if ar == 0:
                    ar = 1 / (index + 1.0)
        ap = 0 if n_right == 0 else tot / n_right
        return result, ap
    if args.metric == 'MRR':
        ar = 0
        for index, res in enumerate(sorted_result):
            if res[2] == 1:
                if ar == 0:
                    ar = 1 / (index + 1.0)
                break
        return result, ar
    if args.metric == 'P5':
        tot, n_right = 0.0, 0.0
        for index, res in enumerate(sorted_result[:5]):
            if res[2] == 1:
                n_right += 1.0
        # ap = 0 if n_right == 0 else tot / n_right
        return n_right / 5, result


def test(model, datas, tokenizer, prior_preds, args):
    model = model.eval()
    results = []
    MAP = 0
    query_aps = {}
    for data in datas:
        # print(data)
        qname = data['id']
        res, ap = test_iterator(
            model, data, tokenizer, prior_preds, args)
        results.append((qname, res, ap))
        MAP += ap
    return MAP / len(datas), results


def M_step(model, batch, args):
    loss = 0
    model = model.train()
    input_ids, token_type_ids, attention_masks, labels, bias, pos_num, select_datas, priors = batch
    if args.cuda:
        input_ids = input_ids.cuda()
        token_type_ids = token_type_ids.cuda()
        attention_masks = attention_masks.cuda()
        labels = labels.cuda()
        priors = priors.cuda()
        bias = bias.cuda()
    score = model((input_ids, token_type_ids, attention_masks))
    score = score + torch.log(priors)
    score = score - bias
    pos_score = score[:pos_num]
    neg_score = score[pos_num:]
    pos_loss = (labels[:pos_num] * torch.nn.functional.logsigmoid(pos_score)).sum()
    neg_loss = torch.nn.functional.logsigmoid(-neg_score)
    neg_loss = neg_loss.sum() / args.sample_num
    # ce_loss = model.criterion(score, labels)
    ce_loss = (pos_loss + neg_loss) / (2.0 * args.mb)
    ce_loss = -ce_loss
    loss = ce_loss
    return loss, ((labels[:pos_num] * pos_score).sum() / args.mb, neg_score.mean(), ce_loss)


def train(model, args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    preds = {}
    # doc_tf = get_doctf(Docs)
    # Doc_inds = prepare(vocab, Docs,args.max_len)
    test_datas = json.load(open(args.datapath))
    # datas = preprocess_data(datas, querys, vocab, preds, args)
    # train_dataset = Traindataset(train_datas, tokenizer, args)
    if args.data_type == 'only_test':
        datas = test_datas
        dev_datas = test_datas
    elif args.data_type=='with_dev':
        dev_datas = json.load(open(args.dev_path))
        datas = test_datas + dev_datas
    elif args.data_type=='all':
        train_datas = json.load(open(args.train_path))
        dev_datas = json.load(open(args.dev_path))
        datas = train_datas + dev_datas + test_datas
    preds = json.load(open(args.preds_path))
    for key in preds:
        # tem_pred = [math.exp(x * args.alpha) for x in preds[key]]
        tem_pred = preds[key]
        tot = sum(tem_pred)
        tem_pred = [x / tot for x in tem_pred]
        preds[key] = tem_pred
    # for i,ele in enumerate(datas):
    #     uid = ele['id']
    #     # preds[uid] = [1.0 for _ in range(len(ele['candidates']))]
    #     preds[uid] = [float(x['score']) / args.alpha for x in ele['candidates']]
    #     preds[uid] = [math.exp(x) for x in preds[uid]]
    #     tot = sum(preds[uid])
    #     preds[uid] = [x / tot for x in preds[uid]]
    # for ele in test_datas:
    #     uid = ele['ORGQ_ID']
    #     preds[uid] = [1.0 / float(x['RELQ_RANKING_ORDER']) for x in ele['RelQuestions']]
    #     # preds[uid] = [1.0 / i for i in range(1,len(ele['RelQuestions']) + 1)]
    #     tot = sum(preds[uid])
    #     preds[uid] = [x / tot for x in preds[uid]]

    prior_preds = preds.copy()
    neg_candidates = []
    for ele in datas:
        neg_candidates += ele['candidates']
    all_dataset = Mydataset(datas, tokenizer, neg_candidates, prior_preds, args)
    if args.optim == 'adamw':
        optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optim == 'adam':
        optim = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    if args.optim == 'sgd':
        optim = SGD(model.parameters(), lr=args.lr)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.update_steps * args.epoch)
    # warmup_scheduler = warmup.LinearWarmup(optim,args.update_steps)
    tot_loss = 0
    loss_time = 0
    max_MAP = 0
    max_dev_MAP = 0
    final_MAP = 0
    stop = 0
    # init
    confirm = 0
    for key in preds:
        lis = preds[key]
        score = max(lis)
        confirm += 1 if score > 0.9 else 0
    logging.info('confirm:%s/%s' % (confirm, len(datas)))
    # print('confirm:', confirm, '/', len(datas))
    # em
    # preds = E_step(model, datas, querys, vocab, args)

    # gloss init
    parameters = [p for p in model.parameters() if p.requires_grad]

    # M_step
    for epoch in range(args.epoch):
        tot_loss = 0
        loss_time = 0
        tot_pos_score = 0
        tot_neg_score = 0
        tot_eloss = 0
        # for i in tqdm(range(args.update_steps)):
        for i in range(args.update_steps):
            # batch = generate_batch(datas, preds, querys, vocab, keys, args)
            batch = all_dataset.generate_batch(preds, args)
            loss, (pos_score, neg_score, e_loss) = M_step(model, batch, args)
            optim.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            # lr_scheduler.step()
            # warmup_scheduler.dampen()
            l = loss.item()
            tot_loss += l
            loss_time += 1
            tot_eloss += e_loss.item()
            tot_pos_score += pos_score.item()
            tot_neg_score += neg_score.item()
            # if i % args.print_every == 0:
            # print('goal:' + str(cal_goal(model,datas,querys,vocab,preds,args)))
        test_MAP, results = test(model, test_datas, tokenizer, prior_preds, args)
        dev_MAP, _ = test(model, dev_datas, tokenizer, prior_preds, args)
        test_MAP = round(test_MAP, 4)
        dev_MAP = round(dev_MAP, 4)
        stop += 1
        writer.add_scalar('loss', tot_loss / loss_time, epoch)
        writer.add_scalars('scores', {'pos_score': tot_pos_score / loss_time, 'neg_score': tot_neg_score / loss_time},
                           epoch)
        writer.add_scalar('dev_MAP', dev_MAP, epoch)
        writer.add_scalar('test_MAP', test_MAP, epoch)
        if dev_MAP > max_dev_MAP:
            stop = 0
            if os.path.exists(args.save_dir + str(max_dev_MAP) + 'all.model.pkl'):
                os.remove(args.save_dir + str(max_dev_MAP) + 'all.test.res')
                os.remove(args.save_dir + str(max_dev_MAP) + 'all.model.pkl')
                os.remove(args.save_dir + str(max_dev_MAP) + 'all.preds.txt')
            max_dev_MAP = dev_MAP
            final_MAP = test_MAP
            write_result(args.save_dir + str(max_dev_MAP) + 'all.test.res', results)
            with open(args.save_dir + str(max_dev_MAP) + 'all.preds.txt', 'w') as fout:
                for key in preds:
                    fout.write(key + '\t' + str(preds[key]) + '\n')
            check_point = {}
            check_point['model_dict'] = model.state_dict()
            torch.save(check_point, args.save_dir +
                       str(max_dev_MAP) + 'all.model.pkl')
        # max_val_MAP = max(val_MAP,max_val_MAP)
        write_result(args.save_dir + 'epoch' + str(epoch) + 'all.test.res', results)
        with open(args.save_dir + 'epoch' + str(epoch) + 'all.preds.txt', 'w') as fout:
            for key in preds:
                fout.write(key + '\t' + str(preds[key]) + '\n')
        logging.info('\nepoch: %s, avg_loss:%s' % (epoch, tot_loss / loss_time))
        loss_time = 0
        tot_loss = 0
        max_MAP = max(max_MAP, test_MAP)
        logging.info('test_MAP: %s/%s' % (test_MAP, max_MAP))
        logging.info('dev_MAP: %s/%s' % (dev_MAP, max_dev_MAP))
        # logging.info('val MAP: %s/%s' %(val_MAP,max_val_MAP))
        if stop > 30:
            writer.add_hparams(
                {'dataset': args.name, 'lr': args.lr, 'mb': args.mb, 'neg_sample': args.neg_sample,
                 'sample_num': args.sample_num,
                 'update_steps': args.update_steps, 'data_type': args.data_type, 'seed': args.seed,
                 'alpha': args.alpha}, {'final_dev_MAP': max_dev_MAP, 'final_test_MAP': final_MAP})
            exit()
        # print(len(datas))
        if args.data_type != 'only_test':
            _, results = test(model, datas, tokenizer, prior_preds, args)
        new_preds = {}
        for ele in results:
            tem_pred = [x[1] for x in ele[1]]
            tot = sum(tem_pred)
            tem_pred = [x / tot for x in tem_pred]
            new_preds[ele[0]] = tem_pred
        tot_dis = 0
        for key in new_preds:
            new_pred = np.array(new_preds[key])
            pred = np.array(preds[key])
            if np.argmax(new_pred) != np.argmax(pred):
                tot_dis += 1
        writer.add_scalar('tot_update_dis', tot_dis, epoch)
        if not args.wo_estep:
            preds.update(new_preds)
        logging.info('E_step')
        confirm = 0
        for key in preds:
            lis = preds[key]
            score = max(lis)
            confirm += 1 if score > 0.9 else 0
        writer.add_scalar('confirm', confirm, epoch)
        logging.info('confirm:%s/%s' % (confirm, len(datas)))
        # model = init_PMI(model,optim,gloss,querys,vocab,args)
        # model = init_PMI(model,optim,gloss,querys,vocab,args)
    writer.add_hparams(
        {'dataset': args.name, 'lr': args.lr, 'mb': args.mb, 'neg_sample': args.neg_sample,
         'sample_num': args.sample_num,
         'update_steps': args.update_steps, 'data_type': args.data_type, 'seed': args.seed,
         'alpha': args.alpha}, {'final_dev_MAP': max_dev_MAP, 'final_test_MAP': final_MAP})





def main(args):
    global writer
    writer = SummaryWriter(comment=args.comments)
    log_dir = writer.log_dir

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    d2v = None
    w2v = None
    args.save_dir = log_dir + '/'
    logging.info('printting hyparameters')
    for arg in vars(args):
        logging.info("%s:%s" % (arg, getattr(args, arg)))
    # if args.d2vpath is not '':
    #    d2v = np.load(args.d2vpath)
    model_module = 'models.'
    model_module = model_module + args.model_type
    model_module = importlib.import_module(model_module)
    Model = model_module.Model
    model = Model(args)
    # model = torch.nn.DataParallel(model)
    if args.cuda:
        model = model.cuda()
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path)
        model_dict = model.state_dict()
        model_dict.update(checkpoint['model_dict'])
        model.load_state_dict(model_dict)
    if args.do_train:
        logging.info('start training')
        try:
            train(model, args)
        except Exception:
            logging.error('Train failed', exc_info=True)
        exit()
    # doc2id = json.load(open(args.doc2id_path))
    # id2doc = {doc2id[key]: key for key in doc2id}
    test_datas = json.load(open(args.datapath))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    prior_preds = {}
    pred = json.load(open(args.pred_path))
    for ele in test_datas:
        prior_preds[ele['id']] = pred[ele['target_word']]
    querys = json.load(open(args.query_path))
    vocab = {}
    gloss = json.load(open(args.gloss_path))
    MAP, val_MAP, results = test(model, test_datas, querys, vocab, gloss, tokenizer, prior_preds, args)

    # Docs = json.load(open(args.docs_path))
    # doc_tf = get_doctf(Docs)
    # Doc_inds = prepare(vocab, Docs)
    # test(model, test_data, querys,Doc_inds, vocab, doc2id, doc_tf,args)
