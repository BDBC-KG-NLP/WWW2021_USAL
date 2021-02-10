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
from transformers import BertTokenizer
from utils import write_result, log_result, trec_evaluate, create_testdata, test_tfidf, bm25, tfidf, get_doctf, cal_IDF
from torch.utils.tensorboard import SummaryWriter
import logging

'''
em4基础上 E步改为softmax
'''


def pad(docs, doc_lens, args):
    max_len = min(args.max_len, max(doc_lens))
    max_len = max(max_len, 5)
    doc_lens = [min(ele, args.max_len) for ele in doc_lens]
    for i in range(len(docs)):
        if len(docs[i]) <= max_len:
            docs[i] = docs[i] + [0] * (max_len - len(docs[i]))
        else:
            docs[i] = docs[i][:max_len]
    return docs, doc_lens


def generate_bertinput(data, sense, Gloss):
    target_word = data['target_word']
    word = target_word.split('#')[0]
    gloss = word + ":" + " ".join(Gloss[sense])
    context = data['context']
    # pos = data['target_position']
    # context = " ".join(context[:pos]) + ' "' + context[pos] + '" ' + " ".join(context[pos+1:])
    context = " ".join(context)
    # doc = '[CLS] ' + context + '[SEP] ' + gloss

    return context, gloss


class Mydataset(object):
    def __init__(self, datas, tokenizer, querys, priors, args):
        self.datas = datas
        self.batch_size = args.mb
        self.select = list(range(len(datas)))
        self.index = 0
        random.shuffle(self.select)
        self.index_generater = self.get_batch()
        self.tokenizer = tokenizer
        self.neg_counts = {}
        self.priors = {}
        for word in querys:
            for sense,value in zip(querys[word],priors[word]):
                self.priors[sense] = value



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

    def generate_batch(self, preds, querys, vocab, keys, Gloss, args):
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
            if len(preds[uid]) > 1:
                select_datas.append(data)
            target_word = data['target_word']
            word = target_word.split('#')[0]
            for i, sense in enumerate(querys[target_word][:20]):
                if preds[uid][i] == 0:
                    continue
                doc = generate_bertinput(data, sense, Gloss)
                docs.append(doc)
                labels.append(preds[uid][i])
                Pw = 1. / len(querys[target_word])
                # Pw = 1. / len(querys[target_word])
                bias.append(math.log(args.sample_num * Pw))
                pos_num += 1
                priors.append(self.priors[sense])
        # neg sample
        for index in select:
            data = self.datas[index]
            target_word = data['target_word']
            word = target_word.split('#')[0]
            # neg_candicate = keys
            if args.neg_sample == 'mix':
                neg_senses = list(
                    np.random.choice(data['neg_candidates'], int(args.sample_num * args.neg_ratio))) + list(
                    np.random.choice(
                        keys, int(args.sample_num * (1 - args.neg_ratio))))
                Pw = 1. / len(keys)
            if args.neg_sample == 'querys':
                neg_candidates = data['neg_candidates']
                neg_senses = np.random.choice(neg_candidates, args.sample_num)
                Pw = 1 / len(neg_candidates)
            elif args.neg_sample == 'keys':
                neg_candidates = keys
                neg_senses = np.random.choice(neg_candidates, args.sample_num)
                Pw = 1 / len(neg_candidates)
            for sense in neg_senses:
                self.neg_counts.setdefault(uid, {})
                self.neg_counts[uid].setdefault(sense, 0)
                self.neg_counts[uid][sense] += 1
                # sense = np.random.choice(neg_senses, 1)[0]
                # gloss = " ".join(Gloss[sense])
                # doc = '[CLS] ' + " ".join(data['context']) + '[SEP] ' + gloss
                doc = generate_bertinput(data, sense, Gloss)
                docs.append(doc)
                bias.append(math.log(args.sample_num * Pw))
                labels.append(0)
                priors.append(self.priors[sense])
        output = self.tokenizer(docs, padding=True, max_length=500)
        input_ids = torch.LongTensor(output['input_ids'])
        token_type_ids = torch.LongTensor(output['token_type_ids'])
        attention_mask = torch.LongTensor(output['attention_mask'])
        labels = torch.Tensor(labels)
        bias = torch.Tensor(bias)
        priors = torch.Tensor(priors)
        return input_ids, token_type_ids, attention_mask, labels, bias, pos_num, select_datas, priors



def encode_docs(doc_ids, Docs, vocab, args):
    docs = [Docs[ele] for ele in doc_ids]
    doc_lens = [len(Docs[ele]) for ele in doc_ids]
    return pad(docs, doc_lens, args)


def prepare(vocab, docs, max_len=1000):
    Doc_inds = {}
    for doc in docs:
        doc_text = [vocab[w] for w in docs[doc]]
        doc_text = doc_text[:min(len(doc_text), max_len)]
        Doc_inds[doc] = doc_text
    return Doc_inds


def test_iterator(model, data, querys, vocab, Gloss, tokenizer, prior_preds, args):
    qname = data['target_word']
    target_word = qname.split('#')[0]

    docs = [generate_bertinput(data, sense, Gloss) for sense in querys[qname]]
    output = tokenizer(docs, padding=True, max_length=500)
    input_ids = torch.LongTensor(output['input_ids'])
    token_type_ids = torch.LongTensor(output['token_type_ids'])
    attention_mask = torch.LongTensor(output['attention_mask'])
    pred = prior_preds[data['id']]
    pred = torch.Tensor(pred)
    if args.cuda:
        input_ids = input_ids.cuda()
        token_type_ids = token_type_ids.cuda()
        attention_mask = attention_mask.cuda()
        pred = pred.cuda()
    with torch.no_grad():
        scores = model((input_ids, token_type_ids, attention_mask))
        scores = torch.softmax(scores, 0)
        pred = torch.pow(pred,0.1)
        if not args.wo_prior:
            scores = scores * pred
            scores = scores / torch.sum(scores)
        # scores = scores[:,1]
        index = torch.argmax(scores).item()
    results = []
    for i, w in enumerate(querys[qname]):
        results.append((w, round(scores[i].item(), 4)))
    return querys[qname][index], results


def test(model, datas, querys, vocab, gloss, tokenizer, prior_preds, args):
    model = model.eval()
    results = []
    for data in datas:
        ans, res = test_iterator(model, data, querys, vocab, gloss, tokenizer, prior_preds, args)
        if ans != -1:
            results.append((data['id'], data['target_sense'], res, ans, data['target_word']))
    with open(args.save_dir + 'tem.res', 'w') as fout:
        for ele in results:
            fout.write(ele[0] + ' ' + ele[3] + '\n')
    test_F1 = evaluate(args.gold_path, args.save_dir + 'tem.res')
    with open(args.save_dir + 'tem.res', 'w') as fout:
        for ele in results:
            if ele[0].startswith('semeval2007'):
                fout.write(ele[0].replace('semeval2007.', '') + ' ' + ele[3] + '\n')
    val_F1 = evaluate(args.val_gold_path, args.save_dir + 'tem.res')
    return test_F1, val_F1, results

def E_step(model, datas, querys, vocab, gloss, tokenizer, prior_preds, args):
    model = model.eval()
    results = []
    for data in datas:
        ans, res = test_iterator(model, data, querys, vocab, gloss, tokenizer, prior_preds, args)
        if ans != -1:
            results.append((data['id'], data['target_sense'], res, ans, data['target_word']))
    return results

def test_val(model, datas, querys, vocab, gloss, tokenizer, prior_preds, args):
    model = model.eval()
    results = []
    with open('tem.res', 'w') as fout:
        for data in datas:
            ans, res = test_iterator(model, data, querys, vocab, gloss, tokenizer, prior_preds, args)
            fout.write(data['id'].replace('semeval2007.', '') + '\t' + ans + '\n')
    return evaluate(args.val_gold_path, 'tem.res')


def error_analysis(model, datas, querys, vocab, args):
    args.error_analysis = True
    model = model.eval()
    results = []
    for data in datas:
        ans, res = test_iterator(model, data, querys, vocab, args)
        if ans != -1 and ans != data['target_sense']:
            results.append((data['id'], data['target_sense'], res, ans))
    with open('error.res', 'w') as fout:
        fout.write("pos\tneg\n")
        for ele in results:
            for i, data in enumerate(ele[2]):
                if data[0] == ele[1]:
                    pos_ans = data
                if data[0] == ele[3]:
                    neg_ans = data
            fout.write(
                "\t".join([str(ele) for ele in pos_ans]) + "\t" + "\t".join([str(ele) for ele in neg_ans]) + "\n")
    return results


# get F1 score
def evaluate(gold_file, res_file):
    eval_res = subprocess.Popen(
        ['java', 'Scorer', gold_file, res_file],
        stdout=subprocess.PIPE, shell=False)
    (out, err) = eval_res.communicate()
    eval_res = out.decode("utf-8")
    eval_res = eval_res.strip().split()
    index = eval_res.index('F1=') + 1
    res = eval_res[index]
    res = res.split('%')[0]
    return float(res)


def test_all(results, args):
    pre = 'data/All_Words_WSD/Evaluation_Datasets/'
    # semeval2007
    res = {}
    datasets = ['semeval2007', 'senseval2', 'senseval3', 'semeval2013', 'semeval2015']
    for key in datasets:
        with open(args.save_dir + 'tem.res', 'w') as fout:
            for ele in results:
                if ele[0].startswith(key):
                    fout.write(ele[0].replace(key + '.', '') + ' ' + ele[3] + '\n')
        F1 = evaluate(pre + key + '/' + key + '.gold.key.txt', args.save_dir + 'tem.res')
        res[key] = F1
    type = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 'r': 'ADV'}
    for key in type:
        with open(args.save_dir + 'tem' + key + '.res', 'w') as fout:
            for ele in results:
                if ele[4].endswith(key):
                    fout.write(ele[0] + ' ' + ele[3] + '\n')
        F1 = evaluate(pre + 'ALL/' + 'ALL.gold.key.txt.' + type[key], args.save_dir + 'tem' + key + '.res')
        res[key] = F1
    return res




def E_loss(model, datas, querys, vocab, Gloss, tokenizer, prior_preds, args):
    if len(datas) == 0:
        return torch.Tensor([0.]).cuda()
    loss = 0
    for data in datas:
        uid = data['id']
        qname = data['target_word']
        target_word = qname.split('#')[0]
        gloss = [" ".join(Gloss[sense]) for sense in querys[qname]]
        doc = " ".join(data['context'])
        docs = ['[CLS] ' + doc + '[SEP] ' + ele for ele in gloss]
        docs = [tokenizer.tokenize(ele) for ele in docs]
        docs = [tokenizer.convert_tokens_to_ids(doc) for doc in docs]
        doc_lens = [len(doc) for doc in docs]
        mask_s = [[1 for _ in range(ele)] for ele in doc_lens]
        docs, doc_lens = pad(docs, doc_lens, args)
        docs = torch.LongTensor(docs)
        mask_s, _ = pad(mask_s, doc_lens, args)
        mask_s = torch.LongTensor(mask_s)
        if args.cuda:
            docs = docs.cuda()
            mask_s = mask_s.cuda()
        score = model((docs, mask_s))
        # score = torch.sigmoid(score)
        pred = torch.softmax(score, dim=0)
        loss += 1 - torch.norm(pred, p=2)
    loss = loss / len(datas)
    return loss


def M_step(model, batch, querys, vocab, preds, tokenizer, Gloss, args):
    loss = 0
    model = model.train()
    input_ids, token_type_ids, attention_masks, labels, bias, pos_num, select_datas, priors = batch
    if args.cuda:
        input_ids = input_ids.cuda()
        token_type_ids = token_type_ids.cuda()
        attention_masks = attention_masks.cuda()
        labels = labels.cuda()
        bias = bias.cuda()
        priors = priors.cuda()
    score = model((input_ids, token_type_ids, attention_masks))
    score = score + torch.log(priors) + 2.5
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


def merge_datas(querys, preds, gloss, args):
    train_datas = []
    keys = list(querys.keys())
    for key in keys:
        lis = querys[key]
        for sense in lis:
            ele = {}
            target_word = key + '_' + sense
            ele['id'] = target_word
            ele['target_word'] = target_word
            querys[target_word] = [sense]
            preds[target_word] = [1.0]
            ele['context'] = gloss[sense]
            train_datas.append(ele)
    return querys, preds, train_datas


def merge_labeled_data(train_data, ):
    pass


def preprocess_data(datas, querys, vocab, preds, args):
    processed_datas = []
    for data in datas:
        qname = data['target_word']
        pred = preds[data['id']]
        pred = torch.Tensor(pred)
        q_inds = [vocab[w] for w in querys[qname]]
        q_inds = torch.LongTensor(q_inds)
        doc = [vocab[w] for w in data['context']]
        docs = [doc for _ in range(len(querys[qname]))]
        doc_lens = torch.LongTensor([len(data['context'])] * len(querys[qname]))
        docs = torch.LongTensor(docs)
        processed_datas.append([data['id'], q_inds, docs, doc_lens])
    return processed_datas


def train(model, args):
    vocab = json.load(open(args.vocab_path))
    querys = json.load(open(args.query_path))
    gloss = json.load(open(args.gloss_path))
    # Docs = json.load(open(args.docs_path))
    pred = json.load(open(args.preds_path))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    preds = {}
    # doc_tf = get_doctf(Docs)
    # Doc_inds = prepare(vocab, Docs,args.max_len)
    test_datas = json.load(open(args.datapath))
    train_datas = json.load(open(args.train_path))
    keys = [key for key in gloss]
    if args.only_test:
        datas = test_datas
    if args.merge_train:
        random.shuffle(train_datas)
        datas = train_datas[:len(test_datas)]
    for ele in datas:
        target_word = ele['target_word']
        uid = ele['id']
        preds[uid] = pred[target_word]
    for ele in test_datas:
        target_word = ele['target_word']
        uid = ele['id']
        preds[uid] = pred[target_word]
    prior_preds = preds.copy()
    for i, data in enumerate(datas):
        target_word = data['target_word']
        data['neg_candidates'] = querys[target_word]
        datas[i] = data

    # if args.labeled:
    #     for i,ele in enumerate(train_datas):
    #         uid = ele['id']
    #         target_word = ele['target_word']
    #         preds[uid] = [0 if sense != ele['target_sense'] else 1 for sense in querys[target_word]]
    #         ele['neg_candidates'] = [sense for sense in querys[target_word] if sense != ele['target_sense']]
    #         train_datas[i] = ele
    #     datas = test_datas + train_datas
    #     query_lens = [len(preds[key]) for key in preds]
    #     print(max(query_lens))
    #     print(sum(query_lens) / len(query_lens))

    # datas = preprocess_data(datas, querys, vocab, preds, args)
    # train_dataset = Traindataset(train_datas, tokenizer, args)
    all_dataset = Mydataset(datas, tokenizer, querys, pred, args)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if args.optim == 'adam':
        optim = Adam(parameters, lr=args.lr,
                     weight_decay=args.weight_decay)
    if args.optim == 'sgd':
        optim = SGD(parameters, lr=args.lr)
    tot_loss = 0
    loss_time = 0
    max_F1 = 0
    max_val_F1 = 0
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
    goal_sum = 0
    # preds = E_step(model, datas, querys, vocab, args)

    # gloss init

    # M_step
    for epoch in range(args.epoch):
        tot_loss = 0
        loss_time = 0
        tot_pos_score = 0
        tot_neg_score = 0
        tot_eloss = 0
        for i in range(args.update_steps):
            # batch = generate_batch(datas, preds, querys, vocab, keys, args)
            batch = all_dataset.generate_batch(preds, querys, vocab, keys, gloss, args)
            loss, (pos_score, neg_score, e_loss) = M_step(model, batch, querys, vocab, prior_preds, tokenizer, gloss,
                                                          args)
            optim.zero_grad()
            loss.backward()
            optim.step()
            l = loss.item()
            tot_loss += l
            loss_time += 1
            tot_eloss += e_loss.item()
            tot_pos_score += pos_score.item()
            tot_neg_score += neg_score.item()
            # if i % args.print_every == 0:
            # print('goal:' + str(cal_goal(model,datas,querys,vocab,preds,args)))
        F1, val_F1, results = test(model, test_datas, querys, vocab, gloss, tokenizer, prior_preds, args)
        stop += 1
        writer.add_scalars('loss', {'loss': tot_loss / loss_time, 'eloss': tot_eloss / loss_time}, epoch)
        writer.add_scalars('scores', {'pos_score': tot_pos_score / loss_time, 'neg_score': tot_neg_score / loss_time},
                           epoch)
        writer.add_scalar('F1', F1, epoch)
        if F1 > max_F1:
            stop = 0
            if os.path.exists(args.save_dir + str(max_F1) + 'all.model.pkl'):
                os.remove(args.save_dir + str(max_F1) + 'all.test.res')
                os.remove(args.save_dir + str(max_F1) + 'all.model.pkl')
                os.remove(args.save_dir + str(max_F1) + 'all.preds.txt')
            max_F1 = F1
            write_result(args.save_dir + str(max_F1) + 'all.test.res', results)
            with open(args.save_dir + str(max_F1) + 'all.preds.txt', 'w') as fout:
                for key in preds:
                    fout.write(key + '\t' + str(preds[key]) + '\n')
            check_point = {}
            check_point['model_dict'] = model.state_dict()
            torch.save(check_point, args.save_dir +
                       str(max_F1) + 'all.model.pkl')
            args.model_path = args.save_dir + str(max_F1) + 'all.model.pkl'
        max_val_F1 = max(val_F1, max_val_F1)
        write_result(args.save_dir + 'epoch' + str(epoch) + 'all.test.res', results)
        with open(args.save_dir + 'epoch' + str(epoch) + 'all.preds.txt', 'w') as fout:
            for key in preds:
                fout.write(key + '\t' + str(preds[key]) + '\n')
        logging.info('\nepoch: %s, avg_loss:%s' % (epoch, tot_loss / loss_time))
        loss_time = 0
        tot_loss = 0
        logging.info('F1: %s/%s' % (F1, max_F1))
        logging.info('val F1: %s/%s' % (val_F1, max_val_F1))
        if stop > 50:
            exit()
        if args.merge_train:
            results = E_step(model,datas,querys,vocab,gloss,tokenizer,prior_preds,args)
        new_preds = {}
        for ele in results:
            tem_pred = [x[1] for x in ele[2]]
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
        with open(args.save_dir + 'epoch' + str(epoch) + 'samplecount.txt', 'w') as fout:
            for ele in test_datas:
                key = ele['id']
                neg_count = all_dataset.neg_counts[key] if key in all_dataset.neg_counts else []
                fout.write(key + '\t' + str(neg_count) + '\n')
        # model = init_PMI(model,optim,gloss,querys,vocab,args)
        # model = init_PMI(model,optim,gloss,querys,vocab,args)


def case_study(model, querys, vocab, docs, args):
    qname = args.case_query
    dnames = [args.case_doc]
    id2word = {vocab[w]: w for w in vocab}
    doc_text = " ".join([id2word[w] for w in docs[args.case_doc]])
    print(doc_text)
    q_inds = [vocab[w] for w in querys[qname]]
    q_inds = torch.LongTensor(q_inds)
    doc_inds = []
    # doc_lens = np.array([len(docs[doc]) for doc in dnames])
    # doc_lens = doc_lens ** args.weight_power
    # doc_lens = doc_lens / np.sum(doc_lens)
    # doc_lens = np.log(doc_lens)
    results = {}
    if args.cuda:
        q_inds = q_inds.cuda()
    with torch.no_grad():
        q_inds = model.word_encode(q_inds)
        scores1 = []
        for i in range(len(q_inds)):
            tem_qind = q_inds[i].unsqueeze(0).expand(
                len(dnames), q_inds[i].size(0))
            doc_inds = encode_docs(model, tem_qind, dnames, docs, vocab, args)
            scores1.append(model.cal_scores(
                tem_qind, doc_inds, case_study=True))
        scores1 = torch.stack(scores1, 1)
        scores = torch.sum(scores1, 1)


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
    vocab = json.load(open(args.vocab_path))
    args.vocab_size = len(vocab)
    args.save_dir = log_dir + '/'
    logging.info('printting hyparameters')
    for arg in vars(args):
        logging.info("%s:%s" % (arg, getattr(args, arg)))
    if args.w2vpath is not '':
        w2v = np.load(args.w2vpath)
    # if args.d2vpath is not '':
    #    d2v = np.load(args.d2vpath)
    model_module = 'models.'
    model_module = model_module + args.model_type
    model_module = importlib.import_module(model_module)
    global SGNS
    SGNS = model_module.SGNS
    model = SGNS(args)
    model = torch.nn.DataParallel(model)
    if args.cuda:
        model = model.cuda()

    if args.do_train:
        logging.info('start training')
        try:
            train(model, args)
        except Exception:
            logging.error('Train failed', exc_info=True)
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path)
        model_dict = model.state_dict()
        model_dict.update(checkpoint['model_dict'])
        model.load_state_dict(model_dict)
    # doc2id = json.load(open(args.doc2id_path))
    # id2doc = {doc2id[key]: key for key in doc2id}
    vocab = json.load(open(args.vocab_path))
    querys = json.load(open(args.query_path))
    test_datas = json.load(open(args.datapath))
    gloss = json.load(open(args.gloss_path))
    pred = json.load(open(args.preds_path))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    prior_preds = {}
    for ele in test_datas:
        prior_preds[ele['id']] = pred[ele['target_word']]
    F1, val_F1, results = test(model, test_datas, querys, vocab, gloss, tokenizer, prior_preds, args)
    res = test_all(results, args)
    res['all'] = F1
    for key in res:
        print(key + ':' + str(res[key]))
    print(F1, val_F1)

    # Docs = json.load(open(args.docs_path))
    # doc_tf = get_doctf(Docs)
    # Doc_inds = prepare(vocab, Docs)
    # test(model, test_data, querys,Doc_inds, vocab, doc2id, doc_tf,args)
