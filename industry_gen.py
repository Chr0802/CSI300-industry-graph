import torch
import pandas as pd
import numpy as np
import pickle
import json
import tqdm 
import logging
from argparse import ArgumentParser

from ltp import LTP
from transformers import BertTokenizer


IND_DIR_PATH = '/home/chenhaoran/FinancialTextAnalysis/coling2018/csi300_industry_graph/'
MODEL_DIR_PATH = '/home/chenhaoran/FinancialTextAnalysis/coling2018/abrt_code/'

TRANSH_FILENAME = 'industry_embeddings.dict' 
NAME2INDS_FILENAME = 'name_to_industries.dict'
TEXT_VOCAB_FILENAME = 'TEXT.vocab._min_count5_sep_word.pkl' # vocab object of dataset's TEXT field
TARGET_VOCAB_FILENAME = 'TARGET.vocab._min_count5_sep_word.pkl' # vocab object of dataset's TEXT field

LTP_DICT_PATH = '/home/chenhaoran/FinancialTextAnalysis/dictionary/Econ_Dict.txt'

def init_logger():
	fmt = "%(asctime)s - %(levelname)s - %(message)s"
	handler = logging.StreamHandler()
	handler.setFormatter(logging.Formatter(fmt))

	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	logger.addHandler(handler)

	return logger

def parse_args():
    parser = ArgumentParser(description='generate industry embedding')
    parser.add_argument('--method', type=str, default='w2v')
    parser.add_argument('--save', action = "store_true")
    args = parser.parse_args()
    return args

def genIndustryEmbedding(emb_name):
    """ `emb_name`: semantic embedding method for industry node, w2v or bert """
    logger = init_logger()
    if emb_name == 'w2v' or emb_name == 'bert':
        logger.info(f'generate industry node embedding with transH and {emb_name}')
    else:
        raise RuntimeError(f'Wrong embedding method param: {emb_name}')
    # load transH embedding as part of industry node embedding
    logger.info('loading transH embedding file and name2inds file ...')
    with open(IND_DIR_PATH + TRANSH_FILENAME, 'r') as f:
        label_to_node_emb = eval(f.readline())
    # load the industries of each listed company
    globals = {'nan': 0}
    with open(IND_DIR_PATH + NAME2INDS_FILENAME, 'r') as f:
        name_to_inds = eval(f.readline(), globals)
    # ensure every industry in inds of name_to_inds have correspond transH embedding
    for key in name_to_inds.keys():
        inds = []
        for ind in name_to_inds[key]:
            if ind:
                if not label_to_node_emb.get(ind):
                    print(key, ind)
                inds.append(ind)
        name_to_inds[key] = inds

    # number the industries
    idx = 0
    label_to_idx = {}
    for key in label_to_node_emb:
        label_to_idx[key] = idx
        idx += 1

    # map targetId to list of industry transH embeddings 
    logger.info('map target company to list of industry transH embeddings ...')
    with open(MODEL_DIR_PATH + TARGET_VOCAB_FILENAME, 'rb') as f:
        target_vocab = pickle.load(f)
    name_to_node_embs = {} # 目标实体名称到list of embedding of industries
    targetId_to_node_embs = {} # 目标实体id到list of embedding of industries
    targetId_to_len_inds = {} # 目标实体id到涉及产业数量
    for key in name_to_inds.keys():
        inds = name_to_inds[key]
        try:
            list_of_node_embs = list(map(lambda x: label_to_node_emb.get(x), inds))
        except TypeError:
            print(key, inds)
            list_of_node_embs = None
        name_to_node_embs[key] = list_of_node_embs
        targetId = target_vocab.stoi.get(key)
        if targetId:
            targetId_to_node_embs[targetId] = list_of_node_embs
            targetId_to_len_inds[targetId] = len(list_of_node_embs)
        else:
            print(key)

    # generate label semantic embedding for every industry node
    if emb_name == 'w2v':
        ltp = LTP()
        with open(MODEL_DIR_PATH + TEXT_VOCAB_FILENAME, 'rb') as f:
            text_vocab = pickle.load(f)
        # user_dict.txt 是词典文件， max_window是最大前向分词窗口
        ltp.init_dict(path=LTP_DICT_PATH, max_window=10)
        # 也可以在代码中添加自定义的词语
        ltp.add_words(words=["</s>", "</p>"], max_window=4)
        def sep_word(string_list):
            segment, _ = ltp.seg(string_list)
            return segment
        # seperate industry label
        list_of_label_token_idx = []
        vocab_dict = text_vocab.stoi
        logger.info('split word for industry label and generate list_of_label_token_idx ...')
        for key in tqdm.tqdm(label_to_idx):
            label_tokens = sep_word([key])[0]
            label_tokens_idx = list(map(lambda x: vocab_dict.get(x) if vocab_dict.get(x) else 0, label_tokens))
            list_of_label_token_idx.append(label_tokens_idx)
        
        # 查看有多少个产业名label没在词表里面 1562
        counter, unk_list = 0, []
        for idx, item in enumerate(list_of_label_token_idx):
            if item == [0]:
                counter += 1
                unk_list.append(idx)
        
        # 公司涉及产业数最多为100，产业名称包含token数最多为12
        MAX_INDS_LEN = 100
        MAX_TOKENS_LEN = 12
        name_to_token_idx = {} # 目标实体名称到list of token idx of industries
        targetId_to_token_idx = {} # 目标实体id到list of token idx of industries
        targetId_inds_len_tokens = {}
        logger.info('generate targetId_to_token_idx and targetId_inds_len_tokens')
        for key in tqdm.tqdm(name_to_inds.keys()):
            inds = name_to_inds[key]
            try:
                list_of_token_idx = list(map(lambda x: list_of_label_token_idx[label_to_idx.get(x)], inds))
                # max_len = max(list(map(lambda x: len(x), list_of_token_idx)))
                # 1 在vocab.stoi中对应的key为<PAD>
                padded_list_of_token_idx = list(map(lambda x: x + [1] * (MAX_TOKENS_LEN - len(x)), list_of_token_idx))
            except TypeError:
                print(key, inds)
                list_of_token_idx = None
            name_to_token_idx[key] = padded_list_of_token_idx
            targetId = target_vocab.stoi.get(key)
            if targetId:
                targetId_to_token_idx[targetId] = padded_list_of_token_idx
                inds_len_tokens = list(map(lambda x: len(x), list_of_token_idx))
                targetId_inds_len_tokens[targetId] = inds_len_tokens
            else:
                print(key)
    # for bert
    else:
        logger.info('load BertTokenizer from pretrained model')
        tokenizer = BertTokenizer.from_pretrained("/home/chenhaoran/FinancialTextAnalysis/model/FinBERT_L-12_H-768_A-12_pytorch")
        pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        logger.info('split word for industry label and generate list_of_label_token_idx_bert ...')
        list_of_label_token_idx_bert = []
        for key in tqdm.tqdm(label_to_idx):
            label_tokens_idx = tokenizer.encode(key, add_special_tokens=False)
            list_of_label_token_idx_bert.append(label_tokens_idx)
        
        MAX_INDS_LEN = 100
        MAX_TOKENS_LEN = 22
        name_to_token_idx = {} # 目标实体名称到list of token idx of industries
        targetId_to_token_idx = {} # 目标实体id到list of token idx of industries
        targetId_inds_len_tokens = {}
        logger.info('generate targetId_to_token_idx and targetId_inds_len_tokens')
        for key in tqdm.tqdm(name_to_inds.keys()):
            inds = name_to_inds[key]
            try:
                list_of_token_idx = list(map(lambda x: list_of_label_token_idx_bert[label_to_idx.get(x)], inds))
                # max_len = max(list(map(lambda x: len(x), list_of_token_idx)))
                padded_list_of_token_idx = list(map(lambda x: x + [pad_index] * (MAX_TOKENS_LEN - len(x)), list_of_token_idx))
            except TypeError:
                print(key, inds)
                list_of_token_idx = None
            name_to_token_idx[key] = padded_list_of_token_idx
            targetId = target_vocab.stoi.get(key)
            if targetId:
                targetId_to_token_idx[targetId] = padded_list_of_token_idx
                inds_len_tokens = list(map(lambda x: len(x), list_of_token_idx))
                targetId_inds_len_tokens[targetId] = inds_len_tokens
            else:
                print(key)
        

    return targetId_to_node_embs, targetId_to_token_idx, targetId_to_len_inds, targetId_inds_len_tokens
        

if __name__ == "__main__":
    args = parse_args()
    targetId_to_node_embs, targetId_to_token_idx, targetId_to_len_inds, targetId_inds_len_tokens = genIndustryEmbedding(args.method)
    if args.save:
        with open(f'/home/chenhaoran/FinancialTextAnalysis/coling2018/csi300_industry_graph/{args.method}/targetId_to_node_embs.pkl', 'wb') as f:
            pickle.dump(targetId_to_node_embs, f)
        with open(f'/home/chenhaoran/FinancialTextAnalysis/coling2018/csi300_industry_graph/{args.method}/targetId_to_token_idx.pkl', 'wb') as f:
            pickle.dump(targetId_to_token_idx,f)
        with open(f'/home/chenhaoran/FinancialTextAnalysis/coling2018/csi300_industry_graph/{args.method}/targetId_to_len_inds.pkl', 'wb') as f:
            pickle.dump(targetId_to_len_inds, f)
        with open(f'/home/chenhaoran/FinancialTextAnalysis/coling2018/csi300_industry_graph/{args.method}/targetId_inds_len_tokens.pkl', 'wb') as f:
            pickle.dump(targetId_inds_len_tokens, f)

