import argparse
import collections
import glob
import json
import math
import numpy as np
import random
from ordered_set import OrderedSet
import os
import pickle
import shutil
from sklearn.metrics import average_precision_score
import sys
import termcolor
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import torch.optim as optim
from tqdm import tqdm

from NAACL import vocabulary
from NAACL import settings
from NAACL import util

WORD_VEC_FILE = 'wordvec/PubMed-and-PMC-w2v.txt'
WORD_VEC_NUM_LINES = 4087447
EMB_SZIE = 200 # size of word embeddings
PARA_EMB_SIZE = 100 # size of paragraph index embeddings
PARA_EMB_MAX_SPAN = 1000
MAX_ENTITIES_PER_TYPE = 200
MAX_NUM_PARAGRAPHS = 200
MAX_NUM_CANDIDATES = 10000
ALL_ENTITY_TYPES = ['drug', 'gene', 'variant']
ALL_ENTITY_TYPES_PAIRS = [('drug', 'gene'), ('drug', 'variant'), ('gene', 'variant')]
MAX_PARAGRAPH_LENGTH = 800
CLIP_THRESH = 5 # Gradient clipping (on L2 norm)
JAX_DEV_PMIDS_FILE = 'jax/jax_dev_pmids.txt'
JAX_TEST_PMIDS_FILE = 'jax/jax_test_pmids.txt'

log_file = None

def log(msg):
    print(msg, file=sys.stderr)
    if log_file:
        print(msg, file=log_file)

ParaMention = collections.namedtuple(
    'ParaMention',['start', 'end', 'type', 'name'])

class Candidate(object):
    def __init__(self, drug=None, gene=None, variant=None, label=None):
        self.drug = drug
        self.gene = gene
        self.variant = variant
        self.label = label

    def remove_entity(self, i, new_label=None):
        '''
        :param i:
        :param new_label:
        :return: Return new Candidate with entity |i| replaced with None.
        '''
        triple = (self.drug, self.gene, self.variant)
        new_triple = triple[:i] + (None,) + triple[i+1:]
        return Candidate(*new_triple, label=new_label)

    def get_entities(self):
        return (self.drug, self.gene, self.variant)

    def is_triple(self):
        return self.drug and self.gene and self.variant

    def get_types(self):
        out = []
        if self.drug:
            out.append('drug')
        if self.gene:
            out.append('gene')
        if self.variant:
            out.append('variant')
        return tuple(out)

    def __key(self):
        return (self.drug, self.gene, self.variant, self.label)

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())


class Example(object):
    def __init__(self, pmid, paragraphs, mentions, triple_candidates, pair_candidates):
        self.pmid = pmid
        self.paragraphs = paragraphs
        self.mentions = mentions
        self.triple_candidates = triple_candidates
        self.pair_candidates = pair_candidates
        self.entities = collections.defaultdict(OrderedSet)
        for m_list in mentions:
            for m in m_list:
                self.entities[m.type].add(m.name)

    @classmethod
    def read_examples(cls, example_json_file):
        results = []
        with open(os.path.join(settings.DATA_DIR, example_json_file)) as f:
            for line in f:
                ex = cls.read_examples(line)
                results.append(ex)

        return results

    @classmethod
    def read_examples(cls, example_json_str):
        example_json = json.loads(example_json_str)
        mentions = [[ParaMention(**mention) for mention in paragraph_mentions]
                    for paragraph_mentions in example_json['mentions']]
        pair_candidates = {}

        for pair_key in example_json['pair_candidates']:
            pair_key_tuple = tuple(json.loads(pair_key))
            pair_candidates[pair_key_tuple] = OrderedSet(Candidate(**x)
                                                         for x in example_json['pair_candidates'][pair_key])
            triple_candidates = {}
            triple_candidates = [Candidate(**x)
                                 for x in example_json['triple_candidates']]

            return cls(example_json['pmid'],
                       example_json['paragraphs'],
                       mentions,
                       triple_candidates,
                       pair_candidates)

class Preprocessor(object):

    def __init__(self, entity_lists, vacab, device):
        self.entity_lists = entity_lists
        self.vocab = vacab
        self.device = device

    def count_labels(self, ex, pair_only=None):
        if pair_only:
            candidates = ex.pair_candidates[pair_only]
        else:
            candidates = ex.triple_candidates

        num_pos = sum(c.label for c in candidates)
        num_neg = sum(1 - c.label for c in candidates)
        return num_neg, num_pos

    def shuffle_entities(self, ex):
        entity_map = {}
        for e_type in ex.entities:
            cur_ents = ex.entities[e_type]
            replacements = random.sample(self.entity_lists[e_type], len(cur_ents))
            for e_old, e_new in zip(cur_ents, replacements):
                entity_map[(e_type, e_old)] = e_new

        new_paras = []
        new_mentions = []
        for p, m_list in zip(ex.paragraphs, ex.mentions):
            new_para = []
            new_m_list =[]
            mentions_at_loc = collections.defaultdict(list)
            in_mention = [False] * len(p)
            for m in m_list:
                mentions_at_loc[m.start].append((m.type, m.name))
                for i in range(m.start, m.end):
                    in_mention[i] = True
            for i in range(len(p)):
                if mentions_at_loc[i]:
                    for e_type, name in mentions_at_loc[i]:
                        e_new = entity_map[(e_type, name)]
                        m = ParaMention(len(new_para), len(new_para)+1, e_type, name)
                        new_m_list.append(m)
                        new_para.append(e_new)
                if not in_mention[i]:
                    new_paras.append(p[i])
            new_paras.append(new_para)
            new_mentions.append(new_m_list)
        return new_paras, new_mentions

    def preprocess(self, ex, pair_only):
        new_paras, new_mentions = self.shuffle_entities(ex)
        para_prep = []
        for para_idx, (para, m_list) in enumerate(zip(new_paras, new_mentions)):
            word_idxs = torch.tensor(self.vocab.indexify_list(para),
                                     dtype=torch.long, device=self.device)

            para_from_start = [
                para_idx / math.pow(PARA_EMB_MAX_SPAN, 2*i / (PARA_EMB_SIZE // 4))
                for i in range(PARA_EMB_SIZE // 4)
            ]

            para_from_end = [
                (len(new_paras)- para_idx) / math.pow(PARA_EMB_MAX_SPAN, 2*i / (PARA_EMB_SIZE // 4))
                for i in range(PARA_EMB_SIZE // 4)
            ]

            para_args = torch.cat([torch.tensor(x, dtype=torch.float, device=self.device)
                                   for x in (para_from_start, para_from_end)])

            para_vec = torch.cat([torch.sin(para_args), torch.cos(para_args)])
            para_prep.append((word_idxs ,para_vec, m_list))

            # sort for pack_padded_sequence
            para_prep.sort(key=lambda x:len(x[0]), reverse=True)
            T, P = len(para_prep[0][0]), len(para_prep)
            para_mat = torch.zeros((T, P), device=self.device, dtype=torch.long)
            for i, x in enumerate(para_prep):
                cur_words = x[0]
                para_mat[:len(cur_words), i] = cur_words

            lenghts = torch.tensor([len(x[0]) for x in para_prep], device=self.device)
            triple_labels = torch.tensor([c.label for c in ex.triple_candidates],
                                         dtype=torch.float, device=self.device)
            pair_labels = {k: torch.tensor([c.label for c in ex.pair_candidates[k]],
                                          dtype=torch.float, device=self.device)
                                    for k in ex.pair_candidates}
            para_vecs = torch.stack([x[1] for x in para_prep], dim=0)
            unlabeled_triple_cands = [Candidate(ex.drug, ex.gene, ex.variant)
                                      for ex in ex.triple_candidates]
            unlabeled_pair_cands = {k: [Candidate(ex.drug, ex.gene, ex.variant)
                                        for ex in ex.pair_candidates[k]]
                                    for k in ex.pair_candidates}
            return (para_mat, lenghts, para_vecs, [x[2] for x in para_prep],
                    unlabeled_triple_cands, unlabeled_pair_cands, triple_labels, pair_labels)

def logsumexp(inputs, dim=None, keepdim=False):
    '''

    :param inputs: A variable with any shape.
    :param dim: An integer.
    :param keepdim: A boolean.
    :return: Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    '''
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

class BackoffModel(nn.Module):
    '''
    Combine triple and pairwise information.
    '''

    def __init__(self, emb_mat, lstm_size, lstm_layers, device, use_lstm=True,
                 use_position=True, pool_method='max', dropout_prob=0.5, vocab=None,
                 pair_only=None):

        super(BackoffModel, self).__init__()
        self.device = device
        self.use_lstm = use_lstm
        self.use_position = use_position
        self.pool_method - pool_method
        self.embs = nn.Embedding.from_pretrained(emb_mat, freeze=False)
        self.vocab = vocab
        self.pair_only =pair_only
        self.dropout = nn.Dropout(p=dropout_prob)
        para_emb_size = PARA_EMB_SIZE if use_position else 0
        if use_lstm:
            self.lstm_layers = lstm_layers
            self.lstm = nn.LSTM(EMB_SZIE + para_emb_size, lstm_size,
                                bidirectional=True, num_layers=lstm_layers)
        else:
            self.emb_linear = nn.Linear(EMB_SZIE + para_emb_size, 2 * lstm_size)
        for t1 ,t2 in ALL_ENTITY_TYPES_PAIRS:
            setattr(self, 'hidden_%s_%s' %
                    (t1, t2), nn.Linear(4 * lstm_size, 2 * lstm_size))
            setattr(self, 'out_%s_%s' % (t1, t2), nn.Linear(2 * lstm_size, 1))
            setattr(self, 'backoff_%s_%s' % (t1, t2), nn.Parameter(
                torch.zeros(1, 2 * lstm_size)))
        self.hidden_triple = nn.Linear(3 * 2 * lstm_size, 2 * lstm_size)
        self.backoff_triple = nn.Parameter(torch.zeros(1, 2 * lstm_size))
        self.hidden_all = nn.Linear(4 * 2 * lstm_size, 2 * lstm_size)
        self.out_triple = nn.Linear(2 * lstm_size, 1)

    def pool(self, grouped_vecs):
        '''

        :param grouped_vecs:
        :return:
        '''

