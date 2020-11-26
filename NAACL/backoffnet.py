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
        if self.pool_method == 'mean':
            return torch.stack([torch.mean(g, dim=0) for g in grouped_vecs])
        elif self.pool_method == 'sum':
            return torch.stack([torch.sum(g, dim=0) for g in grouped_vecs])
        elif self.pool_method == 'max':
            return torch.stack([torch.max(g, dim=0)[0] for g in grouped_vecs])
        elif self.pool_method == 'softmax':
            return torch.stack([logsumexp(g, dim=0) for g in grouped_vecs])
        raise NotImplementedError

    def forward(self, word_idx_mat, lens, para_vecs, mentions,
                triple_candidates, pair_candidates):
        '''

        :param word_idx_mat: list of word indices, size(T, P)
        :param lens: list of paragraph lengths, size(P)
        :param para_vecs: list of paragraph vectors, size(P, pe)
        :param mentions: list of list of ParaMention
        :param triple_candidates: list of unlabeled Candidate
        :param pair_candidates: list of unlabeled Candidate
        :return:
        '''
        T, P = word_idx_mat.shape # T=num_toks, P=num_paras

        # Organize the candidate pairs and triples
        pair_to_idx = {}
        pair_sets = collections.defaultdict(set)
        for(t1, t2), cands in pair_candidates.items():
            pair_to_idx[(t1, t2)] = {c: i for i, c, in enumerate(cands)}
            for c in cands:
                pair_sets[(t1, t2)].add(c)
        triple_to_idx = {c: i for i, c in enumerate(triple_candidates)}

        # Build local embeddings of each word
        embs = self.embs(word_idx_mat) # T, P, e
        if self.use_position:
            para_embs = para_vecs.unsqueeze(0).expand(T, -1,-1) # T, P, pe
            embs = torch.cat([embs, para_embs], dim=2) # T, P, e + pe
        if self.use_lstm:
            lstm_in = rnn.pack_padded_sequence(embs, lens) # T, P, e + pe
            lstm_out_packed, _ = self.lstm(lstm_in)
            embs, _ = rnn.pad_packed_sequence(lstm_out_packed) # T, P, 2*h
        else:
            embs = self.emb_linear(embs) # T, P, 2*h

        # Gather co-occurring mention pairs and triples
        pair_inputs = {(t1, t2):[[] for i in range(len(cands))]
                       for(t1, t2), cands in pair_candidates.items()}
        triple_inputs = [[] for i in range(len(triple_candidates))]

        for para_idx, m_list in enumerate(mentions):
            typed_mentions = collections.defaultdict(list)
            for m in m_list:
                typed_mentions[m.type].append(m)
            for t1, t2 in ALL_ENTITY_TYPES_PAIRS:
                if self.pair_only and self.pair_only !=(t1 ,t2):
                    continue
                for m1 in typed_mentions[t1]:
                    for m2 in typed_mentions[t2]:
                        query_cand = Candidate(**{t1: m1.name, t2: m2.name})
                        if query_cand in pair_to_idx[(t1, t2)]:
                            idx = pair_to_idx[(t1, t2)][query_cand]
                            cur_vecs = torch.cat([embs[m1.start, para_idx, :],
                                                  embs[m2.start, para_idx, :]]) # 4*h
                            pair_inputs[(t1, t2)][idx].append(cur_vecs)
            if self.pair_only:
                continue
            for m1 in typed_mentions['drug']:
                for m2 in typed_mentions['gene']:
                    for m3 in typed_mentions['variant']:
                        query_cand = Candidate(m1.name, m2.name, m3.name)
                        if query_cand in triple_to_idx:
                            idx = triple_to_idx[query_cand]
                            cur_vecs = torch.cat(
                                                [embs[m1.start, para_idx, :],
                                                embs[m2.start, para_idx, :],
                                                embs[m3.start, para_idx, :]]) # 6*h
                            triple_inputs[idx].append(cur_vecs)

        # Compute local mention pair/triple representations
        pair_vecs = {}
        for t1, t2 in ALL_ENTITY_TYPES_PAIRS:
            if self.pair_only and self.pair_only != (t1, t2):
                continue
            cur_group_sizes = [len(vecs) for vecs in pair_inputs[(t1, t2)]]
            if sum(cur_group_sizes) > 0:
                cur_stack = torch.stack([
                    v for vecs in pair_inputs[(t1, t2)] for v in vecs]) # M, 4*h
                cur_m_reps = getattr(self, 'hidden_%s_%s' %
                                     (t1, t2))(cur_stack) # M, 2*h
                cur_pair_grouped_vecs = list(torch.split(cur_m_reps, cur_group_sizes))
                for i in range(len(cur_pair_grouped_vecs)):
                    if cur_pair_grouped_vecs[i].shape[0] == 0: # Back off
                        cur_pair_grouped_vecs[i] = getattr(self,
                                                           'backoff_%s_%s' % (t1, t2))
            else:
                cur_pair_grouped_vecs = [getattr(self, 'backoff_%s_%s' % (t1, t2))
                                         for vecs in pair_inputs[(t1, t2)]]
            pair_vecs[(t1, t2)] = torch.tanh(
                self.pool(cur_pair_grouped_vecs)) # P, 2*h

        if not self.pair_only:
            triple_group_sizes = [len(vecs) for vecs in triple_inputs]
            if sum(triple_group_sizes) > 0:
                triple_stack = torch.stack([
                    v for vecs in triple_inputs for v in vecs]) # M, 6*h
                triple_m_reps = self.hidden_triple(triple_stack) # M, 2*h
                triple_grouped_vecs = list(
                    torch.split(triple_m_reps, triple_group_sizes))
                for i in range(len(triple_grouped_vecs)):
                    if triple_grouped_vecs[i].shape[0] == 0: # back off
                        triple_grouped_vecs[i] = self.backoff_triple

            else:
                triple_grouped_vecs = [self.backoff_triple for vecs in triple_inputs]
            triple_vecs = torch.tanh(self.pool(triple_grouped_vecs)) # C, 2*h

        # Score candidate pairs
        pair_logits = {}
        for t1, t2 in ALL_ENTITY_TYPES_PAIRS:
            if self.pair_only and self.pair_only != (t1, t2):
                continue
            pair_logits[(t1, t2)] = getattr(self, 'out_%s_%s' % (t1, t2))(
                pair_vecs[(t1, t2)])[:, 0] #M
        if self.pair_only:
            return None, pair_logits

        # Score candidate triples
        pair_feats_per_triple = [[], [], []]
        for c in triple_candidates:
            for i in range(3):
                pair = c.remove_entity(i)
                t1, t2 = pair.get_types()
                pair_idx = pair_to_idx[(t1, t2)](pair)
                pair_feats_per_triple[i].append(
                    pair_vecs[(t1, t2)][pair_idx, :]) # 2*h
        triple_feats = torch.cat(
            [torch.stack(pair_feats_per_triple[0]),
             torch.stack(pair_feats_per_triple[1]),
             torch.stack(pair_feats_per_triple[2]),
             triple_vecs],
            dim=1) # C, 8*h
        final_hidden = F.relu(self.hidden_all(triple_feats)) # C, 2*h
        triple_logits = self.out_triple(final_hidden)[:, 0] # C
        return triple_logits, pair_logits


def get_entity_lists():
    entity_lists = {}
    for et in ALL_ENTITY_TYPES:
        entity_lists[et] = ['__%s__' % et
                        for i in range(MAX_ENTITIES_PER_TYPE)]
    # Can streamline, since we're just using single placeholder per entity type
    return entity_lists

def count_labels(name, data, preprocessor, pair_only=None):
    num_neg, num_pos = 0, 0
    for ex in data:
        cur_neg, cur_pos = preprocessor.count_labels(ex, pair_only=pair_only)
        num_neg += cur_neg
        num_pos += cur_pos
    log('%s data: +%d, -%d' % (name, num_pos, num_neg))
    return num_neg, num_pos

def print_data_stats(data, name):
    print(name)
    print('    Max num paragraphs: %d' % max(len(ex.paragraphs) for ex in data))
    print('    Max num triple candidates: %d' % max(
        len(ex.triple_candidates) for ex in data))

def init_word_vecs(device, vocab, all_zero=False):
    num_pretrained = 0
    embs = torch.zeros((len(vocab), EMB_SZIE), dtype=torch.float, device=device)
    if not all_zero:
        with open(os.path.join(settings.DATA_DIR,WORD_VEC_FILE)) as f:
            for line in tqdm(f, total=WORD_VEC_NUM_LINES):
                toks = line.strip().split(' ')
                if len(toks) != EMB_SZIE + 1:
                    continue
                word = toks[0]
                if word in vocab:
                    idx = vocab.get_index(word)
                    embs[idx, :] = torch.tensor([float(x) for x in toks[1:]],
                                                dtype=torch.float, device=device)
                    num_pretrained += 1
        log('Found pre-trained vectors for %d/%d = %.2f%% words' % (
            num_pretrained, len(vocab), 100*0 * num_pretrained /len(vocab)))

    return embs

def train(model, train_data, dev_data, preprocessor, num_epochs, lr, ckpt_iters,
          downsample_to, out_dir, lr_decay=1.0, pos_weight=None, use_pair_loss=True,
          pair_only=None):
    model.train()
    if ckpt_iters > len(train_data):
        ckpt_iters = len(train_data) # Checkpoint at least once per epoch
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    params = [p for p in model.paraments() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    train_data = list(train_data) # Copy before shuffling
    num_iters = 0
    best_ap = 0.0 # Choose checkpoint based on dev average precision
    train_loss = 0.0
    for t in range(num_epochs):
        t0 = time.time()
        random.shuffle(train_data)
        if not downsample_to:
            cur_train = tqdm(train_data)
        else:
            cur_train = train_data # tqdm is annoyingn on downsampled data
        for ex in cur_train:
            model.zero_grad()
            ex_torch = preprocessor.preprocess(ex, pair_only)
            triple_labels, pair_labels = ex_torch[-2:]
            triple_logits, pair_logits = model(*ex_torch[:-2])
            if pair_only:
                loss = loss_func(pair_logits[pair_only], pair_labels[pair_only])
            else:
                loss = loss_func(triple_logits, triple_labels)
                if use_pair_loss:
                    for t1, t2 in ALL_ENTITY_TYPES_PAIRS:
                        loss += loss_func(pair_logits[(t1, t2)], pair_labels[(t1, t2)])
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.paraments(),CLIP_THRESH)
            optimizer.step()
            num_iters += 1
            if num_iters % ckpt_iters == 0:
                model.eval()
                dev_preds, dev_loss = predict(
                    model, dev_data, preprocessor, loss_func=loss_func,
                    use_pair_loss=use_pair_loss, pair_only=pair_only)
                log('Iter %d: train loss = %.6f, dev loss = %.6f' % (
                    num_iters, train_loss / ckpt_iters, dev_loss))

                train_loss = 0.0
                p_doc, r_doc, f1_doc, ap_doc = evaluate(dev_data, dev_preds,
                                                        pair_only=pair_only)
                log('    Document-level : p=%.2f%% r=%.2f%% f1=%.2f%% ap=%.2f%%' % (
                    100 * p_doc, 100 * r_doc, 100 * f1_doc, 100 * ap_doc))
                if out_dir:
                    save_model(model, num_iters, out_dir)
                    model.train()
                    scheduler.step()
            t1 = time.time()
            log('Epoch %s: took %s' % (str(t).rjust(3), util.secs_to_str(t1 - t0)))

def predict(model, data, preprocessor, loss_func=None, use_pair_loss=True, pair_only=None):
    loss = 0.0
    preds = []
    with torch.no_grad():
        for ex in data:
            all_logits = []
            ex_torch = preprocessor.preprocess(ex, pair_only)
            triple_labels, pair_labels = ex_torch[-2:]
            triple_logits, pair_logits = model(*ex_torch[:-2])
            if loss_func:
                if pair_only:
                    loss += loss_func(pair_logits[pair_only], pair_labels[pair_only])
                else:
                    loss += loss_func(triple_logits, triple_labels)
                    if use_pair_loss:
                        for t1 ,t2 in ALL_ENTITY_TYPES_PAIRS:
                            loss += loss_func(pair_logits[(t1 ,t2)], pair_labels[(t1, t2)])
            if pair_only:
                cur_pred = [1 / (1 + np.exp(-z.item())) for z in pair_logits[pair_only]]
            else:
                cur_pred = [1 / (1 + np.exp(-z.item())) for z in pair_logits[pair_only]]
            preds.append(cur_pred)
        out = [preds]
        if loss_func:
            out.append(loss / len(data))
        if len(out) == 1:
            return out[0]
        return out


COLORS = {'drug': 'red', 'variant': 'cyan', 'gene': 'green'}

def pprint_example(ex, f=sys.stdout):
  print('PMID %s' % ex.pmid, file=f)
  for para_idx, (paragraph, m_list) in enumerate(zip(ex.paragraphs, ex.mentions)):
    word_to_type = {}
    for m in m_list:
      for i in range(m.start, m.end):
        word_to_type[i] = m.type
    para_toks = []
    for i in range(len(paragraph)):
      if i in word_to_type:
        para_toks.append(termcolor.colored(
            paragraph[i], COLORS[word_to_type[i]]))
      else:
        para_toks.append(paragraph[i])
    print('    Paragraph %d: %s' % (para_idx, ' '.join(para_toks)), file=f)

def evaluate(data, probs, name=None, threshold=0.5, pair_only=None):
  def get_candidates(ex):
    if pair_only:
      return ex.pair_candidates[pair_only]
    else:
      return ex.triple_candidates
  if name:
    log('== %s, document-level: %d documents, %d candidates (+%d, -%d) ==' % (
        name, len(data), sum(len(get_candidates(ex)) for ex in data),
        sum(1 for ex in data for c in get_candidates(ex) if c.label == 1),
        sum(1 for ex in data for c in get_candidates(ex) if c.label == 0)))
  tp = fp = fn = 0
  y_true = []
  y_pred = []
  for ex, prob_list in zip(data, probs):
    for c, prob in zip(get_candidates(ex), prob_list):
      y_true.append(c.label)
      y_pred.append(prob)
      pred = int(prob > threshold)
      if pred == 1:
        if c.label == 1:
          tp += 1
        else:
          fp += 1
      else:
        if c.label == 1:
          fn += 1
  ap = average_precision_score(y_true, y_pred)
  if name:
    log(util.get_prf(tp, fp, fn, get_str=True))
    log('AvgPrec  : %.2f%%' % (100.0 * ap))
  p, r, f = util.get_prf(tp, fp, fn)
  return p, r, f, ap


def predict_write(model, data, preprocessor, out_dir, ckpt, data_name, pair_only):
  if out_dir:
    if ckpt:
      out_path = os.path.join(out_dir, 'pred_%s_%07d.tsv' % (data_name, ckpt))
    else:
      out_path = os.path.join(out_dir, 'pred_%s.tsv' % data_name)
    # Only one pprint necessary
    pprint_out = os.path.join(out_dir, 'dev_pprint.txt')
  else:
    pprint_out = None
  pred = predict(model, tqdm(data), preprocessor, pair_only=pair_only)
  pprint_predictions(data, pred, preprocessor, fn=pprint_out)
  if out_path:
    write_predictions(data, pred, out_path, pair_only=pair_only)


def pprint_predictions(data, preds, preprocessor, threshold=0.5, fn=None):
  if fn:
    f = open(fn, 'w')
  else:
    f = sys.stdout
  for i, (ex, pred_list) in enumerate(zip(data, preds)):
    pprint_example(ex, f=f)
    new_paras, new_mentions = ex.paragraphs, ex.mentions
    for j, (c, pred) in enumerate(zip(ex.triple_candidates, pred_list)):
      pred_label = pred > threshold
      print('    (%s, %s, %s): pred=%s (p=%.4f), gold=%s, correct=%s' % (
          c.drug, c.gene, c.variant, pred_label, pred,
          c.label == 1, pred_label == (c.label == 1)), file=f)
    print('', file=f)
  if fn:
    f.close()

def write_predictions(data, preds, fn, pair_only=None):
  i = 0
  with open(fn, 'w') as f:
    for ex, pred_list in zip(data, preds):
      if pair_only:
        candidates = ex.pair_candidates[pair_only]
      else:
        candidates = ex.triple_candidates
      for c, pred in zip(candidates, pred_list):
        print('%d\t%s\t%s\t%s\t%s\t%.6f' % (
            i, ex.pmid, c.drug, c.gene, c.variant, pred), file=f)
        i += 1


def make_vocab(train_data, entity_lists, unk_thresh):
  vocab = vocabulary.Vocabulary(unk_threshold=unk_thresh)
  for ents in list(entity_lists.values()):
    for e in ents:
      vocab.add_word_hard(e)
  for ex in tqdm(train_data):
    for p, m_list in zip(ex.paragraphs, ex.mentions):
      in_mention = [False] * len(p)
      for m in m_list:
        for i in range(m.start, m.end):
          in_mention[i] = True
      for i, w in enumerate(p):
        if not in_mention[i]:
          vocab.add_word(w)
  return vocab


def save_model(model, num_iters, out_dir):
  fn = os.path.join(out_dir, 'model.%07d.pth' % num_iters)
  torch.save(model.state_dict(), fn)

def load_model(model, load_dir, device, load_ckpt):
  # if not load_ckpt:
  #   with open(os.path.join(load_dir, 'best_model.txt')) as f:
  #     load_ckpt = int(f.read().strip().split('\t')[0])
  fn = os.path.join(load_dir, 'model.%07d.pth' % load_ckpt)
  log('Loading model from %s' % fn)
  model.load_state_dict(torch.load(fn, map_location=device))

def predict_write(model, data, preprocessor, out_dir, ckpt, data_name, pair_only):
  if out_dir:
    if ckpt:
      out_path = os.path.join(out_dir, 'pred_%s_%07d.tsv' % (data_name, ckpt))
    else:
      out_path = os.path.join(out_dir, 'pred_%s.tsv' % data_name)
    # Only one pprint necessary
    pprint_out = os.path.join(out_dir, 'dev_pprint.txt')
  else:
    pprint_out = None
  pred = predict(model, tqdm(data), preprocessor, pair_only=pair_only)
  pprint_predictions(data, pred, preprocessor, fn=pprint_out)
  if out_path:
    write_predictions(data, pred, out_path, pair_only=pair_only)


def get_ds_train_dev_pmids(pmid_file):
  with open(os.path.join(settings.DATA_DIR, pmid_file)) as f:
    pmids = sorted([pmid.strip() for pmid in f if pmid.strip()])
  random.shuffle(pmids)
  num_train = int(round(len(pmids) * 0.7))
  num_train_dev = int(round(len(pmids) * 0.8))
  train_pmids = set(pmids[:num_train])
  dev_pmids = set(pmids[num_train:num_train_dev])
  return train_pmids, dev_pmids


def parse_args(args):
  parser = argparse.ArgumentParser()
  # Required params
  # parser.add_argument('para_file', help='JSON object storing paragraph text')
  # parser.add_argument('mention_file', help='List of mentions for relevant paragraphs')
  parser.add_argument('--ds-train-dev-file', help='Training examples')
  parser.add_argument('--jax-dev-test-file', help='Dev examples')
  parser.add_argument('--init-pmid-file', default='pmid_lists/init_pmid_list.txt',
                      help='Dev examples')

  # Model architecture
  parser.add_argument('--lstm-size', '-c', default=200,
                      type=int, help='LSTM hidden state size.')
  parser.add_argument('--lstm-layers', '-l', default=1,
                      type=int, help='LSTM number of layers.')
  parser.add_argument('--pool', '-p', choices=['softmax', 'max', 'mean', 'sum'], default='softmax',
                      help='How to pool across mentions')
  parser.add_argument('--no-position', action='store_true',
                      help='Ablate paragraph index encodings')
  parser.add_argument('--no-lstm', action='store_true', help='Ablate LSTM')
  # Training
  parser.add_argument('--num-epochs', '-T', type=int,
                      default=10, help='Training epochs')
  parser.add_argument('--learning-rate', '-r', type=float,
                      default=1e-5, help='Learning rate.')
  parser.add_argument('--dropout-prob', '-d', type=float,
                      default=0.5, help='Dropout probability')
  parser.add_argument('--lr-decay', '-g', type=float, default=1.0,
                      help='Decay learning rate by this much each epoch.')
  parser.add_argument('--balanced', '-b', action='store_true',
                      help='Upweight positive examples to balance dataset')
  parser.add_argument('--pos-weight', type=float, default=None,
                      help='Upweight postiive examples by this much')
  parser.add_argument('--use-pair-loss', action='store_true',
                      help="Multi-task on pair objective")
  # Data
  #parser.add_argument('--data-cache', default=DEFAULT_CACHE)
  parser.add_argument('--data-cache', default=None)
  parser.add_argument('--rng-seed', default=0, type=int, help='RNG seed')
  parser.add_argument('--torch-seed', default=0,
                      type=int, help='torch RNG seed')
  parser.add_argument('--downsample-to', default=None, type=int,
                      help='Downsample to this many examples per split')
  parser.add_argument('--unk-thresh', '-u', default=5, type=int,
                      help='Treat words with fewer than this many counts as <UNK>.')
  parser.add_argument('--print-dev', action='store_true',
                      help='Test on dev data')
  parser.add_argument('--jax', action='store_true', help='Test on JAX data')
  parser.add_argument('--jax-out', default='pred_jax.tsv')
  parser.add_argument('--text-level', choices=['document', 'paragraph', 'sentence'],
                      default='document', help='Split documents paragraph-wise or sentence-wise')
  parser.add_argument('--pair-only', default=None,
                      help='Comma-separated pair of entities to focus on only')
  # CPU vs. GPU
  parser.add_argument('--cpu-only', action='store_true',
                      help='Run on CPU only')
  parser.add_argument('--gpu-id', type=int, default=0,
                      help='GPU ID (default=0)')
  # Saving and loading
  parser.add_argument('--out-dir', '-o', default=None,
                      help='Where to write all output')
  parser.add_argument('--ckpt-iters', '-i', default=10000, type=int,
                      help='Checkpoint after this many training steps.')
  parser.add_argument(
      '--load', '-L', help='Directory to load model parameters and vocabulary')
  parser.add_argument('--load-ckpt', type=int, default=None,
                      help='Which checkpoint to use (default: use best_model.txt)')
  parser.add_argument('--try-all-checkpoints', action='store_true',
                      help='Make predictions for every checkpoint')
  parser.add_argument('--data-dir', help='root data directory')
  # Other
  parser.add_argument('--no-w2v', action='store_true',
                      help='No pre-trained word vectors')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args(args)


def get_all_checkpoints(out_dir):
  fns = glob.glob(os.path.join(out_dir, 'model.*.pth'))
  return sorted([int(os.path.basename(x).split('.')[1]) for x in fns])

def run(OPTS, device):
  # Process pair-only mode
  pair_only = None
  if OPTS.pair_only:
    pair_only = tuple(OPTS.pair_only.split(','))
    if pair_only not in ALL_ENTITY_TYPES_PAIRS:
      raise ValueError('Bad value for pair_only: %s' % OPTS.pair_only)
  entity_lists = get_entity_lists()
  # Read data
  train_pmids_set, dev_ds_pmids_set = get_ds_train_dev_pmids(
      OPTS.init_pmid_file)
  ds_train_dev_data = Example.read_examples(OPTS.ds_train_dev_file)
  # Filter out examples that doesn't contain pair or triple candidates
  if pair_only:
    ds_train_dev_data = [x for x in ds_train_dev_data if pair_only in
                         x.pair_candidates and x.pair_candidates[pair_only]]
  else:
    ds_train_dev_data = [x for x in ds_train_dev_data if x.triple_candidates]
  train_data = [x for x in ds_train_dev_data if x.pmid in train_pmids_set]
  dev_ds_data = [x for x in ds_train_dev_data if x.pmid in dev_ds_pmids_set]
  random.shuffle(train_data)
  random.shuffle(dev_ds_data)

  jax_dev_test_data = Example.read_examples(OPTS.jax_dev_test_file)
  if pair_only:
    jax_dev_test_data = [x for x in jax_dev_test_data if pair_only in
                         x.pair_candidates and x.pair_candidates[pair_only]]
  else:
    jax_dev_test_data = [x for x in jax_dev_test_data if x.triple_candidates]
  random.shuffle(jax_dev_test_data)

  with open(os.path.join(settings.DATA_DIR, JAX_DEV_PMIDS_FILE)) as f:
    dev_jax_pmids_set = set(x.strip() for x in f if x.strip())
  with open(os.path.join(settings.DATA_DIR, JAX_TEST_PMIDS_FILE)) as f:
    test_pmids_set = set(x.strip() for x in f if x.strip())

  dev_jax_data = [x for x in jax_dev_test_data if x.pmid in dev_jax_pmids_set]
  test_data = [x for x in jax_dev_test_data if x.pmid in test_pmids_set]
  log('Read %d train, %d dev dist sup, %d dev jax, %d test examples' %
      (len(train_data), len(dev_ds_data), len(dev_jax_data), len(test_data)))

  vocab = make_vocab(train_data, entity_lists, OPTS.unk_thresh)
  log('Vocab size = %d.' % len(vocab))
  preprocessor = Preprocessor(entity_lists, vocab, device)
  num_neg, num_pos = count_labels('train', train_data, preprocessor,
                                  pair_only=pair_only)
  word_vecs = init_word_vecs(device, vocab, all_zero=OPTS.load or OPTS.no_w2v)
  log('Finished reading data.')

  # Run model
  model = BackoffModel(
      word_vecs, OPTS.lstm_size, OPTS.lstm_layers, device,
      use_lstm=not OPTS.no_lstm, use_position=not OPTS.no_position,
      pool_method=OPTS.pool, dropout_prob=OPTS.dropout_prob,
      vocab=vocab, pair_only=pair_only).to(device=device)
  if OPTS.load:
    load_model(model, OPTS.load, device, OPTS.load_ckpt)
  if OPTS.num_epochs > 0:
    log('Starting training.')
    pos_weight = None
    if OPTS.balanced:
      pos_weight = torch.tensor(float(num_neg) / num_pos, device=device)
    elif OPTS.pos_weight:
      pos_weight = torch.tensor(OPTS.pos_weight, device=device)
    train(model, train_data, dev_ds_data, preprocessor, OPTS.num_epochs,
          OPTS.learning_rate, OPTS.ckpt_iters, OPTS.downsample_to, OPTS.out_dir,
          pos_weight=pos_weight, lr_decay=OPTS.lr_decay,
          use_pair_loss=OPTS.use_pair_loss, pair_only=pair_only)
    log('Finished training.')
  model.eval()
  if OPTS.try_all_checkpoints:
    ckpts = get_all_checkpoints(OPTS.out_dir)
  else:
    ckpts = [None]
  for ckpt in ckpts:
    if ckpt:
      print('== Checkpoint %s == ' % ckpt, file=sys.stderr)
      load_model(model, OPTS.out_dir, device, ckpt)
    predict_write(model, dev_jax_data, preprocessor,
                  OPTS.out_dir, ckpt, 'dev', pair_only)
    predict_write(model, test_data, preprocessor,
                  OPTS.out_dir, ckpt, 'test', pair_only)


def main(OPTS):
  if OPTS.out_dir:
    if os.path.exists(OPTS.out_dir):
      shutil.rmtree(OPTS.out_dir)
    os.makedirs(OPTS.out_dir)
    global log_file
    log_file = open(os.path.join(OPTS.out_dir, 'log.txt'), 'w')
  log(OPTS)
  random.seed(OPTS.rng_seed)
  torch.manual_seed(OPTS.torch_seed)
  if OPTS.cpu_only:
    device = torch.device('cpu')
  else:
    device = torch.device('cuda:%d' % OPTS.gpu_id)
  try:
    run(OPTS, device)
  finally:
    if log_file:
      log_file.close()


if __name__ == '__main__':
  OPTS = parse_args(sys.argv[1:])
  main(OPTS)
