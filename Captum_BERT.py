import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig, PreTrainedTokenizerBase

from captum.attr import visualization as vis
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients, InterpretableEmbeddingBase
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

# from transformers import AutoTokenizer, AutoModelForMaskedLM
# tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
# model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')

# print(torch.__version__) # hx_pc -> 1.6.0 + cu101
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#
# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path: str = 'saved_models/'
# load model
model: nn.Module = BertForQuestionAnswering.from_pretrained(model_path)
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer: PreTrainedTokenizerBase = BertTokenizer.from_pretrained(model_path)

def predict(inputs: list, token_type_ids: list=None, position_ids: list=None, attention_mask: any=None) -> nn.Module:
    return model(inputs, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)

def squad_pos_forward_func(inputs:list, token_type_ids:list=None, position_ids:list=None, attention_mask: any=None,\
                           position:int=0) -> torch.Tensor:
    pred: torch.Tensor = predict(inputs,
                   token_type_ids=token_type_ids,
                   position_ids=position_ids,
                   attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values

# Optional[int]
ref_token_id = tokenizer.pad_token_id

# Optional[int]
sep_token_id = tokenizer.sep_token_id

# Optional[int]
cls_token_id = tokenizer.cls_token_id

interpretable_embedding:  InterpretableEmbeddingBase = configure_interpretable_embedding_layer(model, \
                                                                                              'bert.embeddings')
interpretable_embedding1: InterpretableEmbeddingBase = configure_interpretable_embedding_layer(model, \
                                                                            'bert.embeddings.word_embeddings')
interpretable_embedding2: InterpretableEmbeddingBase = configure_interpretable_embedding_layer(model, \
                                                                            'bert.embeddings.token_type_embeddings')
interpretable_embedding3: InterpretableEmbeddingBase = configure_interpretable_embedding_layer(model, \
                                                                            'bert.embeddings.position_embeddings')




def construct_input_ref_pair(question: str, text: str, ref_token_id: int | str, sep_token_id: int | str, \
                             cls_token_id: int | str) \
                                 -> (torch.Tensor, torch.Tensor, int):
    question_ids: list = tokenizer.encode(question, add_special_tokens=False)
    text_ids: list = tokenizer.encode(text, add_special_tokens=False)

    input_ids: list = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]

    ref_input_ids: list = [cls_token_id] + [ref_token_id] + len(question_ids) + [sep_token_id] + \
        [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)

def construct_input_ref_token_type_pair(input_ids: torch.Tensor, sep_ind:int = 0) -> (torch.Tensor, torch.Tensor):
    seq_len: int = input_ids.size(1)
    token_type_ids: torch.Tensor = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids: torch.Tensor = torch.zeros_like(token_type_ids, device=device) # * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    seq_length: int = input_ids.size(1)
    position_ids: torch.Tensor = torch.arange(seq_length, dtype=torch.long, device=device)
    ref_position_ids: torch.Tensor = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)

    return position_ids, ref_position_ids

def construct_attention_mask(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(input_ids)

def construct_bert_sub_embedding(input_ids: any, ref_input_ids: any, \
                                 torken_type_ids: any, ref_token_type_ids: any, \
                                 position_ids: any, ref_position_ids: any) \
        -> ((torch.Tensor, torch.Tensor),(torch.Tensor, torch.Tensor),(torch.Tensor, torch.Tensor)):
    input_embeddings: torch.Tensor = interpretable_embedding1.indices_to_embeddings(input_ids)
    ref_input_embeddings: torch.Tensor = interpretable_embedding1.indices_to_embeddings(ref_input_ids)

    input_embeddings_token_type: torch.Tensor = interpretable_embedding2.indices_to_embeddings(torken_type_ids)
    ref_input_embeddings_token_type: torch.Tensor = interpretable_embedding2.indices_to_embeddings(ref_token_type_ids)

    input_embeddings_position_ids: torch.Tensor = interpretable_embedding3.indices_to_embeddings(position_ids)
    ref_input_embeddings_position_ids: torch.Tensor = interpretable_embedding3.indices_to_embeddings(ref_position_ids)
    return (input_embeddings, ref_input_embeddings), (input_embeddings_token_type, ref_input_embeddings_token_type), \
           (input_embeddings_position_ids, ref_input_embeddings_position_ids)

def construct_whole_bert_embeddings(input_ids: any, ref_input_ids: any, \
                                    token_type_ids: any=None, ref_token_type_ids: any=None, \
                                    position_ids: any=None, ref_position_ids:any=None)\
                                    -> (torch.Tensor, torch.Tensor):
    input_embeddings: torch.Tensor = interpretable_embedding.indices_to_embeddings(input_ids, \
                                                                                       token_type_ids=token_type_ids, \
                                                                                       position_ids=position_ids)
    ref_input_embeddings: torch.Tensor = interpretable_embedding.indices_to_embeddings(ref_input_ids, \
                                                                                       token_type_ids=token_type_ids,\
                                                                                       position_ids=position_ids)
    return input_embeddings, ref_input_embeddings

question, text = "What is important to us?", "It is important to us to include, empower and support humans of all kinds."

input_ids, ref_input_ids, sep_id = construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id)
token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
attention_mask = construct_attention_mask(input_ids)

indices = input_ids[0].detach().tolist()
all_tokens = tokenizer.convert_ids_to_tokens(indices)

ground_truth = 'to include, empower and support humans of all kinds'

ground_truth_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)
ground_truth_end_ind = indices.index(ground_truth_tokens[-1])
ground_truth_start_ind = ground_truth_end_ind - len(ground_truth_tokens) + 1

start_scores, end_scores = predict(input_ids, \
                                   token_type_ids=token_type_ids, \
                                   position_ids=position_ids, \
                                   attention_mask=attention_mask)


print('Question: ', question)
print('Predicted Answer: ', ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))



