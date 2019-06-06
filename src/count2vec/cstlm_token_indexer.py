from typing import Dict, List
import itertools
import warnings
import torch
import pickle as pkl

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.fields import TextField

import logging as log
from pprint import pprint

from collections import deque
import traceback
from .BiCSTLM import BiCSTLM

import scipy.sparse
from sklearn.random_projection import GaussianRandomProjection
import numpy as np


cstlm = None
linear_projection = None


class SparseVector:
    def __init__(self, indices: List[int], values: List[float]) -> None:
        self.indices: List[int] = indices
        self.values: List[float] = values
    
    # def to_sparse(self):
    #     return scipy.sparse.coo_matrix(
    #         (self.values, ([0 for _ in self.indices], self.indices)), shape=(1, cstlm.size())
    #     )
    
    # def to_dense(self):
    #     return linear_projection.transform(self.to_sparse()).squeeze()


class Count2VecToken:
    def __init__(self, ppmi_next: SparseVector, ppmi_prev: SparseVector) -> None:
        self.ppmi_next: SparseVector = ppmi_next
        self.ppmi_prev: SparseVector = ppmi_prev

    def to_array(self):
        values = []
        values.extend(self.ppmi_next.values)
        values.extend(self.ppmi_prev.values)

        columns = []
        columns.extend(i for i in self.ppmi_next.indices)
        columns.extend(i + cstlm.size() for i in self.ppmi_prev.indices)
        
        rows = [0 for _ in columns]

        sparse = scipy.sparse.coo_matrix((values, (rows, columns)), shape=(1, 2 * cstlm.size()))

        return linear_projection.transform(sparse).squeeze()


def as_tensor_monkey_patch(self, padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
    tensors = {}
    for indexer_name, indexer in self._token_indexers.items():
        desired_num_tokens = {
            indexed_tokens_key: padding_lengths[f"{indexed_tokens_key}_length"]
            for indexed_tokens_key in self._indexer_name_to_indexed_token[indexer_name]
        }
        indices_to_pad = {
            indexed_tokens_key: self._indexed_tokens[indexed_tokens_key]
            for indexed_tokens_key in self._indexer_name_to_indexed_token[indexer_name]
        }
        padded_array = indexer.pad_token_sequence(
            indices_to_pad,
            desired_num_tokens, 
            padding_lengths,
        )
        if indexer_name == "count2vec":
            indexer_tensors = {
                key: torch.Tensor(array) for key, array in padded_array.items()
            }
        else:
            indexer_tensors = {
                key: torch.LongTensor(array) for key, array in padded_array.items()
            }
        tensors.update(indexer_tensors)
    return tensors


TextField.as_tensor = as_tensor_monkey_patch


@TokenIndexer.register("count2vec")
class Count2VecIndexer(TokenIndexer[Dict[str, np.array]]):
    def __init__(
        self,
        namespace: str,
        collection_dir: str,
        reversed_collection_dir: str,
        embedding_dim: int,
        token_min_padding_length: int = 0,
        projection_weights_file: str = None,
    ) -> None:
        self._token_min_padding_length = token_min_padding_length
        self._namespace = namespace
        self._embedding_dim = embedding_dim

        global cstlm
        cstlm = BiCSTLM(collection_dir, reversed_collection_dir)

        global linear_projection
        if projection_weights_file is not None:
            linear_projection = pkl.load(projection_weights_file)
        else:
            linear_projection = (
                GaussianRandomProjection(2 * embedding_dim).fit(np.empty((1, 2 * cstlm.size())))
            )

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        return [index_name]

    @overrides
    def get_padding_lengths(self, token: np.array) -> Dict[str, int]:
        return {}

    @overrides
    def get_padding_token(self) -> np.array:
        return np.zeros(2 * self._embedding_dim)

    @overrides
    def get_token_min_padding_length(self) -> int:
        return 0

    @overrides
    def pad_token_sequence(
        self, 
        tokens: Dict[str, List[np.array]], 
        desired_num_tokens: Dict[str, int], 
        padding_lengths: Dict[str, int]
    ) -> Dict[str, List[np.array]]:
        for key, value in tokens.items():
            while len(value) < desired_num_tokens[key]:
                value.append(self.get_padding_token())
            while len(value) > desired_num_tokens[key]:
                value.pop()
        return tokens

    @overrides
    def tokens_to_indices(
        self,
        tokens: List[Token],
        vocabulary: Vocabulary,
        index_name: str,
    ) -> Dict[str, List[np.array]]:
        ppmi_next_indices = []
        ppmi_next_values = []
        ppmi_prev_indices = []
        ppmi_prev_values = []

        def normalize(word):
            if word == "<SOS>":
                return "<S>"
            if word == "<EOS>":
                return "</S>"
            return word

        pattern = deque()
        for i, token in enumerate(tokens):
            pattern.append(normalize(token.text))
            ppmi_next = cstlm.ppmi_next(pattern)
            ppmi_next_indices.append(list(cstlm.word2id(word) for word in ppmi_next.keys()))
            ppmi_next_values.append(list(ppmi_next.values()))

        pattern = deque()
        for i, token in reversed(list(enumerate(tokens))):
            pattern.appendleft(normalize(token.text))
            ppmi_prev = cstlm.ppmi_prev(pattern)
            ppmi_prev_indices.append(list(cstlm.word2id(word) for word in ppmi_prev.keys()))
            ppmi_prev_values.append(list(ppmi_prev.values()))

        ret = [
            Count2VecToken(
                SparseVector(ppmi_next_indices[i], ppmi_next_values[i]),
                SparseVector(ppmi_prev_indices[i], ppmi_prev_values[i]),
            ).to_array()
            for i in range(len(tokens))
        ]

        return {index_name: ret}
