from typing import Dict, List
import itertools
import warnings
import torch
import pickle as pickle

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

################################################################
# Global CSTLM object shared between multiprocessing processes #
################################################################
cstlm = None

### We need to monkey patch `TextField.as_tensor` to allow FloatTensors.
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
                "count2vec_indices" : torch.LongTensor(padded_array["count2vec_indices"]),
                "count2vec_values" : torch.FloatTensor(padded_array["count2vec_values"])
            }
        else:
            indexer_tensors = {
                key: torch.LongTensor(array) for key, array in padded_array.items()
            }
        tensors.update(indexer_tensors)
    return tensors

### Do the monkey patching
TextField.as_tensor = as_tensor_monkey_patch


@TokenIndexer.register("count2vec")
class Count2VecIndexer(TokenIndexer[Dict[str, List]]):
    """
    This :class:`TokenIndexer` represents tokens as sparse row of ppmi values.

    `tokens_to_indices` return two keys: "count2vec_indices" and "count2vec_values" reprenting the
    column indices and values in the ppmi row.
    """
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

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        return ["{}_indices".format(index_name), "{}_values".format(index_name)]

    @overrides
    def get_padding_lengths(self, token: List) -> Dict[str, int]:
        return {"non_zeros_per_row": len(token)}

    @overrides
    def get_padding_token(self) -> List:
        return []

    @overrides
    def get_token_min_padding_length(self) -> int:
        return 0

    @overrides
    def pad_token_sequence(
        self, 
        tokens: Dict[str, List[List]], 
        desired_num_tokens: Dict[str, int], 
        padding_lengths: Dict[str, int],
    ) -> Dict[str, List[List]]:
        padded_tokens = {}
        for key, token_list in tokens.items():
            assert key in self.get_keys("count2vec")
            # Pad the token list with empty lists
            token_list = pad_sequence_to_length(
                token_list, desired_num_tokens[key], default_value=self.get_padding_token
            )
            # Pad the lists with zeros
            for i, token in enumerate(token_list):
                token_list[i] = pad_sequence_to_length(token, padding_lengths["non_zeros_per_row"])
            padded_tokens[key] = token_list
        return padded_tokens

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary, index_name: str
    ) -> Dict[str, List[List]]:
        indices, values = [[] for _ in range(len(tokens))], [[] for _ in range(len(tokens))]

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
            indices[i].extend(cstlm.word2id(word) for word in ppmi_next.keys())
            values[i].extend(ppmi_next.values())

        pattern = deque()
        for i, token in reversed(list(enumerate(tokens))):
            pattern.appendleft(normalize(token.text))
            ppmi_prev = cstlm.ppmi_prev(pattern)
            indices[i].extend(cstlm.size() + cstlm.word2id(word) for word in ppmi_prev.keys())
            values[i].extend(ppmi_next.values())

        return {
            "{}_indices".format(index_name): indices,
            "{}_values".format(index_name): values,
        }
