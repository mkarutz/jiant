from typing import Dict

import torch
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Elmo
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from overrides import overrides

from pprint import pprint
from .BiCSTLM import BiCSTLM
from collections import deque
import numpy as np

@TokenEmbedder.register("count2vec")
class Count2VecTokenEmbedder(TokenEmbedder):
    """
    Count2Vec Embedder.
    """

    def __init__(self, embedding_dim: int) -> None:
        super(Count2VecTokenEmbedder, self).__init__()
        self._embedding_dim = 2 * embedding_dim

    def get_output_dim(self):
        return self._embedding_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.is_cuda, "Inputs are not on CUDA device."
        return inputs

    @classmethod
    def from_params(cls, params: Params):
        raise "Not implemented"
