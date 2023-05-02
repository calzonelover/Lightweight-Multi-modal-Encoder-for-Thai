from transformers import PretrainedConfig
from typing import List


class ThaiLightWeightEncoderConfig(PretrainedConfig):

    def __init__(
        self,
        input_embedding_dim: int = 300,
        final_embedding_dim: int = 512,
        dropout: float = 0.2,
        word_vector_model_name: str = "thai2fit_wv",
        **kwargs,
    ):
        self.input_embedding_dim = input_embedding_dim
        self.final_embedding_dim = final_embedding_dim
        self.word_vector_model_name = word_vector_model_name
        self.dropout = dropout
        super().__init__(**kwargs)