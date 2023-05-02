from transformers import PreTrainedModel

from pythainlp import word_vector
import torch

from .configuration import ThaiLightWeightEncoderConfig
from .projector import Projector


class ThaiLightWeightEncoderModel(PreTrainedModel):
    config_class = ThaiLightWeightEncoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.wv = word_vector.WordVector(model_name=config.word_vector_model_name)
        self.projector = Projector(
            input_embedding_dim=config.input_embedding_dim,
            final_embedding_dim=config.final_embedding_dim,
            dropout=config.dropout
        )
    
    def forward(self, text: str):
        embed = self.wv.sentence_vectorizer(text, use_mean=True)[0]
        proj_embed = self.projector(torch.from_numpy(embed).float())
        proj_embed = proj_embed.to("cpu").detach().numpy()
        return proj_embed