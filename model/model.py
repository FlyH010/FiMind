from transformers import PretrainedConfig


class FiMindConfig(PretrainedConfig):
    model_type = "fimind"
    def __init__(
        self,
        dropout: float = 0.0,  

    ):
        pass