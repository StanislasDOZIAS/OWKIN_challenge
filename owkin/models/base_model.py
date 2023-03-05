from torch import nn


class BaseModel(nn.Module):
    """
    Base Model with all the strict minimum: meta_type, type, config and name
    """

    def __init__(self):
        super().__init__()
        self.meta_type: str
        self.type: str
        self.config: dict()
        self.name: str

    def compute_name(self):
        name = f"{self.type}/"
        for k, v in self.config.items():
            name += f"{k}_{v}_"
        self.name = name[:-1]
