import torch
from torch import nn

class Regressor(nn.Module):
    def __init__(self, base_model, n_features, neurons_reducer_block=256, p_dropout=0.5):
        super(Regressor, self).__init__()
        
        self.feature_extractor = base_model 

        if neurons_reducer_block > 0:
            self.regressor_head = nn.Sequential(
                nn.Linear(n_features, neurons_reducer_block),
                nn.BatchNorm1d(neurons_reducer_block),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),
                nn.Linear(neurons_reducer_block, 1)
            )
        else:
            self.regressor_head = nn.Linear(n_features, 1)

    def forward(self, img):

        x = self.feature_extractor(img)
        
        x = self.regressor_head(x)
        
        return x