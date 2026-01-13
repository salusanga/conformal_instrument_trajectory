import torch
import torch.nn as nn

class ConformalQuantileRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_quantiles=2, num_tests=2):
        super(ConformalQuantileRegressor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_quantiles = num_quantiles
        self.num_tests = num_tests

        # Shared base layers
        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.num_tests * self.num_quantiles),
        )

    def forward(self, x):
        preds = self.shared_layers(x)
        return preds.view(
            -1, self.num_tests, self.num_quantiles
        )  # Shape: (batch, num_tests, num quantiles)
