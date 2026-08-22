import torch
import torch.nn as nn

class CNN_Transformer_BiGRU(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN_Transformer_BiGRU, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.bigru = nn.GRU(input_size=32, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(32 * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        out, _ = self.bigru(x)
        out = torch.mean(out, dim=1)
        logits = self.fc(out)
        return logits