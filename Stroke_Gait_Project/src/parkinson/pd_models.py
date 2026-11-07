import torch
import torch.nn as nn



class PDCNN1D(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # [batch, 180, 12] -> [batch, 12, 180]
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)



class PDLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=12,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        x = hidden[-1]
        x = self.dropout(x)
        return self.fc(x)


class PDCNNLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.cnn(x)

        x = x.permute(0, 2, 1)

        output, (hidden, cell) = self.lstm(x)

        x = self.dropout(hidden[-1])
        return self.fc(x)



if __name__ == "__main__":

    test_input = torch.randn(32, 180, 12)

    models = {
        "1D CNN": PDCNN1D(),
        "LSTM": PDLSTM(),
        "CNN-LSTM": PDCNNLSTM()
    }

    print("PD MODEL ARCHITECTURE TEST")
    print("-" * 45)

    for name, model in models.items():
        model.eval()

        with torch.no_grad():
            output = model(test_input)

        print(f"{name}: {output.shape}")

        assert output.shape == (32, 2)
