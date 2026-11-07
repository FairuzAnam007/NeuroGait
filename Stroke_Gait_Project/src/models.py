import torch
import torch.nn as nn

class CNN1D(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=12,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            128,
            2
        )


    def forward(self, x):


        x = x.permute(
            0,
            2,
            1
        )


        x = self.features(x)


        x = x.squeeze(-1)



        x = self.dropout(x)

        x = self.classifier(x)


        return x


class LSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=12,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            64,
            2
        )


    def forward(self, x):


        output, (hidden, cell) = self.lstm(x)


        x = hidden[-1]



        x = self.dropout(x)

        x = self.classifier(x)


        return x


class CNNLSTM(nn.Module):

    def __init__(self):

        super().__init__()


        self.cnn = nn.Sequential(

            nn.Conv1d(
                in_channels=12,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            )
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )


        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            64,
            2
        )


    def forward(self, x):


        x = x.permute(
            0,
            2,
            1
        )


        x = self.cnn(x)

        x = x.permute(
            0,
            2,
            1
        )


        output, (hidden, cell) = self.lstm(x)


        x = hidden[-1]



        x = self.dropout(x)

        x = self.classifier(x)


        return x


if __name__ == "__main__":

    dummy_input = torch.randn(
        32,
        520,
        12
    )

    print("MODEL TEST")
    print("------------------------------")

    print("Input shape:")
    print(dummy_input.shape)


    cnn_model = CNN1D()

    cnn_output = cnn_model(
        dummy_input
    )

    print("\n1D CNN Output shape:")
    print(cnn_output.shape)

    lstm_model = LSTMModel()

    lstm_output = lstm_model(
        dummy_input
    )

    print("\nLSTM Output shape:")
    print(lstm_output.shape)


    cnn_lstm_model = CNNLSTM()

    cnn_lstm_output = cnn_lstm_model(
        dummy_input
    )

    print("\nCNN-LSTM Output shape:")
    print(cnn_lstm_output.shape)
