import torch
import torch.nn as nn

class DisasterBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, max_len=50):
        super(DisasterBiLSTM, self).__init__()
        
        # Layer 2: Embedding: Vocab=10,000, dim=128 [cite: 32]
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Layer 3: BiLSTM-1: 128 units/direction (256 total), return_sequences=True [cite: 33]
        self.bilstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        self.dropout_lstm1 = nn.Dropout(0.2) # [cite: 33]

        # Layer 4: BiLSTM-2: 64 units/direction (128 total), return_sequences=False [cite: 34]
        # Input size is 256 because previous layer was Bi-Directional (128*2)
        self.bilstm2 = nn.LSTM(
            input_size=256,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )
        
        # Layer 5 & 6: Dense-1 64 units + ReLU + Dropout 0.5 [cite: 35, 37]
        # Input is 128 (64*2 from BiLSTM2)
        self.fc1 = nn.Linear(128, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        # Layer 7 & 8: Dense-2 32 units + ReLU + Dropout 0.3 [cite: 39, 41]
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        # Layer 9: Output 1 unit, Sigmoid [cite: 42]
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, max_len]
        x = self.embedding(x)
        
        # BiLSTM 1
        x, _ = self.bilstm1(x) 
        x = self.dropout_lstm1(x)
        
        # BiLSTM 2 (We only need the final hidden state if return_sequences=False, 
        # but in PyTorch we usually take the last output for the sequence)
        _, (hidden, _) = self.bilstm2(x)
        
        # Concatenate the final forward and backward hidden states
        # hidden shape: [2, batch, 64] -> [batch, 128]
        cat_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        x = self.fc1(cat_hidden)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x