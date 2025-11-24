import torch
import torch.nn as nn

class DisasterBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, max_len=50):
        super(DisasterBiLSTM, self).__init__()
        
        # Layer 2: Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # TUNED ARCHITECTURE 
        # Reduced units to prevent overfitting
        
        # BiLSTM-1: Reduced to 64 units (was 128)
        self.bilstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=64, 
            batch_first=True,
            bidirectional=True
        )
        # Increased Dropout to 0.3
        self.dropout_lstm1 = nn.Dropout(0.3)

        # BiLSTM-2: Reduced to 32 units (was 64)
        # Input is 128 because previous layer was 64*2
        self.bilstm2 = nn.LSTM(
            input_size=128,
            hidden_size=32,
            batch_first=True,
            bidirectional=True
        )
        
        # Dense Layers
        # Input is 64 (32*2 from BiLSTM2)
        self.fc1 = nn.Linear(64, 32) # Reduced from 64
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(32, 16) # Reduced from 32
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4) # Increased dropout
        
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        
        x, _ = self.bilstm1(x) 
        x = self.dropout_lstm1(x)
        
        _, (hidden, _) = self.bilstm2(x)
        
        # Concatenate hidden states
        # hidden shape: [2, batch, 32] -> [batch, 64]
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