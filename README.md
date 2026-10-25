This project predicts the direction of the USD/EUR exchange rate (up vs. down) using a Temporal Convolutional Network (TCN) implemented in PyTorch. 
Historical exchange-rate sequences from Dukascopy Bank are fed into stacked temporal blocks with dilated causal convolutions, Chomp1d to prevent 
look-ahead leakage, weight normalization, and residual connections; the final hidden state is classified into up/flat/down moves. 
Training is managed by PyTorch Lightning with configurable loss/optimizer and per-epoch logging of loss and directional accuracy.

