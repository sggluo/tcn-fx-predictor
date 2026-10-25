import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import lightning as L


# TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    # exclude some data from -chomp_size to the end
    # in order to prohibit information leakage
    # padding is required since sequence length cannot change
    # but padding works on both left and right sides
    # e.g. when k=3, padding_left = padding_right=1, 
    # at t=0, receptive field covers t=-1,0,1, which incurs leakage
    def forward(self, x):
        # contiguous() keep the memory contiguous
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        # n_inputs: num of features, n_outputs: output channels
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        #self.conv1.weight.data.normal_(0, 0.01)
        self.conv1 = weight_norm(self.conv1)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        #self.conv2.weight.data.normal_(0, 0.01)
        self.conv2 = weight_norm(self.conv2)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        #if self.downsample is not None:
        #    self.downsample.weight.data.normal_(0, 0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, num_outputs=3):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels): # how many layers
            # i=0,1,2...
            dilation = 2**i
            # num_inputs: dim of inputs
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(in_channels, out_channels, kernel_size=kernel_size,
                            stride=1, dilation=dilation,
                            padding=(kernel_size-1)*dilation, dropout=dropout)
            # temporal block = 2*( chomp1d + conv1d + relu + dropout)
            ]
        # *layers: unpack layers
        self.networks = nn.Sequential(*layers)
        # input dim: num_channels[-1]
        self.fc = nn.Linear(num_channels[-1], num_outputs)

    def forward(self, x):
        x = self.networks(x)
        # (bs, num_channels, sequence length) -> (bs, num_channels)
        x = x[:, :, -1]  
        x = self.fc(x) 
        return x 

class LightningTCN(L.LightningModule):
    def __init__(self, model, lr, criterion, optimizer_):
        super().__init__()
        self.model = model
        self.lr =lr
        self.criterion = criterion
        self.optimizer_ = optimizer_
    
    def forward(self,x):
        return self.model(x)

    
    def training_step(self, batch, batch_idx):
        x,y=batch # unpack
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, axis=1)
        acc = (y==preds).float().mean()

        self.log('train loss', loss, prog_bar=True, on_epoch=True)
        self.log('train accuracy', acc, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y=batch # unpack
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, axis=1)
        acc = (y==preds).float().mean()

        self.log('val loss', loss, prog_bar=True, on_epoch=True)
        self.log('val accuracy', acc, prog_bar=True, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = self.optimizer_(self.model.parameters(), lr=self.lr)
        return optimizer
        