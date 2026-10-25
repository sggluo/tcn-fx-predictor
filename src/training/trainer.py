import torch
import torch.nn as nn 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import polars as pl
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from src.models.TCN import TemporalConvNet, LightningTCN
from src.data.processor import sliding_window
from src.configs.configs import BASE_DIR, DATA_DIR, DATA_DIR_RAW, DATA_DIR_PROCESSED, DATA_DIR_FEATURES_TARGETS

def trainer(model, X_train, y_train, X_val, y_val, criterion, optimizer_, lr, epochs, bs=32):
    lightning_model = LightningTCN(model=model, lr=lr, criterion=criterion, optimizer_=optimizer_)
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
        )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4)

    early_stop = EarlyStopping(monitor='val loss', mode='min', patience=10)
    checkpoint_callback = ModelCheckpoint(
        dirpath=BASE_DIR/'checkpoints',
        monitor = 'val loss',
        mode = 'min',
        save_top_k =1,
        filename='best_model',
        verbose=True

    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=50,
        callbacks=[early_stop, checkpoint_callback]
    )


    trainer.fit(lightning_model, train_loader, val_loader)
    return trainer, lightning_model

def objective(trial,sql):
    data = pl.scan_parquet(DATA_DIR_FEATURES_TARGETS/'features_targets').sort('timestamp')
    target_cols = [f'd{n}' for n in [10,30,60,180]]
    X = data.select(pl.all().exclude(target_cols+['timestamp'])).collect().to_numpy()
    y = data.select(target_cols).collect().to_numpy()[:,2]

    cutoff2 = int( y.shape[0] *0.8 )
    X = X[:cutoff2]
    y = y[:cutoff2]

    optimizer_ = torch.optim.Adam
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    num_chanels = trial.suggest_categorical('num_channels',([20,20,20], [40,80,160], [10,10,5], [50,100,500]))
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    kernel_size = trial.suggest_int('kernel_size', 3, 5)
    fold_losses = []
    num_folds = 4
    length = int( y.shape[0] * 0.2 )

    for i in range(num_folds):
        X_train = X[:(i+1)*length]
        y_train = y[:(i+1)*length]
    
        if i != num_folds-1:
            X_val = X[(i+1)*length:(i+2)*length][180:]
            y_val = y[(i+1)*length:(i+2)*length][180:] 
        else:
            X_val = X[(i+1)*length:][180:]
            y_val = y[(i+1)*length:][180:]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_train = sliding_window(X_train, sql).transpose(0,2,1)
        y_train = y_train[sql-1:]
        X_val = sliding_window(X_val, sql).transpose(0,2,1)
        y_val = y_val[sql-1:]
        class_counts = np.bincount(y_train, minlength=3) 
        weights = len(y_train) / (3 * class_counts)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        num_inputs=X_train.shape[1]
        train_ds = TensorDataset(
             torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
         )

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        tcn_model = TemporalConvNet(num_inputs=num_inputs, num_channels=num_chanels, kernel_size=kernel_size, dropout=dropout, num_outputs=3)
        lightning_model = LightningTCN(model=tcn_model, lr=lr, criterion=criterion, optimizer_=optimizer_)
        early_stop = EarlyStopping(
             monitor="val loss",
             patience=5,          
              min_delta=1e-4,       
              mode="min",
              verbose=False,
                 )


        trainer = L.Trainer(
            max_epochs=50,
            accelerator='auto',
            devices=1,
            callbacks=[early_stop],
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(lightning_model, train_loader, val_loader)
        fold_loss = trainer.callback_metrics['val loss'].item()
        fold_losses.append(fold_loss)
    avg = sum(fold_losses) / len(fold_losses)
    print(f"[Trial {trial.number}] avg val loss = {avg:.4f} | params={trial.params}")
   
    return avg  