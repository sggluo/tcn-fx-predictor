import numpy as np
import torch
from src.configs.configs import BASE_DIR, DATA_DIR, DATA_DIR_RAW, DATA_DIR_PROCESSED, DATA_DIR_FEATURES_TARGETS
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

def evaluation(lr, ckpt_path):
    tcn_model = TemporalConvNet(num_inputs=num_inputs, num_channels=num_chanels, kernel_size=4, dropout=0.2587, num_outputs=3)
    lightning_model = LightningTCN(model=tcn_model, lr=9.880820256268033e-05, criterion=None, optimizer_=None)

    checkpoint = torch.load(ckpt_path, map_location='cuda')
    state_dict = checkpoint['state_dict']
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('criterion.')}
    lightning_model.load_state_dict(state_dict)
    y_test = y_test[:,2]
    lightning_model.eval()
    with torch.no_grad():
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.from_numpy(X_test).float()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_test = X_test.to(device)
        lightning_model.to(device)
        y_hat = lightning_model(X_test)  

    preds = torch.argmax(y_hat, axis=1).to('cpu' if device=='cuda' else 'cpu')
    preds = preds.numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()

    test_acc = np.sum( (y_test==preds) ) / len( y_test )
    print(f'Accuracy of test set: {test_acc}') 
    print(classification_report(y_test, preds, digits=4))

    p, r, f1, support = precision_recall_fscore_support(
    y_test, preds, labels=[0, 1, 2], average=None
)
    print("Each class's precision:", p)
    print("Each cl recall:   ", r)
    print("各类样本数:    ", support)

    p_macro, r_macro, _, _ = precision_recall_fscore_support(
    y_test, preds, average='macro'
)
print("macro P/R:", p_macro, r_macro)
   


