import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

import numpy as np
import argparse, time, os
from math import factorial


# hyperparameters
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5 # 20 # 200 # 100 # 50 # 200 # 10
BATCH_SIZE = 150
PATIENCE = 10

def n_unroot(Ntaxa):
    """
    Replicates the function:
        N = factorial(2*Ntaxa - 5) / (factorial(Ntaxa - 3) * 2^(Ntaxa - 3))
    Returns an integer.
    """
    N = factorial(2 * Ntaxa - 5) // (factorial(Ntaxa - 3) * (2 ** (Ntaxa - 3)))
    return int(N)

# Read FASTA convert to numeric
def fasta_pars(aln_file, seq_number, Lmax):
    aln=open(aln_file)
    dic={'A':'0','T':'1','C':'2','G':'3','-':'4'}
    matrix_out=[]
    fasta_dic={}
    for line in aln:
        if line[0]==">":
            header=line[1:].rstrip('\n').strip()
            fasta_dic[header]=[]
        elif line[0].isalpha() or line[0]=='-':
            for base, num in dic.items():
                line=line[:].rstrip('\n').strip().replace(base,num)
            line=list(line)
            line=[int(n) for n in line]
            #Mkae all matrices of equal length for CNN +[-15]*(Lmax-len(line)) 
            fasta_dic[header]+=line+[-15]*(Lmax-len(line)) 
            if len(fasta_dic)==seq_number:
                taxa_block=[]
                for taxa in sorted(list(fasta_dic.keys())):
                    taxa_block.append(fasta_dic[taxa.strip()])
                fasta_dic={}
                matrix_out.append(taxa_block)
    return np.array(matrix_out)

# Read training, validation and test datasets to equalize sizes 
def tv_parse(train, valid, test, seq_number=4):
    tr=open(train)
    va=open(valid)
    te=open(test)
    LT=max([len(r.strip()) for r in tr])
    print("Training largest alignment: "+str(LT))
    LV=max([len(r.strip()) for r in va])
    print("Validation largest alignment: "+str(LV))
    LTE=max([len(r.strip()) for r in te])
    print("Testing largest alignment: "+str(LTE))
    Lmax=max([LT]+[LV]+[LTE])
    tr=fasta_pars(train,seq_number,Lmax)
    va=fasta_pars(valid,seq_number,Lmax)
    te=fasta_pars(test,seq_number,Lmax)
    return tr, va, te

def build_labels(num_taxa, data_array):
    """
    Build one-hot labels: We assume data_array is grouped so that
    each of the n_unroot(Ntaxa) topologies appears equally often.
    
    For example, if data_array.shape[0] = total_samples,
    and total_samples = k * Nlabels, then the labels are repeated
    topology indices from 0..Nlabels-1, each repeated k times.
    """
    Nlabels = n_unroot(num_taxa)
    total_samples = data_array.shape[0]
    repeats = total_samples // Nlabels

    # Each topology index is repeated 'repeats' times
    label_indices = np.repeat(np.arange(Nlabels), repeats)

    # Create one-hot label
    one_hot = np.eye(Nlabels)[label_indices]
    return one_hot

# ----------------------
# PyTorch Model
# ----------------------

class StandardCNN(nn.Module):
    """
    Translate the Keras CNN architecture into PyTorch.
    In Keras, the input shape is (Ntaxa, Aln_length, 1).
    PyTorch uses (batch, channels, height, width).
    So your input will be shaped as (batch_size, 1, Ntaxa, Aln_length) 
    when feeding into this network.
    
    The CNN structure:
    - A sequence of conv -> BN -> Dropout -> AvgPool layers,
      repeated 'conv_pool_n' times (8 in your Keras code).
    - Then Flatten, Dense(1024), Dropout, Dense(Nlabels).
    """
    def __init__(self, Ntaxa, Aln_length, conv_pool_n=8):
        super(StandardCNN, self).__init__()

        # hardcoded hyperparams
        self.conv_x = [4,1,1,1,1,1,1,1]
        self.conv_y = [1,2,2,2,2,2,2,2]
        self.pool   = [1,4,4,4,2,2,2,1]
        self.filter_s = [1024,1024,128,128,128,128,128,128]
        
        # for the classification layer
        self.Nlabels = n_unroot(Ntaxa)
        
        # build convolutional layers
        # each iteration does: ZeroPadding2D + Conv2D + BatchNorm + Dropout + AvgPool
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        in_channels = 1
        
        for l in range(conv_pool_n):
            # ZeroPadding2D equivalent: we only pad along the width dimension (Aln_length dimension)
            # PyTorch: nn.ZeroPad2d(padding=(left,right,top,bottom)).
            # keras code (from paper): ZeroPadding2D(((0,0),(0, conv_y[l]-1))).
            # That’s “top=0, bottom=0, left=0, right=conv_y[l]-1”.
            pad_left   = 0
            pad_right  = self.conv_y[l] - 1 if self.conv_y[l] > 1 else 0
            pad_top    = 0
            pad_bottom = 0
            padding2d  = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))
            
            conv2d = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.filter_s[l],
                kernel_size=(self.conv_x[l], self.conv_y[l]),
                stride=1
            )
            bn = nn.BatchNorm2d(self.filter_s[l])
            dropout = nn.Dropout(p=0.2)
            # Average Pool
            pool2d = nn.AvgPool2d(kernel_size=(1, self.pool[l]))  # pool along the width dimension

            block = nn.Sequential(
                padding2d,
                conv2d,
                bn,
                nn.ReLU(),
                dropout,
                pool2d
            )
            
            self.convs.append(block)
            in_channels = self.filter_s[l]
        
        # After these conv blocks, we Flatten and go to a Dense layer of 1024, dropout, then final Dense.
        self.fc1 = nn.Linear(self._get_flatten_size(BATCH_SIZE, Ntaxa, Aln_length), 1024)
        self.dropout_fc = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(1024, self.Nlabels)
        
    def _get_flatten_size(self, batch_size, Ntaxa, Aln_length):
        """
        Helper to figure out the final flattened size after the stacked conv/pool layers.
        We'll run a dummy forward pass on a single example to compute the shape.
        """
        # Make a dummy input of shape (1, 1, Ntaxa, Aln_length).
        # Then run forward through the conv blocks only.
        # dummy = torch.zeros(batch_size, 1, Ntaxa, Aln_length)
        # dummy = torch.zeros(1, 1, Ntaxa, Aln_length)
        # x = dummy
        # for block in self.convs:
        #     x = block(x)
        # # Flatten
        # flatten_dim = x.numel()
        # return flatten_dim

        dummy = torch.zeros(1, 1, Ntaxa, Aln_length)
        self.eval() # temporarily switch to eval mode to avoid batch normalization function complaining about batch size = 1
        with torch.no_grad():
            for block in self.convs:
                dummy = block(dummy)
        self.train() # switch back to training mode
        flatten_dim = dummy.numel()
        return flatten_dim

    
    def forward(self, x):
        """
        Forward pass. x should be shape (batch_size, 1, Ntaxa, Aln_length)
        """
        # print(f"initial x.shape: {x.shape}")

        for block in self.convs:
            x = block(x)

        # print(f"convolved x.shape: {x.shape}")
        
        x = x.view(x.size(0), -1)  # flatten
        # print(f"flattened x.shape: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)  # logits
        return x


# ----------------------
# Training / Evaluation
# ----------------------

def train_model(model, train_loader, valid_loader, device, 
                epochs=NUM_EPOCHS, patience=PATIENCE, lr=LEARNING_RATE, save_path=f"/home/rl659/4775-final-project/saved_models/full_model/{BATCH_SIZE}_{LEARNING_RATE:.2e}_{NUM_EPOCHS}.pt"):
    """
    A simple training loop with early stopping based on validation loss.
    """
    print(f"epochs: {epochs}")
    print(f"patience: {patience}")
    print(f"learning rate: {lr}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in valid_loader:
                X_val = X_val.to(device, dtype=torch.float32)
                y_val = y_val.to(device, dtype=torch.long)

                outputs_val = model(X_val)
                loss_val = criterion(outputs_val, y_val)
                val_loss += loss_val.item() * X_val.size(0)
        val_loss /= len(valid_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")
        
        # early stopping logic
        if val_loss < best_val_loss - 1e-3:
            best_val_loss = val_loss
            patience_counter = 0
            # save best model
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    # load the best weights
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

# def evaluate_model(model, test_loader, device):
#     """
#     Evaluate the model on a test set.
#     Returns (test_loss, test_accuracy, class_probs, predicted_labels).
#     """
#     criterion = nn.CrossEntropyLoss()
#     model.eval()
    
#     test_loss = 0.0
#     correct = 0
#     class_probs_list = []
#     all_labels = []
    
#     with torch.no_grad():
#         for X_test, y_test in test_loader:
#             X_test = X_test.to(device, dtype=torch.float32)
#             y_test = y_test.to(device, dtype=torch.long)

#             outputs = model(X_test)  # logits
#             loss = criterion(outputs, y_test)
#             test_loss += loss.item() * X_test.size(0)

#             probs = F.softmax(outputs, dim=1)
#             class_probs_list.append(probs.cpu().numpy())
            
#             _, preds = torch.max(outputs, dim=1)
#             correct += torch.sum(preds == y_test).item()
            
#             all_labels.append(preds.cpu().numpy())

#     test_loss /= len(test_loader.dataset)
#     accuracy = correct / len(test_loader.dataset)

#     # concatenate all probabilities and predicted labels
#     class_probs = np.concatenate(class_probs_list, axis=0)
#     predicted_labels = np.concatenate(all_labels, axis=0)
    
#     return test_loss, accuracy, class_probs, predicted_labels

def evaluate_model(model, test_loader, device, num_classes, output_curve_path='/home/rl659/4775-final-project/figures/micro_pr_curve.png'):
    """
    Evaluate the model on a test set.
    Returns (test_loss, test_accuracy, class_probs, predicted_labels).
    Additionally, this function will create a micro-averaged precision-recall curve and save it to a file.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    test_loss = 0.0
    correct = 0
    class_probs_list = []
    all_predicted_labels = []
    all_true_labels = []
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device, dtype=torch.float32)
            y_test = y_test.to(device, dtype=torch.long)

            outputs = model(X_test)  # logits
            loss = criterion(outputs, y_test)
            test_loss += loss.item() * X_test.size(0)

            probs = F.softmax(outputs, dim=1)
            class_probs_list.append(probs.cpu().numpy())
            
            _, preds = torch.max(outputs, dim=1)
            correct += torch.sum(preds == y_test).item()
            
            all_predicted_labels.append(preds.cpu().numpy())
            all_true_labels.append(y_test.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    # concatenate all probabilities, predicted labels, and true labels
    class_probs = np.concatenate(class_probs_list, axis=0)
    predicted_labels = np.concatenate(all_predicted_labels, axis=0)
    true_labels = np.concatenate(all_true_labels, axis=0)
    
    # compute micro-averaged precision-recall curve
    # binarize the true labels for all classes
    true_labels_binarized = label_binarize(true_labels, classes=np.arange(num_classes))
    
    # flatten both the true labels and predicted probabilities for micro-averaging
    y_true_all = true_labels_binarized.ravel()
    y_score_all = class_probs.ravel()
    
    precision, recall, thresholds = precision_recall_curve(y_true_all, y_score_all)
    avg_precision = average_precision_score(y_true_all, y_score_all)
    
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', label='Micro-Averaged (average precision = {:.2f})'.format(avg_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Micro-Averaged Precision-Recall Curve (200 epochs, early stopping at 53/200 epochs)')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(output_curve_path)
    plt.close()
    
    return test_loss, accuracy, class_probs, predicted_labels

def main():
    parser = argparse.ArgumentParser(description='PyTorch run')
    parser.add_argument('--train', dest='TRAIN', help="Training dataset in .npy")
    parser.add_argument('--valid', dest='VALID', help="Validation dataset in .npy")
    parser.add_argument('--test', dest='TEST', help="Test dataset in .npy")
    parser.add_argument('--convert_dataset', dest="convert_dataset", type=bool, help="flag to convert raw FASTA into npy")
    parser.add_argument('-N', dest='Ntaxa', type=int, help="Number of taxa")
    # parser.add_argument('--batch_size', default=100, type=int, help="Batch size (default 100)")
    args = parser.parse_args()

    print(args.convert_dataset)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # parse/load data
    train_data, valid_data, test_data = None, None, None
    if args.convert_dataset:
        print("Parsing raw FASTA files and saving outputs as npy files")
        train_data, valid_data, test_data = tv_parse("/home/rl659/4775-final-project/data/gap50k_500/TRAIN.aln", "/home/rl659/4775-final-project/data/gap50k_500/VALID.aln", "/home/rl659/4775-final-project/data/gap50k_500/TEST.aln")
        np.save("/home/rl659/4775-final-project/data/gap50k_500/TRAIN.npy", train_data)
        np.save("/home/rl659/4775-final-project/data/gap50k_500/VALID.npy", valid_data)
        np.save("/home/rl659/4775-final-project/data/gap50k_500/TEST.npy", test_data)
        print("Done parsing data and saving output")
    else:
        print("Reading input .npy arrays")
        train_data = np.load(args.TRAIN)   # shape: (num_samples, Ntaxa, Aln_length, 1) in Keras code
        valid_data = np.load(args.VALID)
        test_data  = np.load(args.TEST)
        print("Done reading data")
    
    # generate labels (one-hot in Keras). In PyTorch, we'll store them as class indices.
    train_label_onehot = build_labels(args.Ntaxa, train_data)
    valid_label_onehot = build_labels(args.Ntaxa, valid_data)
    test_label_onehot  = build_labels(args.Ntaxa, test_data)
    
    # convert one-hot to integer class labels
    train_label_idx = np.argmax(train_label_onehot, axis=1)
    valid_label_idx = np.argmax(valid_label_onehot, axis=1)
    test_label_idx  = np.argmax(test_label_onehot, axis=1)
    
    # In original code from paper (Keras), the data is shape (N, Ntaxa, Aln_length, 1).
    # PyTorch expects (N, channels, H, W). We can reshape to (N, 1, Ntaxa, Aln_length).
    def reshape_for_torch(arr):
        # shape is (N, Ntaxa, Aln_length)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=-1)  # (N, Ntaxa, Aln_length, 1)

        arr = np.transpose(arr, (0, 3, 1, 2))   # (N, 1, Ntaxa, Aln_length)
        return arr

    print(train_data.shape)
    print(valid_data.shape)
    print(test_data.shape)

    train_data_torch = reshape_for_torch(train_data)
    valid_data_torch = reshape_for_torch(valid_data)
    test_data_torch  = reshape_for_torch(test_data)

    print(train_data_torch.shape)
    
    # ----------------------------
    # Create PyTorch Datasets & Loaders
    # ----------------------------
    
    train_ds = TensorDataset(torch.from_numpy(train_data_torch),
                             torch.from_numpy(train_label_idx))
    valid_ds = TensorDataset(torch.from_numpy(valid_data_torch),
                             torch.from_numpy(valid_label_idx))
    test_ds  = TensorDataset(torch.from_numpy(test_data_torch),
                             torch.from_numpy(test_label_idx))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # build and train model
    # assume the shape is (N, 1, Ntaxa, Aln_length)
    Ntaxa = train_data_torch.shape[2]
    Aln_length = train_data_torch.shape[3]
    
    model_cnn = StandardCNN(Ntaxa, Aln_length, conv_pool_n=8).to(device)
    print(model_cnn)

    start_time = time.time()
    train_model(model_cnn, train_loader, valid_loader, device)
    end_time = time.time()
    print(f"{(end_time - start_time):.1f} sec for training")
    
    # evaluate best model on test set
    # test_loss, test_acc, class_probs, predicted_labels = evaluate_model(model_cnn, test_loader, device)

    test_loss, test_acc, class_probs, predicted_labels = evaluate_model(model_cnn, test_loader, device, num_classes=model_cnn.Nlabels)

    print("Evaluate with best class weights")
    print(f"Test Loss: {test_loss:.4f}   Test Accuracy: {test_acc:.4f}")
    
    # save results
    np.savetxt("test.evals_class.txt", np.array([test_loss, test_acc]), fmt='%f')
    np.savetxt("test.classprobs_class.txt", class_probs, fmt='%f')
    np.savetxt("test.classeslab_class.txt", predicted_labels, fmt='%d')

if __name__ == "__main__":
    # main()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = StandardCNN(Ntaxa=4, Aln_length=890, conv_pool_n=8).to(device)
    model.load_state_dict(torch.load("/home/rl659/4775-final-project/saved_models/full_model/150_1e-3_200.pt"))

    test_data  = np.load("/home/rl659/4775-final-project/data/gap50k_500/TEST.npy")
    test_label_onehot  = build_labels(num_taxa=4, data_array=test_data)
    test_label_idx  = np.argmax(test_label_onehot, axis=1)

    def reshape_for_torch(arr):
        # shape is (N, Ntaxa, Aln_length)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=-1)  # (N, Ntaxa, Aln_length, 1)

        arr = np.transpose(arr, (0, 3, 1, 2))   # (N, 1, Ntaxa, Aln_length)
        return arr

    test_data_torch  = reshape_for_torch(test_data)

    test_ds  = TensorDataset(torch.from_numpy(test_data_torch),
                             torch.from_numpy(test_label_idx))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    
    test_loss, test_acc, class_probs, predicted_labels = evaluate_model(model, test_loader, device, num_classes=model.Nlabels)