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

LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
BATCH_SIZE = 150
PATIENCE = 10

# Change HOME_PATH as needed
HOME_PATH = '/'
PROJECT_PATH = os.path.join(HOME_PATH, "final")
DATA_PATH = os.path.join(PROJECT_PATH, "data")
GAPPED_PATH = os.path.join(DATA_PATH, "gap50k_500")
UNGAPPED_PATH = os.path.join(DATA_PATH, "nogap50k_500")
SAVED_MODEL_PATH = os.path.join(PROJECT_PATH, "saved_models/full_model")
FIGURE_PATH = os.path.join(PROJECT_PATH, "figures")

# Usage Example:
'''
 python model.py --convert_dataset 1 \
 --gapped 1 \
 --train $GAPPED_PATH/TRAIN.npy \
 --valid $GAPPED_PATH/VALID.npy \
 --test $GAPPED_PATH/TEST.npy \
 -N 4
'''

def n_unroot(Ntaxa):
    """
    Replicates the function:
        N = factorial(2*Ntaxa - 5) / (factorial(Ntaxa - 3) * 2^(Ntaxa - 3))
    Returns an integer.
    """
    N = factorial(2 * Ntaxa - 5) // (factorial(Ntaxa - 3) * (2 ** (Ntaxa - 3)))
    return int(N)

def fasta_pars(aln_file, seq_number, Lmax):
    aln = open(aln_file)
    dic = {'A': '0', 'T': '1', 'C': '2','G': '3', '-': '4'}
    matrix_out = []
    fasta_dic = {}
    for line in aln:
        if line.startswith(">"):
            header = line[1:].rstrip('\n').strip()
            fasta_dic[header] = []
        elif line[0].isalpha() or line[0] == '-':
            for base, num in dic.items():
                line = line.rstrip('\n').strip().replace(base, num)
            line = list(line)
            line = [int(n) for n in line]
            fasta_dic[header] += line + [-15]*(Lmax - len(line)) 
            if len(fasta_dic) == seq_number:
                taxa_block = []
                for taxa in sorted(fasta_dic.keys()):
                    taxa_block.append(fasta_dic[taxa.strip()])
                fasta_dic = {}
                matrix_out.append(taxa_block)
    return np.array(matrix_out)

def tv_parse(train, valid, test, seq_number=4):
    tr = open(train)
    va = open(valid)
    te = open(test)
    LT = max([len(r.strip()) for r in tr])
    print("Training largest alignment: " + str(LT))
    LV = max([len(r.strip()) for r in va])
    print("Validation largest alignment: " + str(LV))
    LTE = max([len(r.strip()) for r in te])
    print("Testing largest alignment: " + str(LTE))
    Lmax = max([LT] + [LV] + [LTE])
    tr = fasta_pars(train, seq_number, Lmax)
    va = fasta_pars(valid, seq_number, Lmax)
    te = fasta_pars(test, seq_number, Lmax)
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

    label_indices = np.repeat(np.arange(Nlabels), repeats)

    one_hot = np.eye(Nlabels)[label_indices]
    return one_hot


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, conv_x, conv_y, dropout_p=0.2):
        super(DenseLayer, self).__init__()
        # Asymmetric padding using nn.ZeroPad2d
        pad_left = 0
        pad_right = conv_y - 1
        pad_top = 0
        pad_bottom = 0
        self.pad = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels,
            growth_rate,
            kernel_size=(conv_x, conv_y),
            stride=1,
            padding=0
        )
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, x):
        x = self.pad(x)
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.dropout(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, conv_x, conv_y_list, dropout_p=0.2):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for conv_y in conv_y_list[:num_layers]:
            layer = DenseLayer(in_channels, growth_rate, conv_x, conv_y, dropout_p)
            self.layers.append(layer)
            in_channels += growth_rate
        self.out_channels = in_channels
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)

class DenseNetCNN(nn.Module):
    """
    DenseNet-like CNN architecture with dense connections between layers.
    """
    def __init__(self, Ntaxa, Aln_length, gapped, conv_pool_n=7, growth_rate=32):
        super(DenseNetCNN, self).__init__()

        self.conv_x = [4, 1, 1, 1, 1, 1, 1, 1]
        self.conv_y = [1, 2, 2, 2, 2, 2, 2, 1] if gapped else [1, 2, 2, 2, 2, 2, 2, 7]
        self.pool = [1, 4, 4, 4, 2, 2, 2, 1] if gapped else [1, 2, 2, 2, 2, 2, 2, 7]
        self.filter_s = [1024, 1024, 128, 128, 128, 128, 128, 128]

        self.Nlabels = n_unroot(Ntaxa)
        self.initial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=growth_rate,
            kernel_size=(self.conv_x[0], self.conv_y[0]),
            stride=1,
            padding=(0, self.conv_y[0] - 1)
        )
        self.initial_bn = nn.BatchNorm2d(growth_rate)
        self.initial_relu = nn.ReLU(inplace=True)
        self.initial_dropout = nn.Dropout(p=0.2)
        self.initial_pool = nn.AvgPool2d(kernel_size=(1, self.pool[0]))

        conv_y_list = self.conv_y[1:conv_pool_n+1]

        self.dense_block = DenseBlock(
            num_layers=conv_pool_n,
            in_channels=growth_rate,
            growth_rate=growth_rate,
            conv_x=1,
            conv_y_list=conv_y_list,
            dropout_p=0.2
        )

        self.final_bn = nn.BatchNorm2d(self.dense_block.out_channels)
        self.flatten_size = self._get_flatten_size(Ntaxa, Aln_length)

        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.dropout_fc = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(1024, self.Nlabels)

    def _get_flatten_size(self, Ntaxa, Aln_length):
        """
        Helper to figure out the final flattened size after the stacked conv/pool layers.
        We'll run a dummy forward pass on a single example to compute the shape.
        """
        dummy = torch.zeros(1, 1, Ntaxa, Aln_length)
        self.eval()
        with torch.no_grad():
            x = self.initial_conv(dummy)
            x = self.initial_bn(x)
            x = self.initial_relu(x)
            x = self.initial_dropout(x)
            x = self.initial_pool(x)
            x = self.dense_block(x)
            x = self.final_bn(x)
            x = F.relu(x, inplace=True)
        self.train()
        flatten_dim = x.numel()
        return flatten_dim

    def forward(self, x):
        """
        Forward pass. x should be shape (batch_size, 1, Ntaxa, Aln_length)
        """
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        x = self.initial_dropout(x)
        x = self.initial_pool(x)

        x = self.dense_block(x)
        x = self.final_bn(x)
        x = F.relu(x, inplace=True)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, valid_loader, device, save_path, 
                epochs=NUM_EPOCHS, patience=PATIENCE, lr=LEARNING_RATE):
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
        
        if val_loss < best_val_loss - 1e-3:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

def evaluate_model(model, test_loader, device, num_epochs, data_type, num_classes, output_curve_path):
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

    class_probs = np.concatenate(class_probs_list, axis=0)
    predicted_labels = np.concatenate(all_predicted_labels, axis=0)
    true_labels = np.concatenate(all_true_labels, axis=0)
    
    true_labels_binarized = label_binarize(true_labels, classes=np.arange(num_classes))
    
    y_true_all = true_labels_binarized.ravel()
    y_score_all = class_probs.ravel()
    
    precision, recall, thresholds = precision_recall_curve(y_true_all, y_score_all)
    avg_precision = average_precision_score(y_true_all, y_score_all)
    
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', label='Micro-Averaged (AP = {:.2f})'.format(avg_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Micro-Averaged Precision-Recall Curve ({data_type}, {num_epochs} epochs)')
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
    parser.add_argument('--convert_dataset', dest='convert_dataset', type=int, help="flag to convert raw FASTA into npy", default=0)
    parser.add_argument('--gapped', dest='gapped', type=int, help="flag to determine whether to used gapped or ungapped dataset", required=True)
    parser.add_argument('-N', dest='Ntaxa', type=int, help="Number of taxa", required=True)
    args = parser.parse_args()

    print(f"Convert Dataset Flag: {args.convert_dataset}")
    print(f"Gapped Flag: {args.gapped}")

    data_path = GAPPED_PATH if args.gapped else UNGAPPED_PATH

    print(f"Data path: {data_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_data, valid_data, test_data = None, None, None
    if args.convert_dataset:
        print("Parsing raw FASTA files and saving outputs as .npy files")
        train_data, valid_data, test_data = tv_parse(
            os.path.join(data_path, "TRAIN.aln"), 
            os.path.join(data_path, "VALID.aln"), 
            os.path.join(data_path, "TEST.aln")
        )
        np.save(os.path.join(data_path, "TRAIN.npy"), train_data)
        np.save(os.path.join(data_path, "VALID.npy"), valid_data)
        np.save(os.path.join(data_path, "TEST.npy"), test_data)
        print("Done parsing data and saving output")
    else:
        print("Reading input .npy arrays")
        train_data = np.load(args.TRAIN)
        valid_data = np.load(args.VALID)
        test_data  = np.load(args.TEST)
        print("Done reading data")
    
    train_label_onehot = build_labels(args.Ntaxa, train_data)
    valid_label_onehot = build_labels(args.Ntaxa, valid_data)
    test_label_onehot  = build_labels(args.Ntaxa, test_data)
    
    train_label_idx = np.argmax(train_label_onehot, axis=1)
    valid_label_idx = np.argmax(valid_label_onehot, axis=1)
    test_label_idx  = np.argmax(test_label_onehot, axis=1)
    
    def reshape_for_torch(arr):
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=-1)

        arr = np.transpose(arr, (0, 3, 1, 2))
        return arr

    print(f"Train data shape: {train_data.shape}")
    print(f"Valid data shape: {valid_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    train_data_torch = reshape_for_torch(train_data)
    valid_data_torch = reshape_for_torch(valid_data)
    test_data_torch  = reshape_for_torch(test_data)

    print(f"Train data reshaped for PyTorch: {train_data_torch.shape}")
    
    train_ds = TensorDataset(torch.from_numpy(train_data_torch).float(),
                             torch.from_numpy(train_label_idx).long())
    valid_ds = TensorDataset(torch.from_numpy(valid_data_torch).float(),
                             torch.from_numpy(valid_label_idx).long())
    test_ds  = TensorDataset(torch.from_numpy(test_data_torch).float(),
                             torch.from_numpy(test_label_idx).long())
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    Ntaxa = train_data_torch.shape[2]
    Aln_length = train_data_torch.shape[3]
    
    conv_pool_n = 7

    model_cnn = DenseNetCNN(Ntaxa, Aln_length, args.gapped, conv_pool_n=conv_pool_n, growth_rate=32).to(device)
    print("Model Architecture:")
    print(model_cnn)

    data_type = "gap" if args.gapped else "nogap"

    saved_model_file = os.path.join(SAVED_MODEL_PATH, f"{data_type}_{BATCH_SIZE}_{LEARNING_RATE:.2e}_{NUM_EPOCHS}.pt")

    start_time = time.time()
    train_model(model_cnn, train_loader, valid_loader, device, saved_model_file)
    end_time = time.time()
    print(f"{(end_time - start_time):.1f} sec for training")
    
    pr_curve_file = os.path.join(FIGURE_PATH, f"{data_type}_micro_pr_curve_{NUM_EPOCHS}.png")

    test_loss, test_acc, class_probs, predicted_labels = evaluate_model(
        model_cnn, test_loader, device, 
        num_epochs=NUM_EPOCHS, 
        data_type=data_type, 
        num_classes=model_cnn.Nlabels, 
        output_curve_path=pr_curve_file
    )

    print("Evaluation with best class weights:")
    print(f"Test Loss: {test_loss:.4f}   Test Accuracy: {test_acc:.4f}")
    
    np.savetxt("test.evals_class.txt", np.array([test_loss, test_acc]), fmt='%f')
    np.savetxt("test.classprobs_class.txt", class_probs, fmt='%f')
    np.savetxt("test.classeslab_class.txt", predicted_labels, fmt='%d')

if __name__ == "__main__":
    main()