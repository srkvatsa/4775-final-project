import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder

# first attempt at replicating
# Define CNN Model in PyTorch
class PhyloCNN(nn.Module):
    def __init__(self, Ntaxa, Aln_length, Nlabels, conv_pool_n=8, dropout_rate=0.2):
        super(PhyloCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        conv_x = [4, 1, 1, 1, 1, 1, 1, 1]
        conv_y = [1, 2, 2, 2, 2, 2, 2, 2]
        pool = [1, 4, 4, 4, 2, 2, 2, 1]
        filter_s = [1024, 1024, 128, 128, 128, 128, 128, 128]

        # Convolutional layers
        for l in range(conv_pool_n):
            self.conv_layers.append(
                nn.Conv2d(1, filter_s[l], kernel_size=(conv_x[l], conv_y[l]), stride=1)
            )
            self.conv_layers.append(nn.BatchNorm2d(filter_s[l]))
            self.conv_layers.append(nn.Dropout(dropout_rate))
            self.conv_layers.append(nn.AvgPool2d(kernel_size=(1, pool[l])))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(filter_s[-1] * Aln_length, 1024)
        self.fc2 = nn.Linear(1024, Nlabels)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

# Function to parse fasta and convert to numeric
def fasta_pars(aln_file, seq_number, Lmax):
    dic = {'A': '0', 'T': '1', 'C': '2', 'G': '3', '-': '4'}
    matrix_out = []
    fasta_dic = {}
    with open(aln_file) as aln:
        for line in aln:
            if line[0] == ">":
                header = line[1:].rstrip('\n').strip()
                fasta_dic[header] = []
            elif line[0].isalpha() or line[0] == '-':
                for base, num in dic.items():
                    line = line[:].rstrip('\n').strip().replace(base, num)
                line = list(line)
                line = [int(n) for n in line]
                fasta_dic[header] += line + [-15] * (Lmax - len(line))
                if len(fasta_dic) == seq_number:
                    taxa_block = [fasta_dic[taxa.strip()] for taxa in sorted(fasta_dic.keys())]
                    fasta_dic = {}
                    matrix_out.append(taxa_block)
    return np.array(matrix_out)

# Read training, validation, and test datasets to equalize sizes 
def tv_parse(train, valid, test, seq_number):
    with open(train) as tr, open(valid) as va, open(test) as te:
        LT = max([len(r.strip()) for r in tr])
        LV = max([len(r.strip()) for r in va])
        LTE = max([len(r.strip()) for r in te])
    Lmax = max([LT, LV, LTE])
    tr = fasta_pars(train, seq_number, Lmax)
    va = fasta_pars(valid, seq_number, Lmax)
    te = fasta_pars(test, seq_number, Lmax)
    return tr, va, te

# Training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs=200):
    best_valid_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Valid Loss: {valid_loss/len(valid_loader)}')

# Evaluate model
def evaluate_model(model, test_loader, device):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'Test set: Average loss: {test_loss/len(test_loader.dataset)}, Accuracy: {correct/len(test_loader.dataset)}')

# Main function
def main():
    parser = argparse.ArgumentParser(description='PyTorch run')
    parser.add_argument('-t', help="Training dataset in .npy", dest='TRAIN')
    parser.add_argument('-v', help="Validation dataset in .npy", dest='VALID')
    parser.add_argument('--test', help="Test dataset in .npy", dest='TEST')
    parser.add_argument('-N', help="N taxa", type=int, dest='Ntaxa')
    args = parser.parse_args()

    # Load data
    print("Reading input")
    train_data1 = np.load(args.TRAIN)
    valid_data1 = np.load(args.VALID)
    test_data1 = np.load(args.TEST)
    print("Done")

    # Generate labels
    Nlabels = n_unroot(args.Ntaxa)
    encoder = OneHotEncoder(sparse=False)
    train_label = encoder.fit_transform(np.repeat(range(0, Nlabels), len(train_data1) // Nlabels).reshape(-1, 1))
    valid_label = encoder.transform(np.repeat(range(0, Nlabels), len(valid_data1) // Nlabels).reshape(-1, 1))
    test_label = encoder.transform(np.repeat(range(0, Nlabels), len(test_data1) // Nlabels).reshape(-1, 1))

    # Prepare data for DataLoader
    train_data1 = torch.tensor(train_data1, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    valid_data1 = torch.tensor(valid_data1, dtype=torch.float32).unsqueeze(1)
    test_data1 = torch.tensor(test_data1, dtype=torch.float32).unsqueeze(1)
    train_label = torch.tensor(train_label, dtype=torch.long)
    valid_label = torch.tensor(valid_label, dtype=torch.long)
    test_label = torch.tensor(test_label, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_data1, train_label), batch_size=100, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_data1, valid_label), batch_size=100, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data1, test_label), batch_size=100, shuffle=False)

    # Model parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhyloCNN(args.Ntaxa, train_data1.shape[2], Nlabels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    start_time = time.time()
    train_model(model, train_loader, valid_loader, criterion, optimizer, device)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")

    # Evaluate model
    start_time = time.time()
    evaluate_model(model, test_loader, device)
    end_time = time.time()
    print(f"Testing time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
