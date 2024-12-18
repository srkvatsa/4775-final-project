import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Set device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define constants
NUCLEOTIDE_DICT = {'A': 0, 'T': 1, 'C': 2, 'G': 3, '-': 4}
PADDING_VALUE = -15
BATCH_SIZE = 150
NUM_CLASSES = 3
# LEARNING_RATE = 0.001
# NUM_EPOCHS = 50  # You can adjust based on early stopping

LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Function to read and preprocess the data
def read_alignment_file(file_path):
    alignments = []
    labels = []
    with open(file_path, 'r') as f:
        current_seq = ''
        current_label = ''
        sequences = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq != '':
                    sequences.append(current_seq)
                    current_seq = ''
                current_label = line[1:].strip()
            elif line == '':
                if current_seq != '':
                    sequences.append(current_seq)
                    current_seq = ''
                if sequences:
                    alignments.append(sequences)
                    labels.append(current_label)
                    sequences = []
            else:
                current_seq += line
        # Add the last alignment
        if current_seq != '':
            sequences.append(current_seq)
        if sequences:
            alignments.append(sequences)
            labels.append(current_label)
    return alignments, labels

# Custom Dataset class
class AlignmentDataset(Dataset):
    def __init__(self, alignments, labels):
        self.alignments = alignments
        self.labels = labels
        self.max_length = self.get_max_length()
        print(f'Max sequence length: {self.max_length}')
        self.encoded_alignments = [self.encode_and_pad_alignment(aln) for aln in self.alignments]
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        print(f'Classes found: {self.label_encoder.classes_}')

    def get_max_length(self):
        max_len = 0
        for aln in self.alignments:
            for seq in aln:
                if len(seq) > max_len:
                    max_len = len(seq)
        return max_len

    def encode_and_pad_alignment(self, alignment):
        encoded_seqs = []
        for seq in alignment:
            encoded_seq = [NUCLEOTIDE_DICT.get(char, 5) for char in seq]  # Unknown chars get 5
            # Pad the sequence
            padding_length = self.max_length - len(encoded_seq)
            encoded_seq.extend([PADDING_VALUE] * padding_length)
            encoded_seqs.append(encoded_seq)
        return np.array(encoded_seqs)
    
    def __len__(self):
        return len(self.encoded_alignments)
    
    def __getitem__(self, idx):
        alignment = self.encoded_alignments[idx]
        label = self.encoded_labels[idx]
        # Convert to tensors
        alignment_tensor = torch.tensor(alignment, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return alignment_tensor, label_tensor

# Define the CNN model
class PhyloCNN(nn.Module):
    def __init__(self, input_channels=4):
        super(PhyloCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(1024)
        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(1024)
        # Adjusted convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(6):
            if i == 0:
                conv_layer = nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=1)
            else:
                conv_layer = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
            bn_layer = nn.BatchNorm1d(128)
            self.conv_layers.append(conv_layer)
            self.bn_layers.append(bn_layer)
        # Fully connected layers
        self.fc1 = nn.Linear(128, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, NUM_CLASSES)
        # Dropout
        self.dropout = nn.Dropout(0.2)
        # Pooling
        self.avgpool = nn.AvgPool1d(kernel_size=4)

    def forward(self, x):
        x = x.float().to(device)
        print(f'Input x shape: {x.shape}')
        # First convolutional layer
        x = F.relu(self.bn1(self.conv1(x)))
        print(f'After conv1 x shape: {x.shape}')
        x = self.dropout(x)
        x = F.avg_pool1d(x, kernel_size=1)
        # Second convolutional layer
        x = F.relu(self.bn2(self.conv2(x)))
        print(f'After conv2 x shape: {x.shape}')
        x = self.dropout(x)
        x = F.avg_pool1d(x, kernel_size=2)
        # Adjusted convolutional layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = F.relu(bn(conv(x)))
            print(f'After conv layer {i+3} x shape: {x.shape}')
            x = self.dropout(x)
            x = F.avg_pool1d(x, kernel_size=2)
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        print(f'After pooling x shape: {x.shape}')
        # Fully connected layers
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, patience=5):
    model = model.to(device)
    best_loss = np.inf
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        model.train()
        train_losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Training Loss: {loss.item()}')
        valid_losses = []
        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                valid_losses.append(loss.item())
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {valid_loss}')
        # Early stopping
        if valid_loss + 0.001 < best_loss:
            best_loss = valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), '/home/rl659/4775-final-project/saved_models/best_model.pt')
            print('Validation loss improved. Saving model.')
        else:
            epochs_no_improve += 1
            print(f'No improvement in validation loss for {epochs_no_improve} epochs.')
        if epochs_no_improve >= patience:
            print('Early stopping!')
            break
    # Load best model
    model.load_state_dict(torch.load('/home/rl659/4775-final-project/saved_models/best_model.pt'))
    return model

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_outputs.extend(F.softmax(outputs, dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test alignments: {accuracy}%')
    return all_outputs, all_targets

# Main function
def main():
    # Load datasets
    data_path = "/home/rl659/4775-final-project/data/gap50k_500/"
    train_file = os.path.join(data_path, 'TRAIN.aln')
    val_file = os.path.join(data_path, 'VALID.aln')
    test_file = os.path.join(data_path, 'TEST.aln')


    print('Loading datasets...')
    train_alignments, train_labels = read_alignment_file(train_file)
    valid_alignments, valid_labels = read_alignment_file(val_file)
    test_alignments, test_labels = read_alignment_file(test_file)

    print(f'Number of training samples: {len(train_alignments)}')
    print(f'Number of validation samples: {len(valid_alignments)}')
    print(f'Number of test samples: {len(test_alignments)}')

    # Ensure all alignments have the same number of sequences
    num_sequences = len(train_alignments[0])
    print(f'Number of sequences per alignment: {num_sequences}')

    # Create datasets and loaders
    train_dataset = AlignmentDataset(train_alignments, train_labels)
    valid_dataset = AlignmentDataset(valid_alignments, valid_labels)
    test_dataset = AlignmentDataset(test_alignments, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Adjust NUM_CLASSES if needed based on labels
    global NUM_CLASSES
    NUM_CLASSES = len(train_dataset.label_encoder.classes_)
    print(f'Number of classes: {NUM_CLASSES}')

    # Initialize model, criterion, optimizer
    model = PhyloCNN(input_channels=num_sequences)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    trained_model = train_model(model, train_loader, valid_loader, criterion, optimizer)

    # Evaluate the model
    outputs, targets = evaluate_model(trained_model, test_loader)

if __name__ == '__main__':
    main()