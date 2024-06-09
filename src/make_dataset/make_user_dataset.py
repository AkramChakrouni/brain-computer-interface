"""
This script processes the raw EEG data to create a user-specific dataset and fine-tunes the lastest base pre-trained model 
on this dataset.

Input:
    - Name of user
    - Raw EEG data in .edf format
    
Output:
    - User-specific dataset in their own subfolder
    - User-specific model in their own subfolder

Summary:
    This script demonstrates how to fine-tune a pre-trained model on a user-specific dataset.
     1. Load Pre-trained Model: Load the model trained on a large, general dataset.
     2. Freeze Initial Layers: Freeze the initial layers to retain general features.
     3. Prepare User-specific Data: Prepare a DataLoader for the user-specific data.
     4. Fine-tune the Model: Fine-tune the model using the user-specific data.
"""



# Step 1: Load Pre-trained Model
# Load your pre-trained model. Assume AdvancedEEGMiCNN is your model class.

import torch
import torch.nn as nn

# Define the model architecture
class AdvancedEEGMiCNN(nn.Module):
    def __init__(self):
        super(AdvancedEEGMiCNN, self).__init__()
        # Define the layers of your model (example architecture)
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=32*23*125, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Adjust the output features

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the pre-trained model
model_path = r"C:\School\EE_Y3\Q4\BAP\eeg_thesis_cnn_repo\models\Advanced_EEG_MI_CNN_2024-06-06_17-17-25.pth"
model = AdvancedEEGMiCNN()
model.load_state_dict(torch.load(model_path))


# Step 2: Freeze Initial Layers
# Freeze the initial layers to retain the general features.

# Freeze all layers except the final fully connected layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the parameters of the final layers
model.fc1.weight.requires_grad = True
model.fc1.bias.requires_grad = True
model.fc2.weight.requires_grad = True
model.fc2.bias.requires_grad = True


# Step 3: Prepare User-specific Data
# Prepare the user-specific dataset and DataLoader.

from torch.utils.data import DataLoader, TensorDataset

# Assuming `user_dataset` is your user-specific dataset
# Create DataLoader for user-specific dataset
user_loader = DataLoader(user_dataset, batch_size=16, shuffle=True)


# Step 4: Fine-tune the Model
# Fine-tune the model using the user-specific data.

import torch.optim as optim

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Move model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fine-tuning loop
num_epochs = 5  # Define the number of epochs for fine-tuning

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in user_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(user_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')






