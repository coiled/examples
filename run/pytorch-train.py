"""
This example was adapted from the following PyTorch tutorial
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
"""

import os
import sys
import dask
import coiled
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision import datasets, transforms


coiled.create_software_environment(
    name="pytorch",
    conda={
        "channels": ["pytorch", "nvidia", "conda-forge", "defaults"],
        "dependencies": [
            "python=" + sys.version.split(" ")[0],
            "dask=" + dask.__version__,
            "coiled",
            "pytorch",
            "cudatoolkit",
            "pynvml",
            "torchvision",
        ],
    },
    gpu_enabled=True,
)


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = datasets.FashionMNIST(os.getcwd(), train=True, transform=transform, download=True)
    validation_set = datasets.FashionMNIST(os.getcwd(), train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    return training_loader, validation_loader


class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_one_epoch(model, loss_fn, optimizer, training_loader, device):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Move to GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss

@coiled.run(
    vm_type="g5.xlarge",  # A GPU Instance Type
    software="pytorch",   # Our software environment defined above
    region="us-west-2",   # We find GPUs are easier to get here
)
def train_all_epochs():
    #  Confirm that GPU shows up
    print(
        "Available GPU is "
        f"{torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else '<none>'}"
    )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    training_loader, validation_loader = load_data()
    model = GarmentClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 5
    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print(f'EPOCH {epoch + 1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, loss_fn, optimizer, training_loader, device)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata

                # Move to GPU
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)

                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Return the best model
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_model = model

    print(f"Model on CUDA device: {next(best_model.parameters()).is_cuda}")

    # Move model to CPU so it can be serialized and returned to local machine
    best_model = best_model.to("cpu")

    return best_model

model = train_all_epochs()

# Save model locally
torch.save(model.state_dict(), "model.pt")

# Load model back to your machine for more training, inference, or analysis
# device = torch.device('cpu')
# saved_model = GarmentClassifier()
# saved_model.load_state_dict(torch.load('model.pt', map_location=device))