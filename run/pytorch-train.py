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


def train_one_epoch(model, loss_fn, optimizer, training_loader):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

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
    training_loader, validation_loader = load_data()
    model = GarmentClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 5
    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, loss_fn, optimizer, training_loader)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            return model
            # best_vloss = avg_vloss
            # model_path = f'model_{epoch}'
            # torch.save(model.state_dict(), model_path)

model = train_all_epochs()

# Load 
# device = torch.device('cpu')
# saved_model = GarmentClassifier()
# saved_model.load_state_dict(torch.load('model_3', map_location=device))