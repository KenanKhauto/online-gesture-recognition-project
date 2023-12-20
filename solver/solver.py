import torch
import torch.nn as nn
import torch.optim as optim

class Solver:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device):
        """
        Initialize the Solver with the required components.

        Args:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for the training data.
            test_loader (DataLoader): DataLoader for the testing data.
            criterion (nn.Module): Loss function.
            optimizer (optim.Optimizer): Optimizer for training.
            device (torch.device): The device to run the model on.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def save(self, file_path):
        """
        Save the model state.

        Args:
            file_path (str): Path to the file where the model state will be saved.
        """
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path):
        """
        Load the model state.

        Args:
            file_path (str): Path to the file from which to load the model state.
        """
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))

    def train(self, num_epochs):
        """
        Train the model.

        Args:
            num_epochs (int): Number of epochs to train the model.
        """
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            bn = 1
            for data in self.train_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                inputs = inputs.permute(0, 2, 1, 3, 4)
                # print(labels)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                print(f"Batch {bn} done!")
                bn += 1
                
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(self.train_loader)}')

    def test(self):
        """
        Test the model.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total}%')
