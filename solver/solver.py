import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class Solver:
    def __init__(self, 
                 model, 
                 train_set, 
                 test_set, 
                 criterion, 
                 optimizer, 
                 scheduler, 
                 device, 
                 world_size = None,
                 batch_size = 64, 
                 cnn_trans = False,
                 distr = False):
        """
        Initialize the Solver with the required components.

        Args:
            model (nn.Module): The neural network model.
            train_set (DataLoader): DataLoader for the training data.
            test_set (DataLoader): DataLoader for the testing data.
            criterion (nn.Module): Loss function.
            optimizer (optim.Optimizer): Optimizer for training.
            scheduler (optim.lr_scheduler): learning rate scheduler
            device (torch.device): The device to run the model on.
        """
        train_size = int(0.8 * len(train_set))
        val_size = len(train_set) - train_size

        self.train_set, self.val_set = random_split(train_set, [train_size, val_size])
        self.test_set = test_set

        self.model = model.to(device)

        if distr:
            if world_size > 1:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[device])

        self.criterion = criterion
        self.optimizer = optimizer

        self.device = device

        self.cnn_trans = cnn_trans

        self.scheduler = scheduler

        self.accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=14).to(self.device)
        self.precision_metric = torchmetrics.Precision(task="multiclass", num_classes=14).to(self.device)
        self.recall_metric = torchmetrics.Recall(task="multiclass", num_classes=14).to(self.device)

        self.criterion = criterion
        self.optim = optimizer

        self.train_accuracy = []
        self.train_precision = []
        self.train_recall = []

        self.val_accuracy = []
        self.val_precision = []
        self.val_recall = []

        self.loss_history = []

        self.distr = distr
        if distr:
            self.rank = device
            self.world_size = world_size
        
        self.batch_size = batch_size


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
        self.model.to(self.device)
        best_val_accracy = 0
        best_param = None

        if self.distr:
            self.train_sampler = DistributedSampler(self.train_set, num_replicas=self.world_size, rank=self.rank)
            train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, sampler=self.train_sampler)
        else:
            train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            if self.distr:
                if self.world_size > 1:
                    self.train_sampler.set_epoch(epoch)
            loss_history = []
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                if not self.cnn_trans:
                    inputs = inputs.permute(0, 2, 1, 3, 4)
                # print(labels)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_history.append(loss.item())
            epoch_loss = sum(loss_history) / i
            self.loss_history.append(epoch_loss)

            results_train = self.test(self.train_set, num_samples=1000)
            results_val = self.test(self.val_set, num_samples=1000)

            self.train_accuracy.append(results_train["accuracy"])
            self.train_precision.append(results_train["precision"])
            self.train_recall.append(results_train["recall"])

            self.val_accuracy.append(results_val["accuracy"])
            self.val_precision.append(results_val["precision"])
            self.val_recall.append(results_val["recall"])

            if results_val["accuracy"] > best_val_accracy:
                best_val_accracy = results_val["accuracy"]
                best_param = self.model.state_dict().copy()

        self.scheduler.step()
        self.model.load_state_dict(best_param)

        return {
            "loss":self.loss_history,
            "train_accuracy":self.train_accuracy,
            "train_precision":self.train_precision,
            "train_recall":self.train_recall,
            "val_accuracy":self.val_accuracy,
            "val_precision":self.val_precision,
            "val_recall":self.val_recall,  
        }
    

    def test(self, dataset, num_samples=None):
        """
        Test the model.
        """
        self.model.to(self.device)
        self.model.eval()
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()

        dataset_size = len(dataset)

        if num_samples and num_samples < dataset_size:
            dataset, _ = random_split(dataset, [num_samples, dataset_size - num_samples])

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for data in loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                if not self.cnn_trans:
                    inputs = inputs.permute(0, 2, 1, 3, 4)
                    
                outputs = self.model(inputs)
                self.accuracy_metric(outputs, labels)
                self.precision_metric(outputs, labels)
                self.recall_metric(outputs, labels)
            
            test_accuracy = self.accuracy_metric.compute().item()
            test_recall = self.recall_metric.compute().item()
            test_precision = self.precision_metric.compute().item()
        self.model.train()
        return {
            "accuracy":test_accuracy,
            "precision":test_precision,
            "recall":test_recall
        }

