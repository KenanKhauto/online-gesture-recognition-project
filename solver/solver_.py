import time
import datetime
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
                 distr = False,
                 detector = False,
                 save_every = None,
                 path_to_save = None,
                 num_classes = None):
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
        if num_classes is None:
            num_classes = 14
            if detector:
                num_classes = 2

        train_size = int(0.9 * len(train_set))
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

        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = None

        self.accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.precision_metric = torchmetrics.Precision(task="multiclass", num_classes=num_classes).to(self.device)
        self.recall_metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes).to(self.device)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(self.device)

        self.criterion = criterion
        self.optim = optimizer

        self.train_accuracy = []
        self.train_precision = []
        self.train_recall = []
        self.train_confusion_matrix = []

        self.val_accuracy = []
        self.val_precision = []
        self.val_recall = []
        self.val_confusion_matrix = []

        self.loss_history = []

        self.distr = distr
        if distr:
            self.rank = device
            self.world_size = world_size
        
        self.batch_size = batch_size

        if save_every and path_to_save:
            self.save_every = save_every
            self.path_to_save = path_to_save
        elif save_every and not path_to_save:
            raise ValueError(f"You are trying to save every {save_every} epochs but you have not specified a path")
        else:
            self.save_every = None

        self.detector = detector

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

        total_batches = len(train_loader)
        
        for epoch in range(num_epochs):
            if not self.distr or self.rank == 0:
                start = time.time()
            if self.distr:
                if self.world_size > 1:
                    self.train_sampler.set_epoch(epoch)
            loss_history = []
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                if not self.cnn_trans:
                    inputs = inputs.permute(0, 2, 1, 3, 4)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_history.append(loss.item())

                if i % (total_batches//4) == 0 and (not self.distr or self.rank == 0):
                    elapsed = int(time.time() - start)
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print(f"Epoch {epoch + 1}/{num_epochs} | Batch {i+1}/{total_batches} | Batch Loss {loss.item():.2f} | Elapsed Time for Epoch {elapsed}")

            epoch_loss = sum(loss_history) / i
            self.loss_history.append(epoch_loss)

            if not self.distr or self.rank == 0:
                elapsed = int(time.time() - start)
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"Epoch {epoch + 1}/{num_epochs} | Loss {epoch_loss:.2f} | Elapsed Time for Training {elapsed}")
                print()

                start = time.time()
                print("Calculating Train Accuracy")

            results_train = self.test(self.train_set, num_samples=1000)

            if not self.distr or self.rank == 0:
                elapsed = int(time.time() - start)
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"Train AccU {100*results_train['accuracy']:.2f} | Train Acc {100*results_train['accuracy']:.2f} | Elapsed Time {elapsed}")
                print()

                start = time.time()
                print("Calculating Val Accuracy")

            results_val = self.test(self.val_set, num_samples=1000)

            if not self.distr or self.rank == 0:
                elapsed = int(time.time() - start)
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"Val AccU {100*results_val['accuracy']:.2f} | Val Acc {100*results_val['accuracy']:.2f} | Elapsed Time {elapsed}")
                print()

            self.train_accuracy.append(results_train["accuracy"])
            self.train_precision.append(results_train["precision"])
            self.train_recall.append(results_train["recall"])
            self.train_confusion_matrix.append(results_train["confusion_matrix"])

            self.val_accuracy.append(results_val["accuracy"])
            self.val_precision.append(results_val["precision"])
            self.val_recall.append(results_val["recall"])
            self.val_confusion_matrix.append(results_val["confusion_matrix"])

            if results_val["accuracy"] > best_val_accracy:
                if not self.distr or self.rank == 0:
                    print(f"Saving model {results_val['accuracy']:.2f} > {best_val_accracy:.2f}")
                    print()
                best_val_accracy = results_val["accuracy"]
                best_param = self.model.state_dict().copy()

            if self.scheduler:
                self.scheduler.step()

            if self.distr:
                if self.rank == 0:
                    if self.save_every and self.path_to_save and epoch % self.save_every == 0:
                        self.save(self.path_to_save)
            else:
                if self.save_every and self.path_to_save and epoch % self.save_every == 0:
                    self.save(self.path_to_save)

        self.model.load_state_dict(best_param)

        return {
            "loss":self.loss_history,
            "train_accuracy":self.train_accuracy,
            "train_precision":self.train_precision,
            "train_recall":self.train_recall,
            "train_confusion_matrix":self.train_confusion_matrix,
            "val_accuracy":self.val_accuracy,
            "val_precision":self.val_precision,
            "val_recall":self.val_recall,  
            "val_confusion_matrix":self.val_confusion_matrix,
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
        total_batches = len(loader)

        with torch.no_grad():
            for i, data in enumerate(loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                if not self.cnn_trans:
                    inputs = inputs.permute(0, 2, 1, 3, 4)
                    
                outputs = self.model(inputs)
                self.accuracy_metric(outputs, labels)
                self.precision_metric(outputs, labels)
                self.recall_metric(outputs, labels)
                self.confusion_matrix(outputs, labels)

                if i % (total_batches//2) == 0 and (not self.distr or self.rank == 0):
                    print(f"Batch {i+1}/{total_batches}")

        # Calculate local metrics
        local_accuracy = self.accuracy_metric.compute()
        local_precision = self.precision_metric.compute()
        local_recall = self.precision_metric.compute()
        local_confusion_matrix = self.confusion_matrix.compute()
        self.confusion_matrix.reset()
        self.model.train()

        if self.distr:
            # Convert metrics to tensors and aggregate across all processes
            local_accuracy = torch.tensor(local_accuracy).to(self.device)
            local_precision = torch.tensor(local_precision).to(self.device)
            local_recall = torch.tensor(local_recall).to(self.device)
            local_confusion_matrix = torch.tensor(local_confusion_matrix).to(self.device)

            # Use all_reduce to sum metrics across all processes
            torch.distributed.all_reduce(local_accuracy, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_precision, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_recall, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_confusion_matrix, op=torch.distributed.ReduceOp.SUM)

            # Calculate global metrics
            global_accuracy = local_accuracy / self.world_size
            global_precision = local_precision / self.world_size
            global_recall = local_recall / self.world_size
            global_confusion_matrix = local_confusion_matrix / self.world_size

            # Return metrics from the main process
            if self.rank == 0:
                return {
                    "accuracy": global_accuracy.item(),
                    "precision": global_precision.item(),
                    "recall": global_recall.item(),
                    "confusion_matrix": global_confusion_matrix.cpu()
                }
        else:
            # Non-distributed case, return local metrics directly
            return {
                "accuracy": local_accuracy.item(),
                "precision": local_precision.item(),
                "recall": local_recall.item(),
                "confusion_matrix": local_confusion_matrix.cpu()
            }

