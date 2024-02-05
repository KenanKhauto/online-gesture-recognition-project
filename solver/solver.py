import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from models.utils import extract_landmarks_from_batch, process_batch_for_landmarks

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
                 num_classes = None,
                 use_lstm = False,
                 hand_marks_detector = None):
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
            raise ValueError("You have to give the number of classes")
        
        self.num_classes = num_classes
        self.train_set = train_set 
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

        if save_every and path_to_save:
            self.save_every = save_every
            self.path_to_save = path_to_save
        elif save_every and not path_to_save:
            raise ValueError(f"You are trying to save every {save_every} epochs but you have not specified a path")
        else:
            self.save_every = None

        self.detector = detector

        if use_lstm:
            self.use_lstm = use_lstm
            self.hand_landmarks_detector = hand_marks_detector

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

                if self.use_lstm:

                    processed_batch = process_batch_for_landmarks(inputs, self.device)
                    landmarks = extract_landmarks_from_batch(processed_batch, self.hand_landmarks_detector, self.device)
                    inputs = torch.tensor(landmarks).float().to(self.device)

                
                # if not self.cnn_trans:
                #     inputs = inputs.permute(0, 2, 1, 3, 4)

                outputs = self.model(inputs).float()

                if self.detector:
                    outputs = outputs[:, 1] # Now shape [N]

                    # Making sure labels are also in the correct shape [N]
                    labels = labels.view(-1).float()
                
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.zero_grad()
                self.optimizer.step()
                
                loss_history.append(loss.item())

            epoch_loss = sum(loss_history) / i
            self.loss_history.append(epoch_loss)

            results_train = self.test(self.train_set, num_samples=1000)
            results_val = self.test(self.test_set, num_samples=1000)

            self.train_accuracy.append(results_train["accuracy"])
            self.train_precision.append(results_train["precision"])
            self.train_recall.append(results_train["recall"])

            self.val_accuracy.append(results_val["accuracy"])
            self.val_precision.append(results_val["precision"])
            self.val_recall.append(results_val["recall"])

            if results_val["accuracy"] > best_val_accracy:
                best_val_accracy = results_val["accuracy"]
                best_param = self.model.state_dict().copy()

            if self.scheduler:
                self.scheduler.step()
            self.model.load_state_dict(best_param)

            if self.distr:
                if self.rank == 0:
                    if self.save_every and self.path_to_save and epoch % self.save_every == 0:
                        self.save(self.path_to_save)
            else:
                if self.save_every and self.path_to_save and epoch % self.save_every == 0:
                    self.save(self.path_to_save)

            

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
                if self.use_lstm:
                    processed_batch = process_batch_for_landmarks(inputs, self.device)
                    landmarks = extract_landmarks_from_batch(processed_batch, self.hand_landmarks_detector, self.device)
                    inputs = torch.tensor(landmarks).float()

                # if not self.cnn_trans:
                #     inputs = inputs.permute(0, 2, 1, 3, 4)
                    
                outputs = self.model(inputs)
                self.accuracy_metric(outputs, labels)
                self.precision_metric(outputs, labels)
                self.recall_metric(outputs, labels)
            
        
        # Calculate local metrics
        local_accuracy = self.accuracy_metric.compute()
        local_precision = self.precision_metric.compute()
        local_recall = self.precision_metric.compute()
        self.model.train()

        if self.distr:
            # Convert metrics to tensors and aggregate across all processes
            local_accuracy = torch.tensor(local_accuracy).to(self.device)
            local_precision = torch.tensor(local_precision).to(self.device)
            local_recall = torch.tensor(local_recall).to(self.device)

            # Use all_reduce to sum metrics across all processes
            torch.distributed.all_reduce(local_accuracy, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_precision, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_recall, op=torch.distributed.ReduceOp.SUM)

            # Calculate global metrics
            global_accuracy = local_accuracy / self.world_size
            global_precision = local_precision / self.world_size
            global_recall = local_recall / self.world_size

            # Return metrics from the main process
            if self.rank == 0:
                return {
                    "accuracy": global_accuracy.item(),
                    "precision": global_precision.item(),
                    "recall": global_recall.item()
                }
        else:
            # Non-distributed case, return local metrics directly
            return {
                "accuracy": local_accuracy.item(),
                "precision": local_precision.item(),
                "recall": local_recall.item()
            }

