import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .solver import Solver as _Solver
from edl_playground.edl.metrics import AccuracyConsideringUncertaintyThresh, RejectedCorrectsConsideringUncertaintyTresh, get_probs

class Solver(_Solver):
    def __init__(self, model, train_set, test_set, criterion, optimizer, scheduler, device, uncertainty_thresh: float, activation=F.relu, cnn_trans = False, consider_uncertainty_for_best_model_val_accuracy: bool=True):
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
            activation (function): Activation function used on model output to get evidence
            consider_uncertainty_for_best_model_val_accuracy (bool): Whether to considering the uncertainty threshold in the accuracy calculation for determining the best model
        """
        super().__init__(model, train_set, test_set, criterion, optimizer, scheduler, device, cnn_trans)
        self.activation = activation

        self.accuracy_metric_considering_uncertainty = AccuracyConsideringUncertaintyThresh(uncertainty_thresh).to(self.device)
        self.rejected_corrects_metric = RejectedCorrectsConsideringUncertaintyTresh(uncertainty_thresh).to(self.device)

        self.train_accuracy_considering_uncertainty = []
        self.train_rejected_corrects = []

        self.val_accuracy_considering_uncertainty = []
        self.val_rejected_corrects = []

        if consider_uncertainty_for_best_model_val_accuracy:
            self.best_model_metric = "accuracy_considering_uncertainty"
        else:
            self.best_model_metric = "accuracy"

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
        train_loader = DataLoader(self.train_set, batch_size=128, shuffle=True)
        
        for epoch in range(num_epochs):
            
            loss_history = []

            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                if not self.cnn_trans:
                    inputs = inputs.permute(0, 2, 1, 3, 4)
                # print(labels)
                
                outputs = self.model(inputs)
                evidence = self.activation(outputs)

                individual_loss = self.criterion(evidence, labels, epoch)
                loss = torch.mean(individual_loss)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_history.append(loss.item())
                break

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

            self.train_accuracy_considering_uncertainty.append(results_train["accuracy_considering_uncertainty"])
            self.train_rejected_corrects.append(results_train["rejected_corrects_share"])

            self.val_accuracy_considering_uncertainty.append(results_val["accuracy_considering_uncertainty"])
            self.val_rejected_corrects.append(results_val["rejected_corrects_share"])

            if results_val[self.best_model_metric] > best_val_accracy:
                best_val_accracy = results_val[self.best_model_metric]
                best_param = self.model.state_dict().copy()

        self.scheduler.step()
        self.model.load_state_dict(best_param)

        return {
            "loss":self.loss_history,
            "train_accuracy":self.train_accuracy,
            "train_accuracy_considering_uncertainty":self.train_accuracy_considering_uncertainty,
            "train_rejected_corrects_share":self.train_rejected_corrects,
            "train_precision":self.train_precision,
            "train_recall":self.train_recall,
            "val_accuracy":self.val_accuracy,
            "val_accuracy_considering_uncertainty":self.val_accuracy_considering_uncertainty,
            "val_rejected_corrects_share":self.val_rejected_corrects,
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

        loader = DataLoader(dataset, batch_size=128, shuffle=False)

        with torch.no_grad():
            for data in loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                if not self.cnn_trans:
                    inputs = inputs.permute(0, 2, 1, 3, 4)
                    
                outputs = self.model(inputs)
                evidence = self.activation(outputs)
                probs = get_probs(evidence)

                self.accuracy_metric(probs, labels)
                self.precision_metric(probs, labels)
                self.recall_metric(probs, labels)
                self.accuracy_metric_considering_uncertainty(evidence, labels)
                self.rejected_corrects_metric(evidence, labels)
                break
            
            test_accuracy = self.accuracy_metric.compute().item()
            test_recall = self.recall_metric.compute().item()
            test_precision = self.precision_metric.compute().item()
            test_accuracy_considering_uncertainty = self.accuracy_metric_considering_uncertainty.compute().item()
            test_rejected_corrects = self.rejected_corrects_metric.compute().item()
        self.model.train()
        return {
            "accuracy":test_accuracy,
            "accuracy_considering_uncertainty":test_accuracy_considering_uncertainty,
            "rejected_corrects_share":test_rejected_corrects,
            "precision":test_precision,
            "recall":test_recall
        }