import time
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from .solver import Solver as _Solver
from edl_playground.edl.metrics import AccuracyConsideringUncertaintyThresh, RejectedCorrectsConsideringUncertaintyTresh, get_probs

class Solver(_Solver):
    def __init__(
        self, 
        model, 
        train_set, 
        test_set, 
        criterion, 
        optimizer, 
        scheduler, 
        device,
        uncertainty_thresh: float,
        world_size = None,
        batch_size = 64, 
        cnn_trans = False,
        distr = False,
        detector = False,
        save_every = None,
        path_to_save = None,
        num_classes = None,
        activation=F.relu,
        consider_uncertainty_for_best_model_val_accuracy: bool=True
    ):
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
            uncertainty_thresh (float): Uncertainty threshold above which the model rejects to make predictions
            batch size (int): Batch size used for training and testing
            activation (function): Activation function used on model output to get evidence
            consider_uncertainty_for_best_model_val_accuracy (bool): Whether to considering the uncertainty threshold in the accuracy calculation for determining the best model
        """
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            world_size=world_size,
            batch_size=batch_size,
            cnn_trans=cnn_trans,
            distr=distr,
            detector=detector,
            save_every=save_every,
            path_to_save=path_to_save,
            num_classes=num_classes,
        )
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
        best_val_accracy = -1
        best_param = None

        if self.distr:
            self.train_sampler = DistributedSampler(self.train_set, num_replicas=self.world_size, rank=self.rank)
            train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, sampler=self.train_sampler)
        else:
            train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            start = time.time()
            if self.distr:
                if self.world_size > 1:
                    self.train_sampler.set_epoch(epoch)
            
            loss_history = []

            total_batches = len(train_loader)

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

                if i % (total_batches//4) == 0:
                    elapsed = int(time.time() - start)
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print(f"Epoch {epoch + 1}/{num_epochs} | Batch {i+1}/{total_batches} | Batch Loss {loss.item():.2f} | Elapsed Time for Epoch {elapsed}")

            epoch_loss = sum(loss_history) / i
            self.loss_history.append(epoch_loss)

            elapsed = int(time.time() - start)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss {epoch_loss:.2f} | Elapsed Time for Training {elapsed}")
            print()

            start = time.time()
            print("Calculating Train Accuracy")
            results_train = self.test(self.train_set, num_samples=1000)

            elapsed = int(time.time() - start)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print(f"Train AccU {100*results_train['accuracy_considering_uncertainty']:.2f} | Train Acc {100*results_train['accuracy']:.2f} | Elapsed Time {elapsed}")
            print()

            start = time.time()
            print("Calculating Val Accuracy")
            results_val = self.test(self.val_set, num_samples=1000)

            elapsed = int(time.time() - start)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print(f"Val AccU {100*results_val['accuracy_considering_uncertainty']:.2f} | Val Acc {100*results_val['accuracy']:.2f} | Elapsed Time {elapsed}")
            print()

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
                print(f"Saving model {results_val[self.best_model_metric]:.2f} > {best_val_accracy:.2f}")
                print()
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

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        total_batches = len(loader)

        with torch.no_grad():
            for i, data in enumerate(loader):
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

                if i % (total_batches//2) == 0:
                    print(f"Batch {i+1}/{total_batches}")
            
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