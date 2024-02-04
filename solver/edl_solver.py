import time
import datetime
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from .solver import Solver as _Solver
from edl_playground.edl.metrics import UncertaintyMatrix, get_probs
from edl_playground.edl.distribution import ModelUncertaintyDistribution
from edl_playground.edl.losses import SCORES


def _LD_to_DL(LD: List[dict]) -> Dict[str, list]:
    return {k: [d[k] for d in LD] for k in LD[0]}


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
        best_model_metric: str="Val_AccU",
        uncertainty_distribution: bool=False,
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
        self.uncertainty_thresh = uncertainty_thresh
        self.activation = activation
        self.uncertainty_distribution = uncertainty_distribution

        self.mtx = UncertaintyMatrix(uncertainty_thresh)
        if self.uncertainty_distribution:
            self.distribution = ModelUncertaintyDistribution().to(self.device)

        self.best_model_metric = best_model_metric

        self.metrics = {
            "accuracy": self.accuracy_metric,
            "precision": self.precision_metric,
            "recall": self.recall_metric
        }

        self.results = []

    def train(self, num_epochs):
        """
        Train the model.

        Args:
            num_epochs (int): Number of epochs to train the model.
        """
        self.model.train()
        self.model.to(self.device)
        best_model_score = -1
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

            elapsed = int(time.time() - start)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss {epoch_loss:.2f} | Elapsed Time for Training {elapsed}")
            print()

            start = time.time()
            print("Calculating Train Accuracy")
            results_train = self.test(self.train_set, num_samples=1000)
            metrics = {f"Train_{metric}": v for metric, v in results_train.items()}
            metrics["loss"] = epoch_loss

            elapsed = int(time.time() - start)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print(f"Train AvU {100*results_train['AvU']:.2f} | Train AccU {100*results_train['AccU']:.2f} | Train Acc {100*results_train['Acc']:.2f} | Elapsed Time {elapsed}")
            print()

            start = time.time()
            print("Calculating Val Accuracy")
            results_val = self.test(self.val_set, num_samples=1000)
            metrics = dict(metrics, **{f"Val_{metric}": v for metric, v in results_val.items()})
            self.results.append(metrics)

            elapsed = int(time.time() - start)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print(f"Val AvU {100*results_val['AvU']:.2f} | Val AccU {100*results_val['AccU']:.2f} | Val Acc {100*results_val['Acc']:.2f} | Elapsed Time {elapsed}")
            print()

            if metrics[self.best_model_metric] > best_model_score:
                print(f"Saving model {metrics[self.best_model_metric]:.2f} > {best_model_score:.2f}")
                print()
                best_model_score = metrics[self.best_model_metric]
                best_param = self.model.state_dict().copy()

            self.scheduler.step()

        self.model.load_state_dict(best_param)
        return _LD_to_DL(self.results)
    

    def test(self, dataset, num_samples=None):
        """
        Test the model.
        """
        self.model.to(self.device)
        self.model.eval()
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.mtx.reset()
        if self.uncertainty_distribution:
            self.distribution.reset()

        dataset_size = len(dataset)
        metrics = {}

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

                for metric in self.metrics.values():
                    metric(probs, labels)

                self.mtx(evidence, labels)
                if self.uncertainty_distribution:
                    self.distribution(evidence, labels)

                if i % (total_batches//2) == 0:
                    print(f"Batch {i+1}/{total_batches}")

            for name, metric in self.metrics.items():
                metrics[name] = metric.compute().item()
            
            metrics["uncertainty_matrix"] = self.mtx.compute()

            for score_fn in SCORES:
                metrics[score_fn.__name__] = score_fn(*metrics["uncertainty_matrix"])

            if self.uncertainty_distribution:
                correct, incorrect = [x.cpu().numpy() for x in self.distribution.compute()]
                metrics["correct"] = correct
                metrics["incorrect"] = incorrect

        self.model.train()

        return metrics