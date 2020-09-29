import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable, AnyStr
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from scipy.stats import entropy
import ipdb

from datareader import collate_batch_transformer


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return np.sum(np.argmax(logits, axis=-1) == labels).astype(np.float32) / float(labels.shape[0])


def acc_f1(logits: List, labels: List) -> Tuple[float, float, float, float]:
    logits = np.asarray(logits).reshape(-1, len(logits[0]))
    labels = np.asarray(labels).reshape(-1)
    acc = accuracy(logits, labels)
    average = 'binary' if logits.shape[1] == 2 else None
    P, R, F1, _ = precision_recall_fscore_support(labels, np.argmax(logits, axis=-1), average=average)
    return acc,P,R,F1


def plot_label_distribution(labels: np.ndarray, logits: np.ndarray) -> matplotlib.figure.Figure:
    """ Plots the distribution of labels in the prediction

    :param labels: Gold labels
    :param logits: Logits from the model
    :return: None
    """
    predictions = np.argmax(logits, axis=-1)
    labs, counts = zip(*list(sorted(Counter(predictions).items(), key=lambda x: x[0])))

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.bar(labs, counts, width=0.2)
    ax.set_xticks(labs, [str(l) for l in labs])
    ax.set_ylabel('Count')
    ax.set_xlabel("Label")
    ax.set_title("Prediction distribution")
    return fig


class ClassificationEvaluator:
    """Wrapper to evaluate a model for the task of citation detection

    """

    def __init__(self, dataset: Dataset, device: torch.device, use_domain: bool = True, use_labels: bool = True):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=collate_batch_transformer
        )
        self.device = device
        self.stored_labels = []
        self.stored_logits = []
        self.use_domain = use_domain
        self.use_labels = use_labels

    def micro_f1(self) -> Tuple[float, float, float, float]:
        labels_all = self.stored_labels
        logits_all = self.stored_logits

        logits_all = np.asarray(logits_all).reshape(-1, len(logits_all[0]))
        labels_all = np.asarray(labels_all).reshape(-1)
        acc = accuracy(logits_all, labels_all)
        P, R, F1, _ = precision_recall_fscore_support(labels_all, np.argmax(logits_all, axis=-1), average='binary')

        return acc, P, R, F1

    def evaluate(
            self,
            model: torch.nn.Module,
            plot_callbacks: List[Callable] = [],
            return_labels_logits: bool = False,
            return_votes: bool = False
    ) -> Tuple:
        """Collect evaluation metrics on this dataset

        :param model: The pytorch model to evaluate
        :param plot_callbacks: Optional function callbacks for plotting various things
        :return: (Loss, Accuracy, Precision, Recall, F1)
        """
        model.eval()
        with torch.no_grad():
            labels_all = []
            logits_all = []
            losses_all = []
            votes_all = []
            for batch in tqdm(self.dataloader, desc="Evaluation"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids = batch[0]
                masks = batch[1]
                labels = batch[2]
                domains = batch[3] if self.use_domain else None
                if self.use_labels:
                    loss, logits = model(input_ids, attention_mask=masks, domains=domains, labels=labels)
                    if len(loss.size()) > 0:
                        loss = loss.mean()
                else:
                    (logits,) = model(input_ids, attention_mask=masks, domains=domains)
                    loss = torch.FloatTensor([-1.])
                labels_all.extend(list(labels.detach().cpu().numpy()))
                logits_all.extend(list(logits.detach().cpu().numpy()))
                losses_all.append(loss.item())
                if hasattr(model, 'votes'):
                    votes_all.extend(model.votes.detach().cpu().numpy())

            if not self.use_labels:
                return logits_all

            acc,P,R,F1 = acc_f1(logits_all, labels_all)
            loss = sum(losses_all) / len(losses_all)

            # Plotting
            plots = []
            for f in plot_callbacks:
                plots.append(f(labels_all, logits_all))

            ret_vals = (loss, acc, P, R, F1), plots
            if return_labels_logits:
                ret_vals = ret_vals + ((labels_all, logits_all),)

            if return_votes:
                if len(votes_all) > 0:
                    ret_vals += (votes_all,)
                else:
                    ret_vals += (list(np.argmax(np.asarray(logits_all), axis=1)),)

            return ret_vals

class MultiDatasetClassificationEvaluator:
    """Wrapper to evaluate a model for the task of citation detection

    """

    def __init__(self, datasets: List[Dataset], device: torch.device, use_domain: bool = True):
        self.datasets = datasets
        self.dataloaders = [DataLoader(
            dataset,
            batch_size=8,
            collate_fn=collate_batch_transformer
        ) for dataset in datasets]

        self.device = device
        self.stored_labels = []
        self.stored_logits = []
        self.use_domain = use_domain

    def micro_f1(self) -> Tuple[float, float, float, float]:
        labels_all = self.stored_labels
        logits_all = self.stored_logits

        logits_all = np.asarray(logits_all).reshape(-1, len(logits_all[0]))
        labels_all = np.asarray(labels_all).reshape(-1)
        acc = accuracy(logits_all, labels_all)
        P, R, F1, _ = precision_recall_fscore_support(labels_all, np.argmax(logits_all, axis=-1), average='binary')

        return acc, P, R, F1

    def evaluate(
            self,
            model: torch.nn.Module,
            plot_callbacks: List[Callable] = [],
            return_labels_logits: bool = False,
            return_votes: bool = False
    ) -> Tuple:
        """Collect evaluation metrics on this dataset

        :param model: The pytorch model to evaluate
        :param plot_callbacks: Optional function callbacks for plotting various things
        :return: (Loss, Accuracy, Precision, Recall, F1)
        """
        model.eval()
        with torch.no_grad():
            labels_all = []
            logits_all = []
            losses_all = []
            votes_all = []
            for dataloader in self.dataloaders:
                for batch in tqdm(dataloader, desc="Evaluation"):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids = batch[0]
                    masks = batch[1]
                    labels = batch[2]
                    domains = batch[3] if self.use_domain else None
                    loss, logits = model(input_ids, attention_mask=masks, domains=domains, labels=labels)
                    if len(loss.size()) > 0:
                        loss = loss.mean()
                    labels_all.extend(list(labels.detach().cpu().numpy()))
                    logits_all.extend(list(logits.detach().cpu().numpy()))
                    losses_all.append(loss.item())
                    if hasattr(model, 'votes'):
                        votes_all.extend(model.votes.detach().cpu().numpy())

            # Use the domain with the lowest entropy
            if len(votes_all) > 0:
                votes_all = np.asarray(votes_all).transpose(0,1)
                entropies = [np.mean([entropy(v) for v in votes]) for votes in votes_all]
                domain = np.argmax(entropies)
                logits_all = votes_all[domain]

            acc,P,R,F1 = acc_f1(logits_all, labels_all)
            loss = sum(losses_all) / len(losses_all)

            # Plotting
            plots = []
            for f in plot_callbacks:
                plots.append(f(labels_all, logits_all))

            ret_vals = (loss, acc, P, R, F1), plots
            if return_labels_logits:
                ret_vals = ret_vals + ((labels_all, logits_all),)

            if return_votes:
                if len(votes_all) > 0:
                    ret_vals += (votes_all,)
                else:
                    ret_vals += (list(np.argmax(np.asarray(logits_all, axis=1))),)

            return ret_vals


class DomainClassifierEvaluator:
    """Wrapper to evaluate a model for the task of citation detection

    """

    def __init__(self, dataset: Dataset, device: torch.device):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=collate_batch_transformer
        )
        self.device = device
        self.stored_labels = []
        self.stored_logits = []

    def micro_f1(self) -> Tuple[float, float, float, float]:
        labels_all = self.stored_labels
        logits_all = self.stored_logits

        logits_all = np.asarray(logits_all).reshape(-1, len(logits_all[0]))
        labels_all = np.asarray(labels_all).reshape(-1)
        acc = accuracy(logits_all, labels_all)
        P, R, F1, _ = precision_recall_fscore_support(labels_all, np.argmax(logits_all, axis=-1), average='binary')

        return acc, P, R, F1

    def evaluate(
            self,
            model: torch.nn.Module,
            plot_callbacks: List[Callable] = [],
            return_labels_logits: bool = False,
            return_votes: bool = False
    ) -> Tuple:
        """Collect evaluation metrics on this dataset

        :param model: The pytorch model to evaluate
        :param plot_callbacks: Optional function callbacks for plotting various things
        :return: (Loss, Accuracy, Precision, Recall, F1)
        """
        model.eval()
        with torch.no_grad():
            labels_all = []
            logits_all = []
            losses_all = []
            votes_all = []
            for batch in tqdm(self.dataloader, desc="Evaluation"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids = batch[0]
                masks = batch[1]
                labels = batch[2]
                domains = batch[3]
                loss, logits = model(input_ids, attention_mask=masks, labels=domains)

                labels_all.extend(list(domains.detach().cpu().numpy()))
                logits_all.extend(list(logits.detach().cpu().numpy()))
                losses_all.append(loss.item())
                if hasattr(model, 'votes'):
                    votes_all.extend(model.votes.detach().cpu().numpy())

            acc,P,R,F1 = acc_f1(logits_all, labels_all)
            loss = sum(losses_all) / len(losses_all)

            # Plotting
            plots = []
            for f in plot_callbacks:
                plots.append(f(labels_all, logits_all))

            ret_vals = (loss, acc, P, R, F1), plots
            if return_labels_logits:
                ret_vals = ret_vals + ((labels_all, logits_all),)

            if return_votes:
                if len(votes_all) > 0:
                    ret_vals += (votes_all,)
                else:
                    ret_vals += (list(np.argmax(np.asarray(logits_all), axis=1)),)

            return ret_vals