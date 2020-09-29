import argparse
import gc
import os
import random
from typing import AnyStr
from typing import List
import ipdb
import krippendorff
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import AdamW
from transformers import DistilBertConfig
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from datareader import MultiDomainSentimentDataset
from datareader import collate_batch_transformer
from metrics import MultiDatasetClassificationEvaluator
from metrics import ClassificationEvaluator
from metrics import DomainClassifierEvaluator
from metrics import acc_f1

from metrics import plot_label_distribution
from model import MultiTransformerClassifier
from model import VanillaBert
from model import *


def train_domain_classifier(
        model: torch.nn.Module,
        train_dl: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        validation_evaluator: MultiDatasetClassificationEvaluator,
        n_epochs: int,
        device: AnyStr,
        log_interval: int = 1,
        patience: int = 10,
        model_dir: str = "wandb_local",
        gradient_accumulation: int = 1,
        domain_name: str = ''
):
    #best_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0

    epoch_counter = 0
    total = sum(len(dl) for dl in train_dls)

    # Main loop
    while epoch_counter < n_epochs:
        for i,batch in enumerate(tqdm(train_dl)):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            masks = batch[1]
            labels = batch[2]
            # Testing with random domains to see if any effect
            #domains = torch.tensor(np.random.randint(0, 16, batch[3].shape)).to(device)
            domains = batch[3]

            loss, logits = model(input_ids, attention_mask=masks, labels=domains)
            loss = loss / gradient_accumulation

            if i % gradient_accumulation == 0:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        gc.collect()

        # Inline evaluation
        (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(model)
        print(f"Validation acc: {acc}")

        # Saving the best model and early stopping
        #if val_loss < best_loss:
        if acc > best_acc:
            best_model = model.state_dict()
            best_acc = acc
            torch.save(model.state_dict(), f'{model_dir}/{Path(wandb.run.dir).name}/model_domainclassifier_{domain_name}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            # Stop training once we have lost patience
            if patience_counter == patience:
                break

        gc.collect()
        epoch_counter += 1


def train(
        model: torch.nn.Module,
        train_dls: List[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        validation_evaluator: MultiDatasetClassificationEvaluator,
        n_epochs: int,
        device: AnyStr,
        log_interval: int = 1,
        patience: int = 10,
        model_dir: str = "wandb_local",
        gradient_accumulation: int = 1,
        domain_name: str = ''
):
    #best_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0

    epoch_counter = 0
    total = sum(len(dl) for dl in train_dls)

    # Main loop
    while epoch_counter < n_epochs:
        dl_iters = [iter(dl) for dl in train_dls]
        dl_idx = list(range(len(dl_iters)))
        finished = [0] * len(dl_iters)
        i = 0
        with tqdm(total=total, desc="Training") as pbar:
            while sum(finished) < len(dl_iters):
                random.shuffle(dl_idx)
                for d in dl_idx:
                    domain_dl = dl_iters[d]
                    batches = []
                    try:
                        for j in range(gradient_accumulation):
                            batches.append(next(domain_dl))
                    except StopIteration:
                        finished[d] = 1
                        if len(batches) == 0:
                            continue
                    optimizer.zero_grad()
                    for batch in batches:
                        model.train()
                        batch = tuple(t.to(device) for t in batch)
                        input_ids = batch[0]
                        masks = batch[1]
                        labels = batch[2]
                        # Testing with random domains to see if any effect
                        #domains = torch.tensor(np.random.randint(0, 16, batch[3].shape)).to(device)
                        domains = batch[3]

                        loss, logits, alpha = model(input_ids, attention_mask=masks, domains=domains, labels=labels, ret_alpha = True)
                        loss = loss.mean() / gradient_accumulation
                        if i % log_interval == 0:
                            # wandb.log({
                            #     "Loss": loss.item(),
                            #     "alpha0": alpha[:,0].cpu(),
                            #     "alpha1": alpha[:, 1].cpu(),
                            #     "alpha2": alpha[:, 2].cpu(),
                            #     "alpha_shared": alpha[:, 3].cpu()
                            # })
                            wandb.log({
                                "Loss": loss.item()
                            })

                        loss.backward()
                        i += 1
                        pbar.update(1)

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

        gc.collect()

        # Inline evaluation
        (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(model)
        print(f"Validation acc: {acc}")

        #torch.save(model.state_dict(), f'{model_dir}/{Path(wandb.run.dir).name}/model_{domain_name}.pth')

        # Saving the best model and early stopping
        #if val_loss < best_loss:
        if acc > best_acc:
            best_model = model.state_dict()
            #best_loss = val_loss
            best_acc = acc
            #wandb.run.summary['best_validation_loss'] = best_loss
            torch.save(model.state_dict(), f'{model_dir}/{Path(wandb.run.dir).name}/model_{domain_name}.pth')
            patience_counter = 0
            # Log to wandb
            wandb.log({
                'Validation accuracy': acc,
                'Validation Precision': P,
                'Validation Recall': R,
                'Validation F1': F1,
                'Validation loss': val_loss})
        else:
            patience_counter += 1
            # Stop training once we have lost patience
            if patience_counter == patience:
                break

        gc.collect()
        epoch_counter += 1


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_loc", help="Root directory of the dataset", required=True, type=str)
    parser.add_argument("--train_pct", help="Percentage of data to use for training", type=float, default=0.8)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=0)
    parser.add_argument("--log_interval", help="Number of steps to take between logging steps", type=int, default=1)
    parser.add_argument("--warmup_steps", help="Number of steps to warm up Adam", type=int, default=200)
    parser.add_argument("--n_epochs", help="Number of epochs", type=int, default=2)
    parser.add_argument("--n_dc_epochs", help="Number of epochs for domain classifier training", type=int, default=2)
    parser.add_argument("--pretrained_bert", help="Directory with weights to initialize the shared model with", type=str, default=None)
    parser.add_argument("--pretrained_multi_xformer", help="Directory with weights to initialize the domain specific models", type=str, default=None)
    parser.add_argument("--domains", nargs='+', help='A list of domains to use for training', default=[])
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--run_name", type=str, help="A name for the run", default="pheme-baseline")
    parser.add_argument("--model_dir", help="Where to store the saved model", default="wandb_local", type=str)
    parser.add_argument("--tags", nargs='+', help='A list of tags for this run', default=[])
    parser.add_argument("--batch_size", help="The batch size", type=int, default=16)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", help="l2 reg", type=float, default=0.01)
    parser.add_argument("--n_heads", help="Number of transformer heads", default=6, type=int)
    parser.add_argument("--n_layers", help="Number of transformer layers", default=6, type=int)
    parser.add_argument("--d_model", help="Transformer model size", default=768, type=int)
    parser.add_argument("--ff_dim", help="Intermediate feedforward size", default=2048, type=int)
    parser.add_argument("--gradient_accumulation", help="Number of gradient accumulation steps", default=1, type=int)
    parser.add_argument("--model", help="Name of the model to run", default="VanillaBert")
    parser.add_argument("--indices_dir", help="If standard splits are being used", type=str, default=None)
    parser.add_argument("--ensemble_basic", help="Use averaging for the ensembling method", action="store_true")


    args = parser.parse_args()

    # Set all the seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # See if CUDA available
    device = torch.device("cpu")
    if args.n_gpu > 0 and torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    # model configuration
    bert_model = 'distilbert-base-uncased'
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs
    n_dc_epochs = args.n_dc_epochs
    bert_config = DistilBertConfig.from_pretrained(bert_model, num_labels=2, output_hidden_states=True)


    # wandb initialization
    wandb.init(
        project="domain-adaptation-sentiment-emnlp",
        name=args.run_name,
        config={
            "epochs": n_epochs,
            "learning_rate": lr,
            "warmup": args.warmup_steps,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "train_split_percentage": args.train_pct,
            "bert_model": bert_model,
            "seed": seed,
            "tags": ",".join(args.tags)
        }
    )
    #wandb.watch(model)
    #Create save directory for model
    if not os.path.exists(f"{args.model_dir}/{Path(wandb.run.dir).name}"):
        os.makedirs(f"{args.model_dir}/{Path(wandb.run.dir).name}")

    # Create the dataset
    all_dsets = [MultiDomainSentimentDataset(
        args.dataset_loc,
        [domain],
        DistilBertTokenizer.from_pretrained(bert_model)
    ) for domain in args.domains]
    train_sizes = [int(len(dset) * args.train_pct) for j, dset in enumerate(all_dsets)]
    val_sizes = [len(all_dsets[j]) - train_sizes[j] for j in range(len(train_sizes))]

    accs = []
    Ps = []
    Rs = []
    F1s = []
    # Store labels and logits for individual splits for micro F1
    labels_all = []
    logits_all = []

    for i in range(len(all_dsets)):
        domain = args.domains[i]
        test_dset = all_dsets[i]
        # Override the domain IDs
        k = 0
        for j in range(len(all_dsets)):
            if j != i:
                all_dsets[j].set_domain_id(k)
                k += 1
        test_dset.set_domain_id(k)
        # For test
        #all_dsets = [all_dsets[0], all_dsets[2]]

        # Split the data
        if args.indices_dir is None:
            subsets = [random_split(all_dsets[j], [train_sizes[j], val_sizes[j]])
                       for j in range(len(all_dsets)) if j != i]
        else:
            # load the indices
            dset_choices = [all_dsets[j] for j in range(len(all_dsets)) if j != i]
            subset_indices = defaultdict(lambda: [[], []])
            with open(f'{args.indices_dir}/train_idx_{domain}.txt') as f, \
                    open(f'{args.indices_dir}/val_idx_{domain}.txt') as g:
                for l in f:
                    vals = l.strip().split(',')
                    subset_indices[int(vals[0])][0].append(int(vals[1]))
                for l in g:
                    vals = l.strip().split(',')
                    subset_indices[int(vals[0])][1].append(int(vals[1]))
            subsets = [[Subset(dset_choices[d], subset_indices[d][0]), Subset(dset_choices[d], subset_indices[d][1])] for d in
                       subset_indices]

        train_dls = [DataLoader(
            subset[0],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch_transformer
        ) for subset in subsets]

        val_ds = [subset[1] for subset in subsets]
        # for vds in val_ds:
        #     print(vds.indices)
        validation_evaluator = MultiDatasetClassificationEvaluator(val_ds, device)

        # Create the model
        shared_bert_config = DistilBertConfig.from_pretrained(bert_model, num_labels=len(train_dls))
        bert = DistilBertForSequenceClassification.from_pretrained(bert_model, config=shared_bert_config).to(device)
        # 1) Create a domain classifier with BERT
        shared_bert = VanillaBert(bert).to(device)

        domain_classifier_train_dset = ConcatDataset([subset[0] for subset in subsets])
        domain_classifier_train_dl = DataLoader(
            domain_classifier_train_dset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch_transformer
        )
        domain_classifier_val_dset = ConcatDataset([subset[1] for subset in subsets])
        domain_classifier_val_evaluator = DomainClassifierEvaluator(domain_classifier_val_dset, device)
        # Create the optimizer
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in shared_bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in shared_bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            args.warmup_steps,
            n_dc_epochs * len(domain_classifier_train_dl)
        )
        train_domain_classifier(
            shared_bert,
            domain_classifier_train_dl,
            optimizer,
            scheduler,
            domain_classifier_val_evaluator,
            n_dc_epochs,
            device,
            args.log_interval,
            model_dir=args.model_dir,
            gradient_accumulation=args.gradient_accumulation,
            domain_name=domain
        )
        # load the trained model
        shared_bert.load_state_dict(torch.load(f'{args.model_dir}/{Path(wandb.run.dir).name}/model_domainclassifier_{domain}.pth'))
        # Freeze the parameters
        for p in shared_bert.parameters():
            p.requires_grad = False

        multi_xformer = MultiDistilBertClassifier(
            bert_model,
            bert_config,
            n_domains=len(train_dls)
        ).to(device)

        model = torch.nn.DataParallel(MultiViewTransformerNetworkDomainClassifierAttention(
            multi_xformer,
            shared_bert
        )).to(device)
        # (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(model)
        # print(f"Validation acc starting: {acc}")

        # Create the optimizer
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            args.warmup_steps,
            n_epochs * sum([len(train_dl) for train_dl in train_dls])
        )

        # Train
        train(
            model,
            train_dls,
            optimizer,
            scheduler,
            validation_evaluator,
            n_epochs,
            device,
            args.log_interval,
            model_dir=args.model_dir,
            gradient_accumulation=args.gradient_accumulation,
            domain_name=domain
        )
        # Load the best weights
        model.load_state_dict(torch.load(f'{args.model_dir}/{Path(wandb.run.dir).name}/model_{domain}.pth'))

        evaluator = ClassificationEvaluator(test_dset, device, use_domain=False)
        (loss, acc, P, R, F1), plots, (labels, logits), votes = evaluator.evaluate(
            model,
            plot_callbacks=[plot_label_distribution],
            return_labels_logits=True,
            return_votes=True
        )
        print(f"{domain} F1: {F1}")
        print(f"{domain} Accuracy: {acc}")
        print()

        wandb.run.summary[f"{domain}-P"] = P
        wandb.run.summary[f"{domain}-R"] = R
        wandb.run.summary[f"{domain}-F1"] = F1
        wandb.run.summary[f"{domain}-Acc"] = acc
        Ps.append(P)
        Rs.append(R)
        F1s.append(F1)
        accs.append(acc)
        labels_all.extend(labels)
        logits_all.extend(logits)
        with open(f'{args.model_dir}/{Path(wandb.run.dir).name}/pred_lab.txt', 'a+') as f:
            for p, l in zip(np.argmax(logits, axis=-1), labels):
                f.write(f'{domain}\t{p}\t{l}\n')

    acc, P, R, F1 = acc_f1(logits_all, labels_all)
    # Add to wandb
    wandb.run.summary[f'test-loss'] = loss
    wandb.run.summary[f'test-micro-acc'] = acc
    wandb.run.summary[f'test-micro-P'] = P
    wandb.run.summary[f'test-micro-R'] = R
    wandb.run.summary[f'test-micro-F1'] = F1

    wandb.run.summary[f'test-macro-acc'] = sum(accs) / len(accs)
    wandb.run.summary[f'test-macro-P'] = sum(Ps) / len(Ps)
    wandb.run.summary[f'test-macro-R'] = sum(Rs) / len(Rs)
    wandb.run.summary[f'test-macro-F1'] = sum(F1s) / len(F1s)

    # wandb.log({f"label-distribution-test-{i}": plots[0]})
