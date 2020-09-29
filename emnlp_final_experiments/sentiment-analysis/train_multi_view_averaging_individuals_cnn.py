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
from torch.optim import Adam
from tqdm import tqdm
from transformers import AdamW
from transformers import DistilBertConfig
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from datareader_cnn import MultiDomainSentimentDataset
from datareader_cnn import collate_batch_cnn
from metrics import MultiDatasetClassificationEvaluator
from metrics import ClassificationEvaluator
from metrics import acc_f1

from metrics import plot_label_distribution
from model import MultiTransformerClassifier
from model import VanillaBert
from model import *
from sklearn.model_selection import ParameterSampler
from datareader_cnn import FasttextTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(
        model: torch.nn.Module,
        train_dls: List[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        validation_evaluators: MultiDatasetClassificationEvaluator,
        n_epochs: int,
        device: AnyStr,
        log_interval: int = 1,
        patience: int = 10,
        model_dir: str = "wandb_local",
        gradient_accumulation: int = 1,
        domain_name: str = ''
):
    #best_loss = float('inf')
    best_accs = [0.0]*(len(train_dls)+1)
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

                        loss, logits = model(input_ids, attention_mask=masks, domains=domains, labels=labels)
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


        gc.collect()

        # Inline evaluation
        for v, validation_evaluator in enumerate(validation_evaluators):
            (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(model)
            print(f"Validation acc {v}: {acc}")
            if scheduler is not None:
                scheduler.step(val_loss)
            #torch.save(model.state_dict(), f'{model_dir}/{Path(wandb.run.dir).name}/model_{domain_name}.pth')

            # Saving the best model and early stopping
            #if val_loss < best_loss:
            if acc > best_accs[v]:
                best_accs[v] = acc
                #wandb.run.summary['best_validation_loss'] = best_loss
                if v < len(train_dls):
                    torch.save(model.module.domain_experts[v].state_dict(),
                               f'{model_dir}/{Path(wandb.run.dir).name}/model_{domain_name}_{v}.pth')
                else:
                    torch.save(model.module.shared_model.state_dict(),
                               f'{model_dir}/{Path(wandb.run.dir).name}/model_{domain_name}_{v}.pth')
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


def attention_grid_search(
        model: torch.nn.Module,
        validation_evaluator: MultiDatasetClassificationEvaluator,
        n_epochs: int,
        seed: int
):
    best_weights = model.module.weights
    # initial
    (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(model)
    best_acc = acc
    print(acc)
    # Create the grid search
    param_dict = {1:list(range(0,11)), 2:list(range(0,11)),3:list(range(0,11)),4:list(range(0,11))}
    grid_search_params = ParameterSampler(param_dict, n_iter=n_epochs, random_state=seed)
    for d in grid_search_params:
        weights = [v for k,v in sorted(d.items(), key=lambda x:x[0])]
        weights = np.array(weights) / sum(weights)
        model.module.weights = weights
        # Inline evaluation
        (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(model)
        print(f"Weights: {weights}\tValidation acc: {acc}")

        if acc > best_acc:
            best_weights = weights
            best_acc = acc
            # Log to wandb
            wandb.log({
                'Validation accuracy': acc,
                'Validation Precision': P,
                'Validation Recall': R,
                'Validation F1': F1,
                'Validation loss': val_loss})

        gc.collect()

    return best_weights


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_loc", help="Root directory of the dataset", required=True, type=str)
    parser.add_argument("--train_pct", help="Percentage of data to use for training", type=float, default=0.8)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=0)
    parser.add_argument("--log_interval", help="Number of steps to take between logging steps", type=int, default=1)
    parser.add_argument("--dropout", help="Path to directory with pretrained embeddings", default=0.3, type=float)
    parser.add_argument("--embedding_dim", help="Dimension of embeddings", choices=[50, 100, 200, 300], default=100,
                        type=int)
    parser.add_argument("--n_epochs", help="Number of epochs", type=int, default=2)
    parser.add_argument("--domains", nargs='+', help='A list of domains to use for training', default=[])
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--run_name", type=str, help="A name for the run", default="pheme-baseline")
    parser.add_argument("--model_dir", help="Where to store the saved model", default="wandb_local", type=str)
    parser.add_argument("--tags", nargs='+', help='A list of tags for this run', default=[])
    parser.add_argument("--batch_size", help="The batch size", type=int, default=16)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", help="l2 reg", type=float, default=0.01)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=100)
    parser.add_argument("--kernel_heights", help="filter windows", type=int, nargs='+', default=[2, 4, 5])
    parser.add_argument("--stride", help="stride", type=int, default=1)
    parser.add_argument("--padding", help="padding", type=int, default=0)
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
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs


    # wandb initialization
    wandb.init(
        project="domain-adaptation-sentiment-emnlp",
        name=args.run_name,
        config={
            "epochs": n_epochs,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "train_split_percentage": args.train_pct,
            "seed": seed,
            "tags": ",".join(args.tags)
        }
    )
    #wandb.watch(model)
    #Create save directory for model
    if not os.path.exists(f"{args.model_dir}/{Path(wandb.run.dir).name}"):
        os.makedirs(f"{args.model_dir}/{Path(wandb.run.dir).name}")

    tokenizer = FasttextTokenizer(f"{args.dataset_loc}/vocabulary.txt")

    # Create the dataset
    all_dsets = [MultiDomainSentimentDataset(
        args.dataset_loc,
        [domain],
        tokenizer
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

    accs_avg = []
    Ps_avg = []
    Rs_avg = []
    F1s_avg = []
    # Store labels and logits for individual splits for micro F1
    labels_all_avg = []
    logits_all_avg = []

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
            collate_fn=collate_batch_cnn
        ) for subset in subsets]

        val_ds = [subset[1] for subset in subsets]
        # for vds in val_ds:
        #     print(vds.indices)
        validation_evaluators = [MultiDatasetClassificationEvaluator([vds], device) for vds in val_ds] + [MultiDatasetClassificationEvaluator(val_ds, device, use_domain=False)]

        embeddings = np.load(f"{args.dataset_loc}/fasttext_embeddings.npy")
        model = torch.nn.DataParallel(MultiViewCNNAveragingIndividuals(
            args,
            embeddings,
            len(train_dls)
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
        scheduler = ReduceLROnPlateau(optimizer, verbose=True)

        # Train
        train(
            model,
            train_dls,
            optimizer,
            scheduler,
            validation_evaluators,
            n_epochs,
            device,
            args.log_interval,
            model_dir=args.model_dir,
            gradient_accumulation=args.gradient_accumulation,
            domain_name=domain
        )
        # Load the best weights
        for v in range(len(train_dls)):
            model.module.domain_experts[v].load_state_dict(
                torch.load(f'{args.model_dir}/{Path(wandb.run.dir).name}/model_{domain}_{v}.pth'))
        model.module.shared_model.load_state_dict(
            torch.load(f'{args.model_dir}/{Path(wandb.run.dir).name}/model_{domain}_{len(train_dls)}.pth'))

        # Now get the best way to average them
        model.module.average = True

        # Get it before with just plain averaging
        evaluator = ClassificationEvaluator(test_dset, device, use_domain=False)
        (loss, acc, P, R, F1), plots, (labels, logits), votes = evaluator.evaluate(
            model,
            plot_callbacks=[plot_label_distribution],
            return_labels_logits=True,
            return_votes=True
        )
        print(f"{domain} F1-alt: {F1}")
        print(f"{domain} Accuracy-alt: {acc}")
        print()

        wandb.run.summary[f"{domain}-P-alt"] = P
        wandb.run.summary[f"{domain}-R-alt"] = R
        wandb.run.summary[f"{domain}-F1-alt"] = F1
        wandb.run.summary[f"{domain}-Acc-alt"] = acc
        Ps_avg.append(P)
        Rs_avg.append(R)
        F1s_avg.append(F1)
        accs_avg.append(acc)
        labels_all_avg.extend(labels)
        logits_all_avg.extend(logits)

        # Calculate the best attention weights with a grid search
        weights = attention_grid_search(
            model,
            MultiDatasetClassificationEvaluator(val_ds, device),
            30,
            seed
        )
        model.module.weights = weights
        with open(f'{args.model_dir}/{Path(wandb.run.dir).name}/weights_{domain}.txt', 'wt') as f:
            f.write(str(weights))

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

    acc, P, R, F1 = acc_f1(logits_all_avg, labels_all_avg)
    # Add to wandb
    wandb.run.summary[f'test-loss-alt'] = loss
    wandb.run.summary[f'test-micro-acc-alt'] = acc
    wandb.run.summary[f'test-micro-P-alt'] = P
    wandb.run.summary[f'test-micro-R-alt'] = R
    wandb.run.summary[f'test-micro-F1-alt'] = F1

    wandb.run.summary[f'test-macro-acc-alt'] = sum(accs_avg) / len(accs_avg)
    wandb.run.summary[f'test-macro-P-alt'] = sum(Ps_avg) / len(Ps_avg)
    wandb.run.summary[f'test-macro-R-alt'] = sum(Rs_avg) / len(Rs_avg)
    wandb.run.summary[f'test-macro-F1-alt'] = sum(F1s_avg) / len(F1s_avg)

    # wandb.log({f"label-distribution-test-{i}": plots[0]})
