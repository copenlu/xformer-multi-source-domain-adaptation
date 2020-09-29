import argparse
import gc
import os
import random
from typing import AnyStr
from typing import List
import ipdb
import krippendorff
from collections import defaultdict

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

from datareader import MultiDomainSentimentDataset
from datareader import collate_batch_transformer
from metrics import MultiDatasetClassificationEvaluator
from metrics import ClassificationEvaluator
from metrics import acc_f1

from metrics import plot_label_distribution
from model import MultiTransformerClassifier
from model import VanillaBert
from model import *
from sklearn.model_selection import ParameterSampler
from scipy.special import softmax
from scipy.stats import wasserstein_distance


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_loc", help="Root directory of the dataset", required=True, type=str)
    parser.add_argument("--train_pct", help="Percentage of data to use for training", type=float, default=0.8)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=0)
    parser.add_argument("--pretrained_model", help="Directory with weights to initialize the shared model with", type=str, default=None)
    parser.add_argument("--domains", nargs='+', help='A list of domains to use for training', default=[])
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)

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
    bert_config = DistilBertConfig.from_pretrained(bert_model, num_labels=2, output_hidden_states=True)

    # Create the dataset
    all_dsets = [MultiDomainSentimentDataset(
        args.dataset_loc,
        [domain],
        DistilBertTokenizer.from_pretrained(bert_model)
    ) for domain in args.domains]
    train_sizes = [int(len(dset) * args.train_pct) for j, dset in enumerate(all_dsets)]
    val_sizes = [len(all_dsets[j]) - train_sizes[j] for j in range(len(train_sizes))]

    for i in range(len(all_dsets)):
        domain = args.domains[i]

        test_dset = all_dsets[i]

        dataloader = DataLoader(
            test_dset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_batch_transformer
        )

        bert = DistilBertForSequenceClassification.from_pretrained(bert_model, config=bert_config).to(device)
        # Create the model

        model = torch.nn.DataParallel(MultiViewTransformerNetworkAveragingIndividuals(
            bert_model,
            bert_config,
            len(all_dsets) - 1
        )).to(device)
        model.module.average = True
        # load the trained model

        # Load the best weights
        for v in range(len(all_dsets)-1):
            model.module.domain_experts[v].load_state_dict(torch.load(f'{args.pretrained_model}/model_{domain}_{v}.pth'))
        model.module.shared_bert.load_state_dict(torch.load(f'{args.pretrained_model}/model_{domain}_{len(all_dsets)-1}.pth'))

        logits_all = [[] for d in range(len(all_dsets))]
        for batch in tqdm(dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            masks = batch[1]
            labels = batch[2]
            # Testing with random domains to see if any effect
            # domains = torch.tensor(np.random.randint(0, 16, batch[3].shape)).to(device)
            domains = batch[3]

            logits = model(input_ids, attention_mask=masks, domains=domains, labels=labels, return_logits=True)
            for k,l in enumerate(logits):
                logits_all[k].append(l.detach().cpu().numpy())

        print(domain)
        probs_all = [softmax(np.concatenate(l), axis=-1)[:,1] for l in logits_all]
        for i in range(len(probs_all)):
            for j in range(i+1, len(probs_all)):
                print(wasserstein_distance(probs_all[i], probs_all[j]))
        preds_all = np.asarray([np.argmax(np.concatenate(l), axis=-1) for l in logits_all])
        print(f"alpha: {krippendorff.alpha(preds_all, level_of_measurement='nominal')}")
        print()