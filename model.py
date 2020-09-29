import os
from tqdm import tqdm
import torch
from copy import deepcopy
from typing import List
from torch import nn
from transformers import BertForSequenceClassification
from transformers import BertModel
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertModel
import ipdb
import numpy as np
from torch.nn import functional as F
from transformers import PreTrainedTokenizer
from torch.nn import init
from argparse import Namespace


class GradientReversal(torch.autograd.Function):
    """
    Basic layer for doing gradient reversal
    """
    lambd = 1.0
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReversal.lambd * grad_output.neg()

class VanillaBert(nn.Module):
    """
    A really basic wrapper around BERT
    """
    def __init__(self, bert: BertForSequenceClassification, **kwargs):
        super(VanillaBert, self).__init__()

        self.bert = bert

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None):

        return self.bert(input_ids, attention_mask=attention_mask, labels=labels)


class DomainAdversarialBert(nn.Module):
    """
    A really basic wrapper around BERT
    """
    def __init__(self, bert: BertModel, n_domains: int, n_classes: int = 2, supervision_layer=12, **kwargs):
        super(DomainAdversarialBert, self).__init__()

        self.bert = bert
        self.domain_classifier = nn.Linear(bert.config.hidden_size, n_domains)
        self.supervision_layer = supervision_layer

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None):

        # 1) Get the CLS representation from BERT
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        # (b x n_classes)
        classifier_logits = outputs[0]

        divisor = min(1, 2 * (len(outputs[1]) - self.supervision_layer))
        domain_supervision_layer = outputs[1][self.supervision_layer][:, 0, :]
        adv_input = GradientReversal.apply(domain_supervision_layer)

        adv_logits = self.domain_classifier(adv_input)

        outputs = (classifier_logits,)

        loss_fn = nn.CrossEntropyLoss()
        if domains is not None:
            # Scale the adversarial loss depending on how deep in the network it is
            loss = (1e-3 / divisor) * loss_fn(adv_logits, domains)

            if labels is not None:
                loss += loss_fn(classifier_logits, labels)
            outputs = (loss,) + outputs
        elif labels is not None:
            loss = loss_fn(classifier_logits, labels)
            outputs = (loss,) + outputs

        return outputs


########################################
# Multi-Transformer Network
########################################

class TransformerNetwork(nn.Module):
    """
    Multiple transformers for different domains
    """

    def __init__(
            self,
            bert_embeddings,
            ff_dim: int = 2048,
            d_model: int = 768,
            n_domains: int = 2,
            n_layers: int = 6,
            n_classes: int = 2,
            n_heads: int = 6,
    ):
        super(TransformerNetwork, self).__init__()
        self.ff_dim = ff_dim
        self.d_model = d_model
        self.n_domains = n_domains

        self.bert_embeddings = bert_embeddings

        self.xformer =nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                dim_feedforward=ff_dim
            ),
            n_layers
        )

        # final classifier layers (d_model x n_classes)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None
    ):
        embs = self.bert_embeddings(input_ids=input_ids)

        # Sequence length first
        inputs = embs.permute(1, 0, 2)
        # Flags the 0s instead of 1s
        masks = attention_mask == 0


        output = self.xformer(inputs, src_key_padding_mask=masks)
        pooled_output = output[0]
        logits = self.classifier(pooled_output)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, logits)


class TransformerClassifier(nn.Module):
    """
        Multiple transformers for different domains
        """

    def __init__(
            self,
            bert_embeddings,
            ff_dim: int = 2048,
            d_model: int = 768,
            n_layers: int = 6,
            n_heads: int = 6,
            n_classes: int = 2,
            **kwargs
    ):
        super(TransformerClassifier, self).__init__()
        self.ff_dim = ff_dim
        self.d_model = d_model

        self.bert_embeddings = bert_embeddings

        self.xformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                dim_feedforward=ff_dim
            ),
            n_layers
        )

        # final classifier layers (d_model x n_classes)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None
    ):
        embs = self.bert_embeddings(input_ids=input_ids)

        # Sequence length first
        inputs = embs.permute(1, 0, 2)
        # Flags the 0s instead of 1s
        masks = attention_mask == 0

        output = self.xformer(inputs, src_key_padding_mask=masks)
        pooled_output = output[0]
        logits = self.classifier(pooled_output)
        outputs = (logits,)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs = (loss,) + outputs

        return outputs

class MultiTransformer(nn.Module):
    """
        Multiple transformers for different domains
        """

    def __init__(
            self,
            bert_embeddings,
            ff_dim: int = 2048,
            d_model: int = 768,
            n_domains: int = 2,
            n_layers: int = 6,
            n_heads: int = 6,
    ):
        super(MultiTransformer, self).__init__()
        self.ff_dim = ff_dim
        self.d_model = d_model
        self.n_domains = n_domains

        self.bert_embeddings = bert_embeddings

        self.xformer = nn.ModuleList([nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                dim_feedforward=ff_dim
            ),
            n_layers
        ) for d in range(n_domains)])

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None
    ):
        embs = self.bert_embeddings(input_ids=input_ids)

        # Sequence length first
        inputs = embs.permute(1, 0, 2)
        # Flags the 0s instead of 1s
        masks = attention_mask == 0

        if domains is not None:
            domain = domains[0]
            output = self.xformer[domain](inputs, src_key_padding_mask=masks)
            pooled_output = output[0]
            return pooled_output

        else:
            pooled_outputs = []
            for d in range(self.n_domains):
                output = self.xformer[d](inputs, src_key_padding_mask=masks)
                pooled_outputs.append(output[0])
            return pooled_outputs


class MultiTransformerClassifier(nn.Module):
    """
    Multiple transformers for different domains
    """

    def __init__(
            self,
            bert_embeddings,
            ff_dim: int = 2048,
            d_model: int = 768,
            n_domains: int = 2,
            n_layers: int = 6,
            n_classes: int = 2,
            n_heads: int = 6,
    ):
        super(MultiTransformerClassifier, self).__init__()
        self.ff_dim = ff_dim
        self.d_model = d_model
        self.n_domains = n_domains

        self.multi_xformer = MultiTransformer(
            bert_embeddings,
            ff_dim=ff_dim,
            d_model=d_model,
            n_domains=n_domains,
            n_layers=n_layers,
            n_heads=n_heads
        )

        # final classifier layers (d_model x n_classes)
        self.classifier = nn.ModuleList([nn.Linear(d_model, n_classes) for d in range(n_domains)])

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None
    ):

        if domains is not None:
            domain = domains[0]
            pooled_output = self.multi_xformer(input_ids, attention_mask, domains)
            logits = self.classifier[domain](pooled_output)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return (loss, logits)

        else:
            logits_all = []
            pooled_outputs = self.multi_xformer(input_ids, attention_mask)
            for d,po in enumerate(pooled_outputs):
                logits_all.append(self.classifier[d](po))

            loss_fn = nn.CrossEntropyLoss()
            loss = torch.mean(torch.stack([loss_fn(logits, labels) for logits in logits_all]))
            # b x ndom x 2
            votes = torch.stack(logits_all, dim=1)
            self.votes = votes
            # Normalize with softmax
            logits_all = torch.nn.Softmax(dim=-1)(votes)
            logits = torch.mean(logits_all, dim=1)

            return loss, logits


class MultiDistilBert(nn.Module):
    """
    Multiple transformers for different domains
    """

    def __init__(
            self,
            model_name,
            config,
            n_domains: int = 2,
            init_weights: List = None
    ):
        super(MultiDistilBert, self).__init__()
        self.models = nn.ModuleList([DistilBertModel.from_pretrained(model_name, config=config) for d in range(n_domains)])
        if init_weights is not None:
            if 'distilbert' in list(init_weights.keys())[0]:
                init_weights = {k[11:]: v for k,v in init_weights.items()}
            for m in self.models:
                model_dict = m.state_dict()
                model_dict.update(deepcopy(init_weights))
                m.load_state_dict(model_dict)
        self.n_domains = n_domains
        self.d_model = config.hidden_size

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None
    ):

        if domains is not None:
            domain = domains[0]
            outputs = self.models[domain](input_ids, attention_mask=attention_mask)
            return outputs[0][:,0,:]

        else:
            pooled_outputs = []
            for d in range(self.n_domains):
                output = self.models[d](input_ids, attention_mask=attention_mask)
                pooled_outputs.append(output[0][:,0,:])
            return pooled_outputs


class MultiDistilBertClassifier(nn.Module):
    """
    Multiple transformers for different domains
    """

    def __init__(
            self,
            model_name,
            config,
            n_domains: int = 2,
            n_classes: int = 2,
            init_weights: List = None
    ):
        super(MultiDistilBertClassifier, self).__init__()
        self.multi_xformer = MultiDistilBert(model_name, config, n_domains=n_domains, init_weights=init_weights)
        self.n_domains = n_domains
        self.d_model = config.hidden_size
        # final classifier layers (d_model x n_classes)
        self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, n_classes) for d in range(n_domains)])

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None
    ):

        if domains is not None:
            domain = domains[0]
            output = self.multi_xformer(input_ids, attention_mask=attention_mask, domains=domains)
            logits = self.classifier[domain](output)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return (loss, logits)

        else:
            logits_all = []
            pooled_outputs = self.multi_xformer(input_ids, attention_mask=attention_mask)
            for d,po in enumerate(pooled_outputs):
                logits_all.append(self.classifier[d](po))

            loss_fn = nn.CrossEntropyLoss()
            loss = torch.mean(torch.stack([loss_fn(logits, labels) for logits in logits_all]))
            # b x ndom x 2
            votes = torch.stack(logits_all, dim=1)
            self.votes = votes
            # Normalize with softmax
            logits_all = torch.nn.Softmax(dim=-1)(votes)
            logits = torch.mean(logits_all, dim=1)

            return loss, logits


class MultiTransformerNetwork(nn.Module):
    """
    Multiple transformers for different domains
    """

    def __init__(
            self,
            bert_embeddings,
            ff_dim: int = 2048,
            d_model: int = 768,
            n_domains: int = 2,
            n_layers: int = 6,
            n_classes: int = 2,
            n_heads: int = 6,
    ):
        super(MultiTransformerNetwork, self).__init__()
        self.ff_dim = ff_dim
        self.d_model = d_model
        self.n_domains = n_domains

        self.bert_embeddings = bert_embeddings

        self.xformer = nn.ModuleList([nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                dim_feedforward=ff_dim
            ),
            n_layers
        ) for d in range(n_domains)])

        # final classifier layers (d_model x n_classes)
        self.classifier = nn.ModuleList([nn.Linear(d_model, n_classes) for d in range(n_domains)])

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None
    ):
        embs = self.bert_embeddings(input_ids=input_ids)

        # Sequence length first
        inputs = embs.permute(1, 0, 2)
        # Flags the 0s instead of 1s
        masks = attention_mask == 0

        if domains is not None:
            domain = domains[0]
            output = self.xformer[domain](inputs, src_key_padding_mask=masks)
            pooled_output = output[0]
            logits = self.classifier[domain](pooled_output)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return (loss, logits)

        else:
            logits_all = []
            for d in range(self.n_domains):
                output = self.xformer[d](inputs)
                pooled_output = output[0]
                logits_all.append(self.classifier[d](pooled_output))

            loss_fn = nn.CrossEntropyLoss()
            loss = torch.mean(torch.stack([loss_fn(logits, labels) for logits in logits_all]))
            # b x ndom x 2
            votes = torch.stack(logits_all, dim=1)
            self.votes = votes
            # Normalize with softmax
            logits_all = torch.nn.Softmax(dim=-1)(votes)
            logits = torch.sum(logits_all, dim=1)

            return loss, logits


#############################
# Multi-View domain adaptation modules
#############################
class MultiViewTransformerNetwork(nn.Module):
    """
    Multi-view transformer network for domain adaptation
    """
    def __init__(self, multi_xformer: MultiTransformerNetwork, shared_bert: VanillaBert, n_classes: int = 2):
        super(MultiViewTransformerNetwork, self).__init__()

        self.multi_xformer = multi_xformer.multi_xformer
        self.shared_bert = shared_bert.bert.bert

        self.d_model = self.multi_xformer.d_model
        self.dim_param = nn.Parameter(torch.FloatTensor([self.multi_xformer.d_model]), requires_grad=False)
        self.n_domains = self.multi_xformer.n_domains
        self.n_classes = n_classes

        # Query matrix
        self.Q = nn.Parameter(torch.randn((self.multi_xformer.d_model, self.multi_xformer.d_model)), requires_grad=True)
        # Key matrix
        self.K = nn.Parameter(torch.randn((self.multi_xformer.d_model, self.multi_xformer.d_model)), requires_grad=True)
        # Value matrix
        self.V = nn.Parameter(torch.randn((self.multi_xformer.d_model, self.multi_xformer.d_model)), requires_grad=True)

        # Main classifier
        self.task_classifier = nn.Linear(self.multi_xformer.d_model, n_classes)
        # TODO: Introduce aux tasks if needed

        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            ret_alpha: bool = False
    ):
        # Get all the pooled outputs
        # (n_domain) b x dim
        pooled_outputs = self.multi_xformer(input_ids, attention_mask)
        # b x dim
        shared_output = self.shared_bert(input_ids, attention_mask=attention_mask)[1]
        # Values b x n_domain + 1 x dim
        v = torch.stack(pooled_outputs + [shared_output], dim=1)
        # Queries b x dim
        q = shared_output @ self.Q
        # Keys b*(n_domain + 1) x dim
        k = v.view(-1, self.d_model) @ self.K
        # Attention (scaled dot product) b x (n_domain + 1)
        attn = torch.sum(k.view(-1, self.n_domains + 1, self.d_model) * q.unsqueeze(1), dim=-1) / torch.sqrt(self.dim_param)
        attn = nn.Softmax(dim=-1)(attn)
        v = v.view(-1, self.d_model) @ self.V
        v = v.view(-1, self.n_domains + 1, self.d_model)
        # Attend to the values b x dim
        o = torch.sum(attn.unsqueeze(-1) * v, dim=1)

        # Classifier
        logits = self.task_classifier(o)
        outputs = (logits,)
        if labels is not None:
            # Loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs = (loss,) + outputs
        if ret_alpha:
            outputs += (attn,)
        return outputs


class MultiViewTransformerNetworkAveraging(nn.Module):
    """
    Multi-view transformer network for domain adaptation
    """

    def __init__(self, multi_xformer: MultiTransformerClassifier, shared_bert: VanillaBert, n_classes: int = 2):
        super(MultiViewTransformerNetworkAveraging, self).__init__()

        self.multi_xformer = multi_xformer.multi_xformer
        self.multi_xformer_classifiers = multi_xformer.classifier
        self.shared_bert = shared_bert.bert

        self.d_model = multi_xformer.d_model
        self.dim_param = nn.Parameter(torch.FloatTensor([multi_xformer.d_model]), requires_grad=False)
        self.n_domains = multi_xformer.n_domains
        self.n_classes = n_classes

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            ret_alpha: bool = False
    ):
        # Get all the pooled outputs
        # (n_domain) b x dim
        pooled_outputs = self.multi_xformer(input_ids, attention_mask)
        # b x dim
        outputs = self.shared_bert(input_ids, attention_mask=attention_mask)
        logits_shared = outputs[0]

        softmax = nn.Softmax()
        logits_private = [self.multi_xformer_classifiers[d](pooled_outputs[d]) for d in range(self.n_domains)]

        if domains is not None and self.training:
            logits = [l for j, l in enumerate(logits_private) if j != domains[0]] + [logits_shared]
        else:
            logits = logits_private + [logits_shared]
        attn = 1 / len(logits)

        # b x n_dom(+1) x nclasses
        preds = torch.stack([softmax(logs) for logs in logits], dim=1)
        # Apply attention
        preds = torch.sum(preds * attn, dim=1)
        outputs = (preds,)
        if labels is not None:
            # LogSoftmax + NLLLoss
            loss_fn = nn.NLLLoss()
            loss = (0.5) * loss_fn(torch.log(preds), labels)
            # Strong supervision on in domain
            if domains is not None:
                domain = domains[0]
                domain_logits = logits_private[domain]
                xent = nn.CrossEntropyLoss()
                loss += (0.5) * xent(domain_logits, labels)

            outputs = (loss,) + outputs
        if ret_alpha:
            outputs += (torch.cuda.FloatTensor([attn]).expand_as(preds),)
        return outputs


class MultiViewTransformerNetworkLearnedAveraging(nn.Module):
    """
    Multi-view transformer network for domain adaptation
    """

    def __init__(self, multi_xformer: MultiTransformerClassifier, shared_bert: VanillaBert, n_classes: int = 2):
        super(MultiViewTransformerNetworkLearnedAveraging, self).__init__()

        self.multi_xformer = multi_xformer.multi_xformer
        self.multi_xformer_classifiers = multi_xformer.classifier
        self.shared_bert = shared_bert.bert

        self.d_model = multi_xformer.d_model
        self.dim_param = nn.Parameter(torch.FloatTensor([multi_xformer.d_model]), requires_grad=False)
        self.n_domains = multi_xformer.n_domains
        self.n_classes = n_classes
        self.alpha_params = nn.Parameter(torch.ones(self.n_domains + 1))

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            ret_alpha: bool = False
    ):
        # Get all the pooled outputs
        # (n_domain) b x dim
        pooled_outputs = self.multi_xformer(input_ids, attention_mask)
        # b x dim
        outputs = self.shared_bert(input_ids, attention_mask=attention_mask)
        logits_shared = outputs[0]

        softmax = nn.Softmax(dim=-1)
        logits_private = [self.multi_xformer_classifiers[d](pooled_outputs[d]) for d in range(self.n_domains)]

        if domains is not None and self.training:
            logits = [l for j, l in enumerate(logits_private) if j != domains[0]] + [logits_shared]
            indices = [j for j, l in enumerate(logits_private) if j != domains[0]] + [self.n_domains]
            alpha_weights = torch.gather(self.alpha_params, 0, torch.cuda.LongTensor(indices))
            attn = softmax(alpha_weights).unsqueeze(0).unsqueeze(2)
        else:
            logits = logits_private + [logits_shared]
            alpha_weights = self.alpha_params
            attn = softmax(alpha_weights).unsqueeze(0).unsqueeze(2)

        # b x n_dom(+1) x nclasses
        preds = torch.stack([softmax(logs) for logs in logits], dim=1)
        # Apply attention
        preds = torch.sum(preds * attn, dim=1)
        outputs = (preds,)
        if labels is not None:
            # LogSoftmax + NLLLoss
            loss_fn = nn.NLLLoss()
            loss = (0.5) * loss_fn(torch.log(preds), labels)
            # Strong supervision on in domain
            if domains is not None:
                domain = domains[0]
                domain_logits = logits_private[domain]
                xent = nn.CrossEntropyLoss()
                loss += (0.5) * xent(domain_logits, labels)

            outputs = (loss,) + outputs
        if ret_alpha:
            outputs += (attn,)
        return outputs


class MultiViewTransformerNetworkAveragingIndividuals(nn.Module):
    """
    Multi-view transformer network for domain adaptation
    """

    def __init__(self, bert_model, bert_config, n_domains: int = 2, n_classes: int = 2):
        super(MultiViewTransformerNetworkAveragingIndividuals, self).__init__()

        self.domain_experts = nn.ModuleList([DistilBertForSequenceClassification.from_pretrained(bert_model, config=bert_config)]*n_domains)
        self.shared_bert = DistilBertForSequenceClassification.from_pretrained(bert_model, config=bert_config)

        self.n_domains = n_domains
        self.n_classes = n_classes

        # Default weight is averaging
        self.weights = [1. / (self.n_domains + 1)] * (self.n_domains + 1)

        self.average = False

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            return_logits: bool = False
    ):

        logits_shared = self.shared_bert(input_ids, attention_mask=attention_mask)[0]

        softmax = nn.Softmax()

        if not self.average:
            if domains is not None:
                logits = self.domain_experts[domains[0]](input_ids, attention_mask=attention_mask)[0]
                # b x n_dom(+1) x nclasses
                preds = softmax(logits)
            else:
                logits = logits_shared
                # b x n_dom(+1) x nclasses
                preds = softmax(logits)
        else:
            logits_private = [self.domain_experts[d](input_ids, attention_mask=attention_mask)[0] for d in
                              range(self.n_domains)]
            logits = logits_private + [logits_shared]
            if return_logits:
                return logits
            attn = torch.cuda.FloatTensor(self.weights).view(1, -1, 1)
            # b x n_dom(+1) x nclasses
            preds = torch.stack([softmax(logs) for logs in logits], dim=1)
            # Apply attention
            preds = torch.sum(preds * attn, dim=1)

        outputs = (preds,)
        if labels is not None:
            # LogSoftmax + NLLLoss
            loss_fn = nn.NLLLoss()
            xent = nn.CrossEntropyLoss()
            loss = loss_fn(torch.log(preds), labels) + xent(logits_shared, labels)

            outputs = (loss,) + outputs
        return outputs


class MultiViewTransformerNetworkDomainClassifierIndividuals(nn.Module):
    """
    Multi-view transformer network for domain adaptation
    """

    def __init__(self, bert_model, bert_config, domain_classifier, n_domains: int = 2, n_classes: int = 2):
        super(MultiViewTransformerNetworkDomainClassifierIndividuals, self).__init__()

        self.domain_experts = nn.ModuleList([DistilBertForSequenceClassification.from_pretrained(bert_model, config=bert_config)]*n_domains)
        self.domain_classifier = domain_classifier

        self.n_domains = n_domains
        self.n_classes = n_classes

        self.average = False

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None
    ):

        softmax = nn.Softmax(dim=-1)
        if not self.average:
            logits = self.domain_experts[domains[0]](input_ids, attention_mask=attention_mask)[0]
            # b x n_dom(+1) x nclasses
            preds = softmax(logits)
        else:
            logits = [self.domain_experts[d](input_ids, attention_mask=attention_mask)[0] for d in
                              range(self.n_domains)]
            logits_shared = self.domain_classifier(input_ids, attention_mask=attention_mask)[0]
            attn = nn.Softmax(dim=-1)(logits_shared).unsqueeze(-1)
            # b x n_dom(+1) x nclasses
            preds = torch.stack([softmax(logs) for logs in logits], dim=1)
            # Apply attention
            preds = torch.sum(preds * attn, dim=1)

        outputs = (preds,)
        if labels is not None:
            # LogSoftmax + NLLLoss
            loss_fn = nn.NLLLoss()
            xent = nn.CrossEntropyLoss()
            loss = loss_fn(torch.log(preds), labels)

            outputs = (loss,) + outputs
        return outputs


class MultiViewTransformerNetworkSelectiveWeight(nn.Module):
    """
    Multi-view transformer network for domain adaptation
    """

    def __init__(self, multi_xformer: MultiTransformerClassifier, shared_bert: VanillaBert, n_classes: int = 2):
        super(MultiViewTransformerNetworkSelectiveWeight, self).__init__()

        self.multi_xformer = multi_xformer.multi_xformer
        self.multi_xformer_classifiers = multi_xformer.classifier
        self.shared_bert = shared_bert.bert

        self.d_model = multi_xformer.d_model
        self.dim_param = nn.Parameter(torch.FloatTensor([multi_xformer.d_model]), requires_grad=False)
        self.n_domains = multi_xformer.n_domains
        self.n_classes = n_classes
        # Default weight is averaging
        self.weights = [1./(self.n_domains + 1)] * (self.n_domains + 1)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            ret_alpha: bool = False
    ):
        # Get all the pooled outputs
        # (n_domain) b x dim
        pooled_outputs = self.multi_xformer(input_ids, attention_mask)
        # b x dim
        outputs = self.shared_bert(input_ids, attention_mask=attention_mask)
        logits_shared = outputs[0]

        softmax = nn.Softmax()
        logits_private = [self.multi_xformer_classifiers[d](pooled_outputs[d]) for d in range(self.n_domains)]

        if domains is not None and self.training:
            logits = [l for j, l in enumerate(logits_private) if j != domains[0]] + [logits_shared]
            attn = 1 / len(logits)
        else:
            logits = logits_private + [logits_shared]
            attn = torch.cuda.FloatTensor(self.weights).view(1,-1,1)

        # b x n_dom(+1) x nclasses
        preds = torch.stack([softmax(logs) for logs in logits], dim=1)
        # Apply attention
        preds = torch.sum(preds * attn, dim=1)
        outputs = (preds,)
        if labels is not None:
            # LogSoftmax + NLLLoss
            loss_fn = nn.NLLLoss()
            loss = (0.5) * loss_fn(torch.log(preds), labels)
            # Strong supervision on in domain
            if domains is not None:
                domain = domains[0]
                domain_logits = logits_private[domain]
                xent = nn.CrossEntropyLoss()
                loss += (0.5) * xent(domain_logits, labels)

            outputs = (loss,) + outputs
        if ret_alpha:
            outputs += (torch.cuda.FloatTensor([attn]).expand_as(preds),)
        return outputs


class MultiViewTransformerNetworkProbabilities(nn.Module):
    """
    Multi-view transformer network for domain adaptation
    """

    def __init__(self, multi_xformer: MultiTransformerClassifier, shared_bert: VanillaBert, n_classes: int = 2):
        super(MultiViewTransformerNetworkProbabilities, self).__init__()

        self.multi_xformer = multi_xformer.multi_xformer
        self.multi_xformer_classifiers = multi_xformer.classifier
        self.shared_bert = shared_bert.bert

        self.d_model = multi_xformer.d_model
        self.dim_param = nn.Parameter(torch.FloatTensor([multi_xformer.d_model]), requires_grad=False)
        self.n_domains = multi_xformer.n_domains
        self.n_classes = n_classes

        # Query matrix
        self.Q = nn.Parameter(torch.randn((multi_xformer.d_model, multi_xformer.d_model)), requires_grad=True)
        # Key matrix
        self.K = nn.Parameter(torch.randn((multi_xformer.d_model, multi_xformer.d_model)), requires_grad=True)
        # Value matrix
        self.V = nn.Parameter(torch.randn((multi_xformer.d_model, multi_xformer.d_model)), requires_grad=True)

        # Main classifier
        #self.task_classifier = nn.Linear(multi_xformer.d_model, n_classes)
        # TODO: Introduce aux tasks if needed

        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            ret_alpha: bool = False
    ):
        # Get all the pooled outputs
        # (n_domain) b x dim
        pooled_outputs = self.multi_xformer(input_ids, attention_mask)
        # b x dim
        outputs = self.shared_bert(input_ids, attention_mask=attention_mask)
        shared_output = outputs[1][-1][:,0,:]
        logits_shared = outputs[0]

        # Values b x n_domain (+ 1) x dim
        if domains is not None and self.training:
            attend_values = [p for j, p in enumerate(pooled_outputs) if j != domains[0]]
            v = torch.stack(attend_values + [shared_output], dim=1)
        else:
            v = torch.stack(pooled_outputs + [shared_output], dim=1)

        # Queries b x dim
        q = shared_output @ self.Q
        # Keys b*(n_domain + 1) x dim
        k = v.view(-1, self.d_model) @ self.K
        # Attention (scaled dot product) b x (n_domain + 1)
        attn = torch.sum(k.view(-1, v.shape[1], self.d_model) * q.unsqueeze(1), dim=-1) / torch.sqrt(
            self.dim_param)
        attn = nn.Softmax(dim=-1)(attn)

        softmax = nn.Softmax()
        logits_private = [self.multi_xformer_classifiers[d](pooled_outputs[d]) for d in range(self.n_domains)]

        if domains is not None and self.training:
            logits = [l for j, l in enumerate(logits_private) if j != domains[0]] + [logits_shared]
        else:
            logits = logits_private + [logits_shared]

        # b x n_dom(+1) x nclasses
        preds = torch.stack([softmax(logs) for logs in logits], dim=1)
        # Apply attention
        preds = torch.sum(preds * attn.unsqueeze(-1), dim=1)
        outputs = (preds,)
        if labels is not None:
            # LogSoftmax + NLLLoss
            loss_fn = nn.NLLLoss()
            loss = (0.5) * loss_fn(torch.log(preds), labels)
            # Strong supervision on in domain
            if domains is not None:
                domain = domains[0]
                domain_logits = logits_private[domain]
                xent = nn.CrossEntropyLoss()
                loss += (0.5) * xent(domain_logits, labels)

            outputs = (loss,) + outputs
        if ret_alpha:
            outputs += (attn,)
        return outputs


class MultiViewTransformerNetworkProbabilitiesAdversarial(nn.Module):
    """
    Multi-view transformer network for domain adaptation
    """

    def __init__(self, multi_xformer: MultiTransformerClassifier, shared_bert: VanillaBert, n_classes: int = 2, supervision_layer: int = 12):
        super(MultiViewTransformerNetworkProbabilitiesAdversarial, self).__init__()

        self.multi_xformer = multi_xformer.multi_xformer
        self.multi_xformer_classifiers = multi_xformer.classifier
        self.shared_bert = shared_bert.bert
        self.d_model = multi_xformer.d_model

        # Add one extra for the target data
        self.domain_classifier = nn.Linear(self.d_model, multi_xformer.n_domains + 1)
        self.supervision_layer = supervision_layer

        self.dim_param = nn.Parameter(torch.FloatTensor([multi_xformer.d_model]), requires_grad=False)
        self.n_domains = multi_xformer.n_domains
        self.n_classes = n_classes

        # Query matrix
        self.Q = nn.Parameter(torch.randn((multi_xformer.d_model, multi_xformer.d_model)), requires_grad=True)
        # Key matrix
        self.K = nn.Parameter(torch.randn((multi_xformer.d_model, multi_xformer.d_model)), requires_grad=True)
        # Value matrix
        self.V = nn.Parameter(torch.randn((multi_xformer.d_model, multi_xformer.d_model)), requires_grad=True)

        # Main classifier
        #self.task_classifier = nn.Linear(multi_xformer.d_model, n_classes)
        # TODO: Introduce aux tasks if needed

        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            ret_alpha: bool = False
    ):
        # Get all the pooled outputs
        # (n_domain) b x dim
        pooled_outputs = self.multi_xformer(input_ids, attention_mask)
        # b x dim
        outputs = self.shared_bert(input_ids, attention_mask=attention_mask)
        divisor = min(1, 2 * (len(outputs[1]) - self.supervision_layer))
        shared_output = outputs[1][-1][:,0,:]
        logits_shared = outputs[0]

        # Domain adversarial bit
        domain_supervision_layer = outputs[1][self.supervision_layer][:,0,:]
        adv_input = GradientReversal.apply(domain_supervision_layer)
        adv_logits = self.domain_classifier(adv_input)

        # Values b x n_domain (+ 1) x dim
        if domains is not None and self.training:
            attend_values = [p for j, p in enumerate(pooled_outputs) if j != domains[0]]
            v = torch.stack(attend_values + [shared_output], dim=1)
        else:
            v = torch.stack(pooled_outputs + [shared_output], dim=1)
        # Queries b x dim
        q = shared_output @ self.Q
        # Keys b*(n_domain + 1) x dim
        k = v.view(-1, self.d_model) @ self.K
        # Attention (scaled dot product) b x (n_domain + 1)
        attn = torch.sum(k.view(-1, v.shape[1], self.d_model) * q.unsqueeze(1), dim=-1) / torch.sqrt(
            self.dim_param)
        attn = nn.Softmax(dim=-1)(attn)

        softmax = nn.Softmax()
        logits_private = [self.multi_xformer_classifiers[d](pooled_outputs[d]) for d in range(self.n_domains)]
        if domains is not None and self.training:
            logits = [l for j, l in enumerate(logits_private) if j != domains[0]] + [logits_shared]
        else:
            logits = logits_private + [logits_shared]

        # b x n_dom+1 x nclasses
        preds = torch.stack([softmax(logs) for logs in logits], dim=1)
        # Apply attention
        preds = torch.sum(preds * attn.unsqueeze(-1), dim=1)
        outputs = (preds,)
        if labels is not None:
            # LogSoftmax + NLLLoss
            loss_fn = nn.NLLLoss()
            loss = 0.5*loss_fn(torch.log(preds), labels)
            if domains is not None:
                domain = domains[0]
                xent = nn.CrossEntropyLoss()
                domain_logits = logits_private[domain]
                loss += 0.5*xent(domain_logits, labels)
                # Scale the adversarial loss depending on how deep in the network it is
                loss += (1e-3 / divisor) * xent(adv_logits, domains)

            outputs = (loss,) + outputs
        # For unsupervised adversarial loss
        elif domains is not None:
            domain = domains[0]
            xent = nn.CrossEntropyLoss()
            # Scale the adversarial loss depending on how deep in the network it is
            loss = (1e-3 / divisor) * xent(adv_logits, domains)
            outputs = (loss,) + outputs

        if ret_alpha:
            outputs += (attn,)
        return outputs


class MultiViewTransformerNetworkDomainClassifierAttention(nn.Module):
    """
    Multi-view transformer network for domain adaptation
    """

    def __init__(self, multi_xformer: MultiTransformerClassifier, shared_bert: VanillaBert, n_classes: int = 2):
        super(MultiViewTransformerNetworkDomainClassifierAttention, self).__init__()

        self.multi_xformer = multi_xformer.multi_xformer
        self.multi_xformer_classifiers = multi_xformer.classifier
        self.shared_bert = shared_bert

        self.d_model = multi_xformer.d_model
        self.dim_param = nn.Parameter(torch.FloatTensor([multi_xformer.d_model]), requires_grad=False)
        self.n_domains = multi_xformer.n_domains
        self.n_classes = n_classes

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            ret_alpha: bool = False
    ):
        # Get all the pooled outputs
        # (n_domain) b x dim
        pooled_outputs = self.multi_xformer(input_ids, attention_mask)
        # b x n_domains
        logits_shared = self.shared_bert(input_ids, attention_mask=attention_mask)[0]
        attn = nn.Softmax(dim=-1)(logits_shared)

        softmax = nn.Softmax(dim=-1)
        logits = [self.multi_xformer_classifiers[d](pooled_outputs[d]) for d in range(self.n_domains)]

        # attend to classifiers based on the output of the domain classifier
        # b x n_dom x nclasses
        preds = torch.stack([softmax(logs) for logs in logits], dim=1)
        # Apply attention
        preds = torch.sum(preds * attn.unsqueeze(-1), dim=1)

        outputs = (preds,)
        if labels is not None:
            # LogSoftmax + NLLLoss
            loss_fn = nn.NLLLoss()
            loss = loss_fn(torch.log(preds), labels)

            outputs = (loss,) + outputs
        if ret_alpha:
            outputs += (attn,)
        return outputs


class MultiViewTransformerNetworkDomainAdversarial(nn.Module):
    """
    Multi-view transformer network for domain adaptation
    """
    def __init__(self, multi_xformer: MultiTransformerClassifier, shared_bert: VanillaBert, n_classes: int = 2, n_domains: int = 3, supervision_layer: int = 12):
        super(MultiViewTransformerNetworkDomainAdversarial, self).__init__()

        self.multi_xformer = multi_xformer.multi_xformer
        self.shared_bert = shared_bert.bert.bert

        self.d_model = multi_xformer.d_model
        self.dim_param = nn.Parameter(torch.FloatTensor([multi_xformer.d_model]), requires_grad=False)
        self.n_xformers = multi_xformer.n_domains
        self.n_classes = n_classes
        self.n_domains = n_domains
        self.supervision_layer = supervision_layer

        # Query matrix
        self.Q = nn.Parameter(torch.randn((multi_xformer.d_model, multi_xformer.d_model)), requires_grad=True)
        # Key matrix
        self.K = nn.Parameter(torch.randn((multi_xformer.d_model, multi_xformer.d_model)), requires_grad=True)
        # Value matrix
        self.V = nn.Parameter(torch.randn((multi_xformer.d_model, multi_xformer.d_model)), requires_grad=True)

        # Main classifier
        self.domain_classifier = nn.Linear(self.d_model, n_domains)
        self.task_classifier = nn.Linear(multi_xformer.d_model, n_classes)
        # TODO: Introduce aux tasks if needed

        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            ret_alpha: bool = False
    ):
        # Get all the pooled outputs
        # (n_domain) b x dim
        pooled_outputs = self.multi_xformer(input_ids, attention_mask)
        # b x dim
        outputs = self.shared_bert(input_ids, attention_mask=attention_mask)
        shared_output = outputs[1][:,0,:]
        divisor = min(1, 2 * (len(outputs[1]) - self.supervision_layer))

        # Values b x n_domain + 1 x dim
        v = torch.stack(pooled_outputs + [shared_output], dim=1)
        # Queries b x dim
        q = shared_output @ self.Q
        # Keys b*(n_domain + 1) x dim
        k = v.view(-1, self.d_model) @ self.K
        # Attention (scaled dot product) b x (n_domain + 1)
        attn = torch.sum(k.view(-1, self.n_xformers + 1, self.d_model) * q.unsqueeze(1), dim=-1) / torch.sqrt(self.dim_param)
        attn = nn.Softmax(dim=-1)(attn)
        v = v.view(-1, self.d_model) @ self.V
        v = v.view(-1, self.n_xformers + 1, self.d_model)
        # Attend to the values b x dim
        o = torch.sum(attn.unsqueeze(-1) * v, dim=1)

        # Classifier
        logits = self.task_classifier(o)

        # Domain adversarial bit
        domain_supervision_layer = outputs[2][self.supervision_layer][:,0,:].squeeze()
        adv_input = GradientReversal.apply(domain_supervision_layer)
        adv_logits = self.domain_classifier(adv_input)

        outputs = (logits,)

        loss_fn = nn.CrossEntropyLoss()
        if domains is not None:
            # Scale the adversarial loss depending on how deep in the network it is
            loss = (1e-3 / divisor) * loss_fn(adv_logits, domains)
            if labels is not None:
                loss += loss_fn(logits, labels)
            outputs = (loss,) + outputs
        elif labels is not None:
            loss = loss_fn(logits, labels)
            outputs = (loss,) + outputs
        if ret_alpha:
            outputs += (attn,)

        return outputs


class MultiViewCNNAveragingIndividuals(nn.Module):
    """
    Multi-view transformer network for domain adaptation
    """

    def __init__(self, args: Namespace, embeddings: np.array, n_domains: int = 2, n_classes: int = 2):
        super(MultiViewCNNAveragingIndividuals, self).__init__()

        self.domain_experts = nn.ModuleList([NLICNN(embeddings, args, n_classes)]*n_domains)
        self.shared_model = NLICNN(embeddings, args, n_classes)

        self.n_domains = n_domains
        self.n_classes = n_classes

        # Default weight is averaging
        self.weights = [1. / (self.n_domains + 1)] * (self.n_domains + 1)

        self.average = False

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            return_logits: bool = False
    ):

        logits_shared = self.shared_model(input_ids, attention_mask=attention_mask)

        softmax = nn.Softmax()

        if not self.average:
            if domains is not None:
                logits = self.domain_experts[domains[0]](input_ids, attention_mask=attention_mask)
                # b x n_dom(+1) x nclasses
                preds = softmax(logits)
            else:
                logits = logits_shared
                # b x n_dom(+1) x nclasses
                preds = softmax(logits)
        else:
            logits_private = [self.domain_experts[d](input_ids, attention_mask=attention_mask) for d in
                              range(self.n_domains)]
            logits = logits_private + [logits_shared]
            if return_logits:
                return logits
            attn = torch.cuda.FloatTensor(self.weights).view(1, -1, 1)
            # b x n_dom(+1) x nclasses
            preds = torch.stack([softmax(logs) for logs in logits], dim=1)
            # Apply attention
            preds = torch.sum(preds * attn, dim=1)

        outputs = (preds,)
        if labels is not None:
            # LogSoftmax + NLLLoss
            loss_fn = nn.NLLLoss()
            xent = nn.CrossEntropyLoss()
            loss = loss_fn(torch.log(preds), labels) + xent(logits_shared, labels)

            outputs = (loss,) + outputs
        return outputs


_glove_path = "glove.6B.{}d.txt".format


def _get_glove_embeddings(embedding_dim: int, glove_dir: str):
    word_to_index = {}
    word_vectors = []

    with open(os.path.join(glove_dir, _glove_path(embedding_dim))) as fp:
        for line in tqdm(fp.readlines(), desc=f'Loading Glove embeddings {_glove_path}'):
            line = line.split(" ")

            word = line[0]
            word_to_index[word] = len(word_to_index)

            vec = np.array([float(x) for x in line[1:]])
            word_vectors.append(vec)

    return word_to_index, word_vectors


def get_embeddings(embedding_dim: int, embedding_dir: str, tokenizer: PreTrainedTokenizer):
    """
    :return: a tensor with the embedding matrix - ids of words are from vocab
    """
    word_to_index, word_vectors = _get_glove_embeddings(embedding_dim, embedding_dir)

    embedding_matrix = np.zeros((len(tokenizer), embedding_dim))

    for id in range(0, max(tokenizer.vocab.values())+1):
        word = tokenizer.ids_to_tokens[id]
        if word not in word_to_index:
            word_vector = np.random.rand(embedding_dim)
        else:
            word_vector = word_vectors[word_to_index[word]]

        embedding_matrix[id] = word_vector

    return torch.nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float), requires_grad=True)


class NLICNN(torch.nn.Module):
    def __init__(self, embeddings: np.array, args: Namespace, n_labels: int):
        super(NLICNN, self).__init__()
        self.args = args

        self.embedding = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.weight = torch.nn.Parameter(torch.tensor(embeddings, dtype=torch.float), requires_grad=True)

        self.dropout = torch.nn.Dropout(args.dropout)

        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(args.in_channels, args.out_channels,
                                                    (kernel_height, embeddings.shape[1]),
                                                    args.stride, args.padding)
                            for kernel_height in args.kernel_heights])

        output_units = n_labels #if n_labels > 2 else 1
        self.final = torch.nn.Linear(len(args.kernel_heights) * args.out_channels, output_units)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)

        return max_out

    def forward(self, input, attention_mask):
        input = self.embedding(input) * attention_mask.unsqueeze(-1) # Zero out padding
        # input.size() = (batch_size, num_seq, embedding_length)
        input = input.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        input = self.dropout(input)

        conv_out = [self.conv_block(input, self.conv_layers[i]) for i in range(len(self.conv_layers))]
        all_out = torch.cat(conv_out, 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        output = self.final(fc_in)

        return output

