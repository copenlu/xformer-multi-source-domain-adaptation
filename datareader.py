from xml.dom import minidom
from typing import AnyStr
from typing import List
from typing import Tuple
import unicodedata
import pandas as pd
import json
import glob
import ipdb

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


domain_map = {
    'gourmet_food': 0,
    'jewelry_&_watches': 1,
    'outdoor_living': 2,
    'grocery': 3,
    'computer_&_video_games': 4,
    'beauty': 5,
    'baby': 6,
    'software': 7,
    'magazines': 8,
    'camera_&_photo': 9,
    'music': 10,
    'video': 11,
    'health_&_personal_care': 12,
    'toys_&_games': 13,
    'sports_&_outdoors': 14,
    'apparel': 15,
    'books': 16,
    'kitchen_&_housewares': 17,
    'electronics': 18,
    'dvd': 19
}

twitter_domain_map = {
    'charliehebdo': 0,
    'ferguson': 1,
    'germanwings-crash': 2,
    'ottawashooting': 3,
    'sydneysiege': 4,
    'health': 5
}

def text_to_batch_transformer(text: List, tokenizer: PreTrainedTokenizer, text_pair: AnyStr = None) -> Tuple[List, List]:
    """Turn a piece of text into a batch for transformer model

    :param text: The text to tokenize and encode
    :param tokenizer: The tokenizer to use
    :param: text_pair: An optional second string (for multiple sentence sequences)
    :return: A list of IDs and a mask
    """
    if text_pair is None:
        input_ids = [tokenizer.encode(t, add_special_tokens=True, max_length=tokenizer.max_len) for t in text]
    else:
        input_ids = [tokenizer.encode(t, text_pair=p, add_special_tokens=True, max_length=tokenizer.max_len) for t,p in zip(text, text_pair)]

    masks = [[1] * len(i) for i in input_ids]

    return input_ids, masks


def collate_batch_transformer(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = [i[0][0] for i in input_data]
    masks = [i[1][0] for i in input_data]
    labels = [i[2] for i in input_data]
    domains = [i[3] for i in input_data]

    max_length = max([len(i) for i in input_ids])

    input_ids = [(i + [0] * (max_length - len(i))) for i in input_ids]
    masks = [(m + [0] * (max_length - len(m))) for m in masks]

    assert (all(len(i) == max_length for i in input_ids))
    assert (all(len(m) == max_length for m in masks))
    return torch.tensor(input_ids), torch.tensor(masks), torch.tensor(labels), torch.tensor(domains)


def collate_batch_transformer_with_index(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List]:
    return collate_batch_transformer(input_data) + ([i[-1] for i in input_data],)


def read_xml(dir: AnyStr, domain: AnyStr, split: AnyStr = 'positive'):
    """ Convert all of the ratings in amazon product XML file to dicts

    :param xml_file: The XML file to convert to a dict
    :return: All of the rows in the xml file as dicts
    """
    reviews = []
    split_map = {'positive': 1, 'negative': 0, 'unlabelled': -1}
    in_review_text = False
    with open(f'{dir}/{domain}/{split}.review', encoding='utf8', errors='ignore') as f:
        for line in f:
            if '<review_text>' in line:
                reviews.append({'text': '', 'label': split_map[split], 'domain': domain_map[domain]})
                in_review_text = True
                continue
            if '</review_text>' in line:
                in_review_text = False
                reviews[-1]['text'] = reviews[-1]['text'].replace('\n', ' ').strip()
            if in_review_text:
                reviews[-1]['text'] += line
    return reviews


class MultiDomainSentimentDataset(Dataset):
    """
    Implements a dataset for the multidomain sentiment analysis dataset
    """
    def __init__(
            self,
            dataset_dir: AnyStr,
            domains: List,
            tokenizer: PreTrainedTokenizer,
            domain_ids: List = None
    ):
        """

        :param dataset_dir: The base directory for the dataset
        :param domains: The set of domains to load data for
        :param: tokenizer: The tokenizer to use
        :param: domain_ids: A list of ids to override the default domain IDs
        """
        super(MultiDomainSentimentDataset, self).__init__()
        data = []
        for domain in domains:
            data.extend(read_xml(dataset_dir, domain, 'positive'))
            data.extend(read_xml(dataset_dir, domain, 'negative'))

        self.dataset = pd.DataFrame(data)
        if domain_ids is not None:
            for i in range(len(domain_ids)):
                data[data['domain'] == domain_map[domains[i]]][2] = domain_ids[i]
        self.tokenizer = tokenizer

    def set_domain_id(self, domain_id):
        """
        Overrides the domain ID for all data
        :param domain_id:
        :return:
        """
        self.dataset['domain'] = domain_id

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item) -> Tuple:
        row = self.dataset.values[item]
        input_ids, mask = text_to_batch_transformer([row[0]], self.tokenizer)
        label = row[1]
        domain = row[2]
        return input_ids, mask, label, domain, item


class MultiDomainTwitterDataset(Dataset):
    """
    Implements a dataset for the multidomain sentiment analysis dataset
    """
    def __init__(
            self,
            dataset_dir: AnyStr,
            domains: List,
            tokenizer: PreTrainedTokenizer,
            health_data_loc: AnyStr = None,
            domain_ids: List = None
    ):
        """

        :param dataset_dir: The base directory for the dataset
        :param domains: The set of domains to load data for
        :param: tokenizer: The tokenizer to use
        :param: domain_ids: A list of ids to override the default domain IDs
        """
        super(MultiDomainTwitterDataset, self).__init__()
        rumours = []
        non_rumours = []
        d_ids = []
        self.name = "_".join(domains)
        for domain in domains:
            if domain != 'health':
                for source_tweet_file in glob.glob(f'{dataset_dir}/{domain}-all-rnr-threads/non-rumours/**/source-tweets/*.json'):
                    with open(source_tweet_file) as js:
                        tweet = json.load(js)
                    non_rumours.append(tweet['text'])
                    d_ids.append(twitter_domain_map[domain])
                for source_tweet_file in glob.glob(f'{dataset_dir}/{domain}-all-rnr-threads/rumours/**/source-tweets/*.json'):
                    with open(source_tweet_file) as js:
                        tweet = json.load(js)
                        rumours.append(tweet['text'])
                        d_ids.append(twitter_domain_map[domain])
            elif health_data_loc is not None:
                health_dataset = pd.read_csv(health_data_loc, sep="\t", header=None)
                # Remove unknowns
                health_dataset = health_dataset[health_dataset[1] != 0]
                # Transform the text
                health_dataset[0] = health_dataset[0].apply(lambda x: x[10:] if 'RT @xxxxx ' == x[:10] else x)
                # Drop duplicates
                health_dataset = health_dataset.drop_duplicates()
                statements = [v[0] for v in health_dataset.values]
                lblmap = {1: 0, -1: 1}
                labels = [lblmap[v[1]] for v in health_dataset.values]
                rumours.extend([s for s,l in zip(statements, labels) if l == 1])
                non_rumours.extend([s for s,l in zip(statements, labels) if l == 0])
                d_ids.extend([twitter_domain_map[domain]] * len(labels))


        self.dataset = pd.DataFrame(rumours + non_rumours, columns=['statement'])
        self.dataset['label'] = [1] * len(rumours) + [0] * len(non_rumours)
        self.dataset['statement'] = self.dataset['statement'].str.normalize('NFKD')
        self.dataset['domain'] = d_ids

        self.tokenizer = tokenizer

    def set_domain_id(self, domain_id):
        """
        Overrides the domain ID for all data
        :param domain_id:
        :return:
        """
        self.dataset['domain'] = domain_id

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item) -> Tuple:
        row = self.dataset.values[item]
        input_ids, mask = text_to_batch_transformer([row[0]], self.tokenizer)
        label = row[1]
        domain = row[2]
        return input_ids, mask, label, domain, item


