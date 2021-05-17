import csv
import copy
import os
import sys
from random import shuffle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
import json

tokenizer = BertTokenizer.from_pretrained("../datasets/bert-base-uncased")
token_table = {'ecommerce': 'electronic commerce'}


def load_data(data_config, data_path=None, data_type='allData', use_previousData=False):
    cache_file_head = data_config['method_str'] #data_path.split("/")[-1]

    if use_previousData:

        print("load dataset from cache")
        dataset = dataEngine.from_dict(torch.load(os.path.join('cache', cache_file_head + '.dataset')))
        encoded_tag, tag_mask = torch.load(os.path.join('cache', cache_file_head + '.encoded_tag')), \
                                torch.load(os.path.join('cache', cache_file_head + '.tag_mask'))

    else:
        print("build dataset")
        if not os.path.exists('cache'):
            os.makedirs('cache')

        dataset = dataEngine(data_config=data_config)

        if data_type == 'All':

            data = dataset.load_allData(data_path)

            data = np.array(data)
            ind = np.random.RandomState(seed=10).permutation(len(data))

            split = int(len(data) * data_config['data_split'])
            split2 = int(len(data) * 0.8)
            split3 = int(len(data) * 1)

            dataset.train_data = data[ind[:split]].tolist()
            dataset.unlabeled_train_data = data[ind[:split2]].tolist()
            dataset.test_data = data[ind[split2:split3]].tolist()

        elif data_type == 'TrainTest':

            file = os.path.join(data_path, 'train.pkl')
            data = dataset.load_TrainTestData(file)
            dataset.train_data = data
            dataset.unlabeled_train_data = data[:400]

            file = os.path.join(data_path, 'test.pkl')
            dataset.test_data = dataset.load_TrainTestData(file)
            # dataset.test_data = dataset.load_TrainTestData(file, 55, 59)


        torch.save(dataset.to_dict(), os.path.join('cache', cache_file_head + '.dataset'))
        encoded_tag, tag_mask = dataset.encode_tag()
        torch.save(encoded_tag, os.path.join('cache', cache_file_head + '.encoded_tag'))
        torch.save(tag_mask, os.path.join('cache', cache_file_head + '.tag_mask'))

    return dataset, encoded_tag, tag_mask


class dataEngine(Dataset):
    def __init__(self, train_data=None, unlabeled_train_data=None, test_data=None,
                    tag2id={}, id2tag={}, co_occur_mat=None, tfidf_dict=None, data_config={}):
        self.train_data = train_data
        self.unlabeled_train_data = unlabeled_train_data
        self.test_data = test_data

        self.tag2id = tag2id
        self.id2tag = id2tag

        self.use_tags = {}
        self.tags_instances = {}

        self.co_occur_mat = co_occur_mat
        self.tfidf_dict = tfidf_dict

        self.data_config = data_config

    @classmethod
    def from_dict(cls, data_dict):
        return dataEngine(data_dict.get('train_data'),
                       data_dict.get('unlabeled_train_data'),
                       data_dict.get('test_data'),
                       data_dict.get('tag2id'),
                       data_dict.get('id2tag'),
                       data_dict.get('co_occur_mat'),
                       data_dict.get('tfidf_dict'))

    def to_dict(self):
        data_dict = {
            'train_data': self.train_data,
            'unlabeled_train_data': self.unlabeled_train_data,
            'test_data': self.test_data,
            'tag2id': self.tag2id,
            'id2tag': self.id2tag,
            'co_occur_mat': self.co_occur_mat,
            'tfidf_dict': self.tfidf_dict
        }
        return data_dict

    def get_tags_num(self):
        return len(self.tag2id)

    def encode_tag(self):
        tag_ids = []
        tag_token_num = []
        for i in range(self.get_tags_num()):
            tag = self.id2tag[i]
            tokens = tokenizer.tokenize(tag)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            tag_ids.append(token_ids)
            tag_token_num.append(len(tokens))
        max_num = max(tag_token_num)
        padded_tag_ids = torch.zeros((self.get_tags_num(), max_num), dtype=torch.long)
        mask = torch.zeros((self.get_tags_num(), max_num))
        for i in range(self.get_tags_num()):
            mask[i, :len(tag_ids[i])] = 1.
            padded_tag_ids[i, :len(tag_ids[i])] = torch.tensor(tag_ids[i])
        return padded_tag_ids, mask

    def collate_fn(self, batch):
        # construct input
        inputs = [e['dscp_ids'] for e in batch]  #e['title_ids'] +
        dscp_tokens = [e['dscp_tokens'] for e in batch]


        lengths = np.array([len(e) for e in inputs])
        max_len = np.max(lengths)  #_to_max_length=True , truncation=True
        inputs = [tokenizer.prepare_for_model(e, max_length=max_len+2, pad_to_max_length=True) for e in inputs]
        # inputs = [tokenizer.prepare_for_model(e, max_length=max_len + 2, pad_to_max_length=True, truncation=True) for e in inputs]

        ids = torch.LongTensor([e['input_ids'] for e in inputs])
        token_type_ids = torch.LongTensor([e['token_type_ids'] for e in inputs])
        attention_mask = torch.FloatTensor([e['attention_mask'] for e in inputs])
        # construct tag
        tags = torch.zeros(size=(len(batch), self.get_tags_num()))
        for i in range(len(batch)):
            tags[i, batch[i]['tag_ids']] = 1.

        dscp = [e['dscp'] for e in batch]
        # label_mask = torch.tensor([e['label'] for e in batch]).byte()
        # label_mask = torch.nonzero(label_mask).squeeze(-1)

        return (ids, token_type_ids, attention_mask, dscp_tokens), tags, dscp


    def load_allData(self, f):
        data = []

        document = []
        tag_occurance = {}
        # csv.field_size_limit(sys.maxsize)
        with open(f, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:

                if len(row) != 4:
                    continue
                _, _, _, tag = row

                tag = tag.strip().split('###')
                tag = [t for t in tag if t != '']

                for t in tag:
                    if t not in tag_occurance:
                        tag_occurance[t] = 1
                    else:
                        tag_occurance[t] += 1

        # ignored_tags = set(['Tools','Applications','Other', 'API', 'Software-as-a-Service','Platform-as-a-Service',
        # 'Data-as-a-Service'])  #
        for tag in tag_occurance:
            if self.data_config['min_tagFrequence'] <= tag_occurance[tag] <= self.data_config['max_tagFrequence']:
                self.use_tags[item[0]] = item[1]

        print('Total number of tags: {}'.format(len(tag_occurance)))
        print(sorted(tag_occurance.items(), key=lambda x: x[1], reverse=True))

        with open(f, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:

                if len(row) != 4:
                    continue
                id, title, dscp, tag = row

                title_tokens = tokenizer.tokenize(title.strip())
                dscp_tokens = title_tokens + tokenizer.tokenize(dscp.strip())

                if len(dscp_tokens) > 510:
                    if self.data_config['overlength_handle'] == 'truncation':
                        dscp_tokens = dscp_tokens[:510]
                    else:
                        continue

                dscp_ids = tokenizer.convert_tokens_to_ids(dscp_tokens)

                tag = tag.strip().split('###')
                tag = [t for t in tag if t != '']

                if self.use_tags is not None:
                    tag = [t for t in tag if t in self.use_tags]

                # if len(set(tag)) < 2:
                #     continue

                if len(tag) == 0:
                    continue

                for t in tag:
                    if t not in self.tag2id:
                        tag_id = len(self.tag2id)
                        self.tag2id[t] = tag_id
                        self.id2tag[tag_id] = t

                tag_ids = [self.tag2id[t] for t in tag]

                data.append({
                    'id': int(id),
                    'dscp_ids': dscp_ids,
                    'dscp_tokens': dscp_tokens,
                    'tag_ids': tag_ids,
                    'dscp': dscp
                })

        print("The number of tags for training: {}".format(len(self.tag2id)))
        # os.makedirs('cache', exist_ok=True)

        return data
        

    def load_TrainTestData(self, file,  minwords=0, maxwords=1000):
        data = []
        document = []

        taglen = 0
        item = 0
        i=0
        with open(file, 'rb') as pklfile:

            reader = pickle.load(pklfile)

            for row in reader:
                skip = False
                no_skip = False
                if len(row) != 4:
                    continue

                id = row["id"]
                title = row["name"]
                dscp = row["descr"]
                tag = row["tags"]

                title_tokens = tokenizer.tokenize(title.strip())
                dscp_tokens = title_tokens + tokenizer.tokenize(dscp.strip())

                # print(title_tokens)         

                if len(dscp_tokens) > maxwords or len(dscp_tokens) < minwords:
                    if 'camp' not in title_tokens:
                        continue
                    
                if len(dscp_tokens) > 510:
                    i+=1
                    if self.data_config['overlength_handle'] == 'truncation':
                        dscp_tokens = dscp_tokens[:510]
                    else:
                        continue

                dscp_ids = tokenizer.convert_tokens_to_ids(dscp_tokens)

                # if self.use_tags is not None:
                tag = [t for t in tag]

                if len(tag) == 0:
                    continue
                taglen += len(tag)
                item += 1

                for t in tag:
                    if t not in self.tag2id:
                        tag_id = len(self.tag2id)
                        self.tag2id[t] = tag_id
                        self.id2tag[tag_id] = t

                tag_ids = [self.tag2id[t] for t in tag]
                
                
                data.append({
                    'id': int(id),
                    'dscp_ids': dscp_ids,
                    'dscp_tokens': dscp_tokens,
                    'tag_ids': tag_ids,
                    'dscp': dscp,
                    'label': 1
                })

        print("The number of tags for training: {}".format(len(self.tag2id)))
        # print(self.id2tag)
        print("taglen: {}".format(taglen/item))
        print("The number of tags of which the length is over 512: {}".format(i))
        print(item)
        return data
