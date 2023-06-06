from transformers import AutoTokenizer
from fairseq.data import data_utils
import torch
from typing import TypeVar, Optional, Iterator
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pytorch_lightning as pl
import os
import pickle
import math
import json
from tqdm import tqdm

class BertDataset(Dataset):
    def __init__(self, max_token=512, device='cpu', pad_idx=0, data_path=None, label_dict=None):
        self.device = device
        super(BertDataset, self).__init__()
        self.data = data_utils.load_indexed_dataset(
            data_path + '/tok', None, 'mmap'
        )
        self.labels = data_utils.load_indexed_dataset(
            data_path + '/Y', None, 'mmap'
        )
        self.max_token = max_token
        self.pad_idx = pad_idx

    def __getitem__(self, item):
        if isinstance(item, int):
            data = self.data[item][:self.max_token - 2].to(
                self.device)
            labels = self.labels[item].to(self.device)
            return {'data': data, 'label': labels, 'idx': item}
        data, label = [], []
        for i in item:
            data.append(self.data[i][:self.max_token - 2])
            label.append(self.labels[i])

        # data = self.data[item][:self.max_token - 2].to(
        #     self.device)
        # labels = self.labels[item].to(self.device)
        return {'data': data, 'label': label, 'idx': item}

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        label = torch.stack([b['label'] for b in batch], dim=0)
        data = torch.full([len(batch), self.max_token], self.pad_idx, device=label.device, dtype=batch[0]['data'].dtype)
        idx = [b['idx'] for b in batch]
        for i, b in enumerate(batch):
            data[i][:len(b['data'])] = b['data']
        return data, label, idx
    
    def collate_fn_1(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        batch = batch[0]

        label = torch.stack(batch['label'], dim=0)
        data = torch.full([len(batch['data']), self.max_token], self.pad_idx, device=label.device, dtype=batch['data'][0].dtype)
        idx = batch['idx']
        for i, b in enumerate(batch['data']):
            data[i][:len(b)] = b

        return data, label, idx
    
    def get_label(self, idx):
        return self.labels[idx]
    
class BertDataset_rcv(Dataset):
    def __init__(self, max_token=512, device='cpu', pad_idx=0, data_path=None, label_dict=None, is_test=False):
        self.device = device
        super(BertDataset_rcv, self).__init__()
        
        self.max_token = max_token
        self.pad_idx = pad_idx
        self.is_test = is_test
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        data = []
        labels = []

        num_lines = sum(1 for line in open(data_path,'r'))

        total = 0
        # load the json file of data_path
        with open(data_path, 'r') as f:
            for line in tqdm(f, total=num_lines):
                total += 1
                if (not self.is_test) and ('rcv' in data_path) and (total > 100000):
                    break
                line = json.loads(line)
                data.append(tokenizer.encode(line['token'], truncation=True, max_length=self.max_token, padding='max_length', add_special_tokens=True))

                one_hot = np.zeros(len(label_dict))
                for label in line['label']:
                    one_hot[label_dict[label]] = 1
                labels.append(one_hot)
        self.data = torch.from_numpy(np.array(data))
        self.labels = torch.from_numpy(np.array(labels))

    def __getitem__(self, item):
        if isinstance(item, int):
            data = self.data[item].to(
                self.device)
            labels = self.labels[item].to(self.device)
            return {'data': data, 'label': labels, 'idx': item}
        data, label = [], []
        for i in item:
            data.append(self.data[i])
            label.append(self.labels[i])

        # data = self.data[item][:self.max_token - 2].to(
        #     self.device)
        # labels = self.labels[item].to(self.device)
        return {'data': data, 'label': label, 'idx': item}

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        label = torch.stack([b['label'] for b in batch], dim=0)
        data = torch.full([len(batch), self.max_token], self.pad_idx, device=label.device, dtype=batch[0]['data'].dtype)
        idx = [b['idx'] for b in batch]
        for i, b in enumerate(batch):
            data[i][:len(b['data'])] = b['data']
        return data, label, idx
    
    def collate_fn_1(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        batch = batch[0]

        label = torch.stack(batch['label'], dim=0)
        data = torch.full([len(batch['data']), self.max_token], self.pad_idx, device=label.device, dtype=batch['data'][0].dtype)
        idx = batch['idx']
        for i, b in enumerate(batch['data']):
            data[i][:len(b)] = b

        return data, label, idx
    
    def get_label(self, idx):
        return self.labels[idx]

class BertDataset_wos(Dataset):
    def __init__(self, max_token=512, device='cpu', pad_idx=0, data_path=None, label_dict=None, is_test=False):
        self.device = device
        super(BertDataset_wos, self).__init__()
        
        self.max_token = max_token
        self.pad_idx = pad_idx
        self.is_test = is_test
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        data = []
        labels = []

        num_lines = sum(1 for line in open(data_path,'r'))

        total = 0
        # load the json file of data_path
        with open(data_path, 'r') as f:
            for line in tqdm(f, total=num_lines):
                total += 1
                line = json.loads(line)
                data.append(tokenizer.encode(line['doc_token'], truncation=True, max_length=self.max_token, padding='max_length', add_special_tokens=True))

                one_hot = np.zeros(len(label_dict))
                for label in line['doc_label']:
                    one_hot[label_dict[label]] = 1
                labels.append(one_hot)
        self.data = torch.from_numpy(np.array(data))
        self.labels = torch.from_numpy(np.array(labels))

    def __getitem__(self, item):
        if isinstance(item, int):
            data = self.data[item].to(
                self.device)
            labels = self.labels[item].to(self.device)
            return {'data': data, 'label': labels, 'idx': item}
        data, label = [], []
        for i in item:
            data.append(self.data[i])
            label.append(self.labels[i])

        # data = self.data[item][:self.max_token - 2].to(
        #     self.device)
        # labels = self.labels[item].to(self.device)
        return {'data': data, 'label': label, 'idx': item}

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        label = torch.stack([b['label'] for b in batch], dim=0)
        data = torch.full([len(batch), self.max_token], self.pad_idx, device=label.device, dtype=batch[0]['data'].dtype)
        idx = [b['idx'] for b in batch]
        for i, b in enumerate(batch):
            data[i][:len(b['data'])] = b['data']
        return data, label, idx
    
    def collate_fn_1(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        batch = batch[0]

        label = torch.stack(batch['label'], dim=0)
        data = torch.full([len(batch['data']), self.max_token], self.pad_idx, device=label.device, dtype=batch['data'][0].dtype)
        idx = batch['idx']
        for i, b in enumerate(batch['data']):
            data[i][:len(b)] = b

        return data, label, idx
    
    def get_label(self, idx):
        return self.labels[idx]
    
class HierarchicalBatchSampler(Sampler):
    def __init__(self, batch_size: int,
        drop_last: bool, dataset: Subset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        positive_threshold: int = 5,
        hard_negative_threshold: int = 10,
        easy_negative_threshold: int = 30) -> None:

        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset.dataset
        self.indices = dataset.indices
        self.positive_threshold = positive_threshold
        self.hard_negative_threshold = hard_negative_threshold
        self.easy_negative_threshold = easy_negative_threshold

        self.epoch=0
        # if num_replicas is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Requires distributed package to be available")
        #     num_replicas = dist.get_world_size()
        # if rank is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Requires distributed package to be available")
        #     rank = dist.get_rank()
        self.num_replicas = 1
        self.rank = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.indices) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.indices) - self.num_replicas) / \
                self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(
                len(self.indices) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        print(self.total_size, self.num_replicas, self.batch_size,
              self.num_samples, len(self.indices), self.rank)

    def hamming_distance_by_matrix(self, labels):
        # labels = (batch_size, num_labels)
        return torch.matmul(labels, (1 - labels).T) + torch.matmul(1 - labels, labels.T) # (batch_size, batch_size)
    
    def random_unvisited_sample(self, anchor_label, visited, indices, remaining, num_attempt=10):
        attempt = 0
        positives = []
        hard_negatives = []
        easy_negatives = []

        indices_tensor = torch.tensor(remaining)

        while attempt < num_attempt:
            # random sample 10 samples from the indices list
            rand_indices = indices_tensor[torch.randint(len(indices_tensor), (10,))]
            # get the labels of the random samples
            rand_labels = [self.dataset.get_label(self.indices[idx]) for idx in rand_indices] + [anchor_label]
            rand_labels = torch.stack(rand_labels, dim=0) # (11, num_labels)

            # compute the hamming distance between the anchor label and the random labels
            hamming_distance = self.hamming_distance_by_matrix(rand_labels) # (11, 11)
            # get the hamming distance between the anchor label and the random labels
            hamming_distance = hamming_distance[-1, :-1] # (10, )

            # iterate through the distance and add them to positives if 1<dist<=self.positives_threshold
            for i, dist in enumerate(hamming_distance):
                idx = rand_indices[i].item()
                if (idx in visited) or (dist < 1) or (dist > self.easy_negative_threshold):
                    continue
                elif (dist <= self.positive_threshold):
                    positives.append(idx)
                elif (dist <= self.hard_negative_threshold):
                    hard_negatives.append(idx)
                elif (dist <= self.easy_negative_threshold):
                    easy_negatives.append(idx)

                visited.add(idx)

            if (len(positives) >= 3) and (len(easy_negatives) >= 2) and (len(hard_negatives) >= 2):
                break
            attempt += 1
            
        # print(len(positives), len(hard_negatives), len(easy_negatives))

        if len(positives) > 3:
            positives = positives[:3]
        if len(hard_negatives) > 2:
            hard_negatives = hard_negatives[:2]
        if len(easy_negatives) > 2:
            easy_negatives = easy_negatives[:2]
        return positives, hard_negatives, easy_negatives

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        batch = []
        visited = set()
        raw_indices = torch.randperm(len(self.indices), generator=g).tolist()
        # indices = [self.indices[i] for i in indices] # random shuffle
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            raw_indices += raw_indices[:(self.total_size - len(raw_indices))]
        else:
            # remove tail of data to make it evenly divisible.
            raw_indices = raw_indices[:self.total_size]

        assert len(raw_indices) == self.total_size

        # subsample
        raw_indices = raw_indices[self.rank:self.total_size:self.num_replicas]
        assert len(raw_indices) == self.num_samples
        

        remaining = list(set(raw_indices).difference(visited))
        while len(remaining) > self.batch_size:
            idx = raw_indices[torch.randint(len(raw_indices), (1,))]
            batch.append(idx)
            visited.add(idx)
            # TODO this should get all the labels above the current label
            anchor_label = self.dataset.get_label(self.indices[idx])

            # sample easy or hard negative / positive
            positives, hard_negatives, easy_negatives = self.random_unvisited_sample(anchor_label, visited, raw_indices, remaining, num_attempt=10)
            
            # extend the batch with the three lists: positive, hard negative, easy negative
            batch.extend(positives)
            batch.extend(hard_negatives)
            batch.extend(easy_negatives)

            # update visited and remaining
            visited.update(positives)
            visited.update(hard_negatives)
            visited.update(easy_negatives)

            remaining = list(set(raw_indices).difference(visited))
            if len(batch) >= self.batch_size:
                batch = batch[:self.batch_size]
                yield batch
                batch = []
            remaining = list(set(raw_indices).difference(visited))

        if (len(remaining) > self.batch_size) and not self.drop_last:
            batch.update(list(remaining))
            batch = batch[:self.batch_size]
            yield batch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
        
class DInterface(pl.LightningDataModule):
    def __init__(self, args, tokenizer, label_depths, data_path, device, label_dict, positive_threshold: int = 5,
        hard_negative_threshold: int = 10,
        easy_negative_threshold: int = 30):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.label_depths = label_depths
        self.data_path = data_path
        self.device = device
        self.positive_threshold = positive_threshold
        self.hard_negative_threshold = hard_negative_threshold
        self.easy_negative_threshold = easy_negative_threshold
        # if 'bgc' in data_path:
        #     self.label_dict = label_dict
        # else:
        self.label_dict = {v: k for k, v in label_dict.items()}

    def setup(self, stage=None):
        # Load data
        # label_dict = torch.load(os.path.join(data_path, 'bert_label_dict.pt'))
        # label_dict = {i: self.tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}

        # with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
        #     new_label_dict = pickle.load(f)
        if 'rcv1' in self.data_path:
            train_data_path = os.path.join(self.data_path, 'rcv1_train_all.json')
            val_data_path = os.path.join(self.data_path, 'rcv1_val_all.json')
            test_data_path = os.path.join(self.data_path, 'rcv1_test.json')

            label_dict = self.label_dict
            self.train_dataset = BertDataset_rcv(data_path=train_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.dev_dataset = BertDataset_rcv(data_path=val_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.test_dataset = BertDataset_rcv(data_path=test_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, is_test=self.args.test_only)
            self.dataset = self.train_dataset
        elif 'bgc' in self.data_path:
            train_data_path = os.path.join(self.data_path, 'train_data.jsonl')
            val_data_path = os.path.join(self.data_path, 'dev_data.jsonl')
            test_data_path = os.path.join(self.data_path, 'test_data.jsonl')

            label_dict = self.label_dict
            self.train_dataset = BertDataset_rcv(data_path=train_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.dev_dataset = BertDataset_rcv(data_path=val_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.test_dataset = BertDataset_rcv(data_path=test_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.dataset = self.train_dataset
        elif 'patent' in self.data_path:
            train_data_path = os.path.join(self.data_path, 'train.jsonl')
            val_data_path = os.path.join(self.data_path, 'valid.jsonl')
            test_data_path = os.path.join(self.data_path, 'test.jsonl')

            label_dict = self.label_dict
            self.train_dataset = BertDataset_rcv(data_path=train_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.dev_dataset = BertDataset_rcv(data_path=val_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.test_dataset = BertDataset_rcv(data_path=test_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.dataset = self.train_dataset
        elif 'aapd' in self.data_path:
            train_data_path = os.path.join(self.data_path, 'train.jsonl')
            val_data_path = os.path.join(self.data_path, 'val.jsonl')
            test_data_path = os.path.join(self.data_path, 'test.jsonl')

            label_dict = self.label_dict
            self.train_dataset = BertDataset_rcv(data_path=train_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.dev_dataset = BertDataset_rcv(data_path=val_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.test_dataset = BertDataset_rcv(data_path=test_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.dataset = self.train_dataset

        elif 'wos' in self.data_path:
            train_data_path = os.path.join(self.data_path, 'wos_train.json')
            val_data_path = os.path.join(self.data_path, 'wos_val.json')
            test_data_path = os.path.join(self.data_path, 'wos_test.json')

            label_dict = self.label_dict
            self.train_dataset = BertDataset_wos(data_path=train_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.dev_dataset = BertDataset_wos(data_path=val_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.test_dataset = BertDataset_wos(data_path=test_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict)
            self.dataset = self.train_dataset

        else:
            dataset = BertDataset(data_path=self.data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id)
            self.dataset = dataset

            split = torch.load(os.path.join(self.data_path, 'split.pt'))
            self.train_dataset = Subset(dataset, split['train'])
            self.dev_dataset = Subset(dataset, split['val'])
            self.test_dataset = Subset(dataset, split['test'])

        # self.train_sampler = HierarchicalBatchSampler(batch_size=self.args.batch, dataset=self.train_dataset, drop_last=True,
        #                                               positive_threshold=self.positive_threshold,
        #                                               hard_negative_threshold=self.hard_negative_threshold,
        #                                               easy_negative_threshold=self.easy_negative_threshold)

    def train_dataloader(self):
        # return DataLoader(self.train_dataset, sampler=self.train_sampler, batch_size=1, collate_fn=self.dataset.collate_fn_1)
        return DataLoader(self.train_dataset, batch_size=self.args.batch, shuffle=False, collate_fn=self.dataset.collate_fn)
        
    def val_dataloader(self):
        val_dataloader = DataLoader(self.dev_dataset, batch_size=self.args.batch, shuffle=False, collate_fn=self.dataset.collate_fn)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.batch, shuffle=False, collate_fn=self.dataset.collate_fn)
        return [val_dataloader, test_dataloader]
        return DataLoader(self.dev_dataset, batch_size=self.args.batch, shuffle=False, collate_fn=self.dataset.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch, shuffle=False, collate_fn=self.dataset.collate_fn)
    
