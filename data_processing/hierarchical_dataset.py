'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import json
import torch
import math
from typing import TypeVar, Optional, Iterator
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
# import torchvision.transforms as transforms
from PIL import Image
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

# def txt_parse(f):
#     result = []
#     with open(f) as fp:
#         line = fp.readline()
#         result.append(line)
#         while line:
#             line = fp.readline()
#             result.append(line)
#     return result


# class DeepFashionHierarchihcalDataset(Dataset):
#     def __init__(self, list_file, class_map_file, repeating_product_ids_file, transform=None):
#         with open(list_file, 'r') as f:
#             data_dict = json.load(f)
#         assert len(data_dict['images']) == len(data_dict['categories'])
#         num_data = len(data_dict['images'])
#         self.transform = transform
#         self.augment_transform = transforms.RandomChoice([
#             transforms.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.)),
#             transforms.RandomHorizontalFlip(1),
#             transforms.ColorJitter(0.4, 0.4, 0.4)])

#         with open(class_map_file, 'r') as f:
#             self.class_map = json.load(f)
#         self.repeating_product_ids = txt_parse(repeating_product_ids_file)
#         self.filenames = []
#         self.category = []
#         self.labels = {}
#         for i in range(num_data):
#             filename = data_dict['images'][i]
#             category = self.class_map[data_dict['categories'][i]]
#             product, variation, image = self.get_label_split(filename)
#             if product not in self.repeating_product_ids:
#                 if category not in self.labels:
#                     self.labels[category] = {}
#                 if product not in self.labels[category]:
#                     self.labels[category][product] = {}
#                 if variation not in self.labels[category][product]:
#                     self.labels[category][product][variation] = {}
#                 self.labels[category][product][variation][image] = i # Category: category number, Product: product ID(unique for each of the product), variation: variation number, image: image number for the same variation
#                 self.category.append(category)
#                 self.filenames.append(filename)

#     def get_label_split(self, filename):
#         split = filename.split('/')
#         image_split = split[-1].split('.')[0].split('_')
#         return int(split[-2][3:]), int(image_split[0]), int(image_split[1])

#     def get_label_split_by_index(self, index):   
#         filename = self.filenames[index]
#         category = self.category[index]
#         product, variation, image = self.get_label_split(filename)

#         return category, product, variation, image

#     def __getitem__(self, index):
#         images0, images1, labels = [], [], []
#         for i in index:
#             image = Image.open(self.filenames[i])
#             label = list(self.get_label_split_by_index(i))
#             if self.transform:
#                 image0, image1 = self.transform(image)
#             images0.append(image0)
#             images1.append(image1)
#             labels.append(label)

#         return [torch.stack(images0), torch.stack(images1)], torch.tensor(labels)

#     def random_sample(self, label, label_dict):
#         curr_dict = label_dict
#         top_level = True
#         #all sub trees end with an int index
#         while type(curr_dict) is not int:
#             if top_level:
#                 random_label = label
#                 if len(curr_dict.keys()) != 1:
#                     while (random_label == label):
#                         random_label = random.sample(curr_dict.keys(), 1)[0]
#             else:
#                 random_label = random.sample(curr_dict.keys(), 1)[0]
#             curr_dict = curr_dict[random_label]
#             top_level = False
#         return curr_dict

#     def __len__(self):
#         return len(self.filenames)


# class DeepFashionHierarchihcalDatasetEval(Dataset):
#     def __init__(self, list_file, class_map_file, repeating_product_ids_file, transform=None):
#         with open(list_file, 'r') as f:
#             data_dict = json.load(f)
#         assert len(data_dict['images']) == len(data_dict['categories'])
#         num_data = len(data_dict['images'])

#         self.transform = transform
#         self.augment_transform = transforms.RandomChoice([
#             transforms.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.)),
#             transforms.RandomHorizontalFlip(1),
#             transforms.ColorJitter(0.4, 0.4, 0.4)])

#         with open(class_map_file, 'r') as f:
#             self.class_map = json.load(f)
#         self.repeating_product_ids = txt_parse(repeating_product_ids_file)
#         self.filenames = []
#         self.category = []
#         self.labels = {}
#         for i in range(num_data):
#             filename = data_dict['images'][i]

#             category = self.class_map[data_dict['categories'][i]]

#             product, variation, image = self.get_label_split(filename)
#             if product not in self.repeating_product_ids:
#                 if category not in self.labels:
#                     self.labels[category] = {}
#                 if product not in self.labels[category]:
#                     self.labels[category][product] = {}
#                 if variation not in self.labels[category][product]:
#                     self.labels[category][product][variation] = {}
#                 self.labels[category][product][variation][image] = i
#                 self.category.append(category)
#                 self.filenames.append(filename)

#     def get_label_split(self, filename):
#         split = filename.split('/')
#         image_split = split[-1].split('.')[0].split('_')
#         return int(split[-2][3:]), int(image_split[0]), int(image_split[1])

#     def get_label_split_by_index(self, index):
#         filename = self.filenames[index]
#         category = self.category[index]
#         product, variation, image = self.get_label_split(filename)

#         return category, product, variation, image

#     def __getitem__(self, index):
#         image = Image.open(self.filenames[index])
#         label = list(self.get_label_split_by_index(index))
#         if self.transform:
#             image = self.transform(image)

#         return image, label

#     def random_sample(self, label, label_dict):
#         curr_dict = label_dict
#         top_level = True
#         #all sub trees end with an int index
#         while type(curr_dict) is not int:
#             if top_level:
#                 random_label = label
#                 if len(curr_dict.keys()) != 1:
#                     while (random_label == label):
#                         random_label = random.sample(curr_dict.keys(), 1)[0]
#             else:
#                 random_label = random.sample(curr_dict.keys(), 1)[0]
#             curr_dict = curr_dict[random_label]
#             top_level = False
#         return curr_dict

#     def __len__(self):
#         return len(self.filenames)

def get_leaf(labels, label_path):
    leaf = set()
    for label in labels:
        leaf = leaf - set(label_path[label])
        leaf.add(label)
    return list(leaf)

class BertDataset_rcv(Dataset):
    def __init__(self, max_token=512, device='cpu', pad_idx=0, data_path=None, label_dict=None, is_test=False,
                 label_path=None):
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
        label_hier = {}
        sample2leaf = {}
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
                leaves = get_leaf(line['label'], label_path)
                sample2leaf[total - 1] = leaves

                for leaf in leaves:
                    path = label_path[leaf]
                    top_label = True
                    top_dict = None
                    for label in path:
                        if top_label:
                            if label not in label_hier:
                                label_hier[label] = {}
                            top_dict = label_hier[label]
                            top_label = False
                        else:
                            if label not in top_dict:
                                top_dict[label] = {}
                            top_dict = top_dict[label]
                    top_dict[total - 1] = 1

        self.label_path = label_path
        self.label_hier = label_hier
        self.sample2leaf = sample2leaf
                # get paths of the sample

                # go through the sample path and add the label to the label hierarchy
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
        print(batch[0].keys())
        print(len(batch[0]['data']))
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
    
    def random_sample(self, label, label_dict):
        curr_dict = label_dict
        top_level = True
        #all sub trees end with an int index
        while type(curr_dict) is not int:
            if top_level:
                random_label = label
                if len(curr_dict.keys()) != 1:
                    while (random_label == label):
                        random_label = random.sample(curr_dict.keys(), 1)[0]
            else:
                random_label = random.sample(curr_dict.keys(), 1)[0]
            curr_dict = curr_dict[random_label]
            top_level = False
        return curr_dict
    
    def get_split_by_index(self, idx):
        leaves = self.sample2leaf[idx]
        label_by_level = {}
        for leaf in leaves:
            path = self.label_path[leaf]
            for i, label in enumerate(path):
                if i not in label_by_level:
                    label_by_level[i] = set()
                label_by_level[i].add(label)
        return label_by_level
    
class HierarchicalBatchSampler(Sampler):
    def __init__(self, batch_size: int,
        drop_last: bool, dataset: BertDataset_rcv,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,) -> None:

        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.epoch=0
        self.num_replicas = 1
        self.rank = 0
        
        # if num_replicas is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Requires distributed package to be available")
        #     num_replicas = dist.get_world_size()
        # if rank is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Requires distributed package to be available")
        #     rank = dist.get_rank()
        # self.num_replicas = num_replicas
        # self.rank = rank
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / \
                self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(
                len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        print(self.total_size, self.num_replicas, self.batch_size,
              self.num_samples, len(self.dataset), self.rank)

    def random_unvisited_sample(self, label, label_dict, visited, indices, remaining, num_attempt=10):
        attempt = 0
        while attempt < num_attempt:
            idx = self.dataset.random_sample(
                label, label_dict)
            if idx not in visited and idx in indices:
                visited.add(idx)
                return idx
            attempt += 1
        idx = remaining[torch.randint(len(remaining), (1,))]
        visited.add(idx)
        return idx

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        batch = []
        visited = set()
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        remaining = list(set(indices).difference(visited))
        while len(remaining) > self.batch_size:
            idx = indices[torch.randint(len(indices), (1,))]
            batch.append(idx)
            visited.add(idx)
            label_by_level = self.dataset.get_split_by_index(
                idx)
            for level, labels in label_by_level.items():
                rand_label = random.sample(labels, 1)[0]
                label_path = self.dataset.label_path[rand_label]
                label_hier = self.dataset.label_hier
                for i, label in enumerate(label_path):
                    label_hier = label_hier[label]
                idx = self.random_unvisited_sample(
                    rand_label, label_hier, visited, indices, remaining)
                batch.append(idx)
                visited.add(idx)
            remaining = list(set(indices).difference(visited))
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
            remaining = list(set(indices).difference(visited))

        if (len(remaining) > self.batch_size) and not self.drop_last:
            batch.update(list(remaining))
            yield batch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

