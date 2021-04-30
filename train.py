import json
from  nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json', mode='r') as f:
    intents = json.load(f)

#print(intents)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


#print(all_words)
ignore_words = ['?','!',',','.']
stemmed_words = [stem(w) for w in all_words if w not in ignore_words]
#print(stemmed_words)

sorted_all_words = sorted(set(stemmed_words))
sorted_tags = sorted(set(tags))
print(sorted_all_words)

X_train =[]
y_train =[]

for (pattern, tag) in xy:
    bag = bag_of_words(pattern, tag)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)



class  ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def get_item(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


batch_size = 8
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
