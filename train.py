import json
import numpy as np
from model import NeuralNet
from  nltk_utils import bag_of_words, tokenize, stem

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


with open('intents.json', mode='r') as f:
    intents = json.load(f)

#print(intents)

all_words = []
tags = []
xy = []
#loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    #add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        #tokenize each sentence
        w=tokenize(pattern)
        all_words.extend(w)
        #add to xy pair
        xy.append((w, tag))


#print(all_words)
#remove dupicates and sort
ignore_words = ['?','!',',','.']
stemmed_words = [stem(w) for w in all_words if w not in ignore_words]
#print(stemmed_words)

all_words = sorted(set(stemmed_words))
tags = sorted(set(tags))
#print(sorted_all_words)

X_train =[]
y_train =[]

for (pattern, tag) in xy:
    bag = bag_of_words(pattern, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

#hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000


class  ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

     # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


#print(input_size, len(all_words))
#print(output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#train the model
#if __name__ == '__main__':
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            #forward pass
            outputs = model(words)
            loss  = criterion(outputs, labels)

            #backwardpass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if (epoch+1)%100 == 0 :
            print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item(): .4f}' )

print(f'final loss , loss={loss.item(): .4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
