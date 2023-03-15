import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.autograd import Variable
from datetime import datetime, timedelta
import itertools
np.random.seed(123)
import warnings
warnings.filterwarnings('ignore')
from NCF import NCF
from DL import MovieLensTrainDataset


# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')  # Use the first available GPU
    print('Using CUDA device:', device)
    print('Device index:', torch.cuda.current_device())
else:
    device = torch.device('cpu')
    print('CUDA is not available, using CPU.')

numdays = timedelta(days = 1000)
num_epoch = 3

ratings = pd.read_csv('movie_rec\\movies_20m\\rating.csv', 
parse_dates=['timestamp'])
ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'] \
                                .rank(method='first', ascending=False)
last_ratings = ratings[ratings['rank_latest'] == 1]

ratings['rank_first'] = ratings.groupby(['userId'])['timestamp'] \
                                .rank(method='first', ascending=True)
first_ratings = ratings[ratings['rank_first'] == 1]

print("Filtering users for more than", numdays, "days")
ids_to_keep = []
for id in last_ratings['userId'].values:
    dt = last_ratings[last_ratings['userId'] == id].timestamp.iloc[0] - first_ratings[first_ratings['userId'] == id].timestamp.iloc[0]
    if numdays < dt:
        ids_to_keep.append(id)
print("Keeping " + str(len(ids_to_keep)) + " users out of "
      + str(len(ratings.userId.unique())))
ratings = ratings[ratings.userId.isin(ids_to_keep)]


train_ratings = ratings[ratings['rank_latest'] != 1]
test_ratings = last_ratings

# drop columns that we no longer need
train_ratings = train_ratings[['userId', 'movieId', 'rating']]
test_ratings = test_ratings[['userId', 'movieId', 'rating']]
ratings = ratings.drop(["rank_latest", "rank_first"], axis = 1)

num_users = ratings['userId'].max()+1
num_items = ratings['movieId'].max()+1
all_movieIds = ratings['movieId'].unique()

model = NCF(num_users, num_items).cuda()
train_dataloader = DataLoader(MovieLensTrainDataset(ratings, all_movieIds),
                              batch_size=512, num_workers=0)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epoch):
    losses = []
    steps = 0
    for i, data in enumerate(train_dataloader, 0):
        user_input, item_input, labels = data
        optimizer.zero_grad()
        user_input = Variable(user_input.cuda())
        item_input = Variable(item_input.cuda())
        labels = torch.as_tensor(labels)
        labels = Variable(labels.cuda())
        predicted_labels = model(user_input, item_input).squeeze()
        loss = criterion(predicted_labels,torch.tensor(labels, dtype=torch.float32))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        steps += 1
        if steps % 2000 == 0:
            print('Epoch: {} Loss: {:.4f}'.format(epoch,np.mean(losses)))
    print("End of epoch " + str(epoch) + ", mean loss: " + str(np.mean(losses)))

    # User-item pairs for testing
test_user_item_set = set(zip(test_ratings['userId'], test_ratings['movieId']))

# Dict of all items that are interacted with by each user
user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()

hits = []
for (u,i) in (test_user_item_set):
    interacted_items = user_interacted_items[u]
    not_interacted_items = set(all_movieIds) - set(interacted_items)
    selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
    test_items = selected_not_interacted + [i]
    
    predicted_labels = np.squeeze(model(torch.tensor([u]*100).cuda(), 
                                        torch.tensor(test_items).cuda()).cpu().detach().numpy())
    
    top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]
    
    if i in top10_items:
        hits.append(1)
    else:
        hits.append(0)
        
print("The Hit Ratio @ 10 is {:.2f}%".format(np.average(hits) * 100))