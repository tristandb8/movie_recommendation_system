import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from datetime import datetime, timedelta
np.random.seed(123)
import warnings
warnings.filterwarnings('ignore')
from NCF import NCF
from DL import MovieLensTrainDataset
import sys


def main(argv):
    num_epoch = 10
    filterbytime = False # Don't use
    keptUsersRatio = 0.1
    numdays = timedelta(days = 356*3)
    includeRating = True
    useTime = True

    print("------------ params ------------")
    print("num_epoch:", num_epoch)
    print("includeRating:", includeRating)
    print("useTime:", useTime)
    print("filterbytime:", filterbytime)
    if filterbytime:
        print("numdays:", numdays)
    else:
        print("keptUsersRatio:", keptUsersRatio)
    print("--------------------------------")
    sys.stdout.flush()

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')  # Use the first available GPU
        print('Using CUDA device:', device)
        print('Device index:', torch.cuda.current_device())
    else:
        device = torch.device('cpu')
        print('CUDA is not available, using CPU.')

    ratings = pd.read_csv('movie_20m/rating.csv', 
    parse_dates=['timestamp'])
    ratings["timestamp"] = (ratings.timestamp-ratings.timestamp.min()).astype('timedelta64[M]')
    ratings["timestamp"] = ratings["timestamp"].astype(int)
    timerange = ratings["timestamp"].max() + 1
    ratings.rating *= 2
    ratings.rating = ratings.rating.astype(int)

    if filterbytime:
        ratings['rank_first'] = ratings.groupby(['userId'])['timestamp'] \
                                        .rank(method='first', ascending=True)
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'] \
                                        .rank(method='first', ascending=False)
        last_ratings = ratings[ratings['rank_latest'] == 1]
        first_ratings = ratings[ratings['rank_first'] == 1]

        print("Filtering users for more than", numdays, "days")
        ids_to_keep = []
        for id in last_ratings['userId'].values:
            dt = last_ratings[last_ratings['userId'] == id].timestamp.iloc[0] - first_ratings[first_ratings['userId'] == id].timestamp.iloc[0]
            if numdays < dt:
                ids_to_keep.append(id)
        print("Keeping " + str(len(ids_to_keep)) + " users out of "
            + str(len(ratings.userId.unique())))
        sys.stdout.flush()
        ratings = ratings[ratings.userId.isin(ids_to_keep)]
    else:
        rand_userIds = np.random.choice(ratings['userId'].unique(), 
                                    size=int(len(ratings['userId'].unique())*keptUsersRatio), 
                                    replace=False)

        ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]

        print('There are {} rows of data from {} users'.format(len(ratings), len(rand_userIds)))

    # Last movie watched by each user is used as testing data
    # while rest is used as training data
    ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'] \
                                        .rank(method='first', ascending=False)
    train_ratings = ratings[ratings['rank_latest'] != 1]
    test_ratings = ratings[ratings['rank_latest'] == 1]

    # drop columns that we no longer need
    train_ratings = train_ratings[['userId', 'movieId', 'rating', 'timestamp']]
    test_ratings = test_ratings[['userId', 'movieId', 'rating', 'timestamp']]
    ratings = ratings[['userId', 'movieId', 'rating', 'timestamp']]

    num_users = ratings['userId'].max()+1
    num_items = ratings['movieId'].max()+1
    all_movieIds = ratings['movieId'].unique()

    model = NCF(num_users, num_items, includeRating, timerange, useTime).cuda()
    train_dataloader = DataLoader(MovieLensTrainDataset(train_ratings, includeRating, useTime),
                                batch_size=512, num_workers=0)
    
    criterionB = nn.BCELoss().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters())

    # User-item pairs for testing
    if useTime:
        test_user_item_set = set(zip(test_ratings['userId'], test_ratings['movieId'], ratings['timestamp'], test_ratings['rating']))
    else:
        test_user_item_set = set(zip(test_ratings['userId'], test_ratings['movieId'], test_ratings['rating']))
    # Dict of all items that are interacted with by each user
    user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()



    for epoch in range(num_epoch):
        losses = []
        steps = 0
        for i, data in enumerate(train_dataloader, 0):
            if useTime:
                user_input, item_input, time_input, labels = data
                time_input = Variable(time_input.cuda())
            else:
                user_input, item_input, labels = data
            optimizer.zero_grad()
            user_input = Variable(user_input.cuda())
            item_input = Variable(item_input.cuda())
            
            labels = torch.as_tensor(labels)
            labels = Variable(labels.cuda())
            if useTime:
                predB, pred = model(user_input, item_input, time_input)#.squeeze()
            else:
                predB, pred = model(user_input, item_input, False)#.squeeze()

            predB = predB.squeeze()
            lossS = criterionB(predB,torch.tensor(labels > 0, dtype=torch.float32))
            if includeRating:
                pred = pred.squeeze()
                lossS = criterionB(predB,torch.tensor(labels > 0, dtype=torch.float32))
                lossR = criterion(pred,labels.long())
                loss = lossS + lossR
            else:
                loss = lossS
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            steps += 1
            # if steps % 2000 == 0:
            #     print('Epoch: {} Loss: {:.4f}'.format(epoch,np.mean(losses)))
        print("End of epoch " + str(epoch) + ", mean loss: " + str(np.mean(losses)))
        getTestAcc(test_user_item_set, user_interacted_items, all_movieIds, model, epoch, useTime, includeRating)
        sys.stdout.flush()



def getTestAcc(test_user_item_set, user_interacted_items, all_movieIds, model, epoch, useTime, includeRating):
    hits = []
    correctRatings = []
    for data in (test_user_item_set):
        if useTime:
            u,i,t,r = data
            test_times = [t] * 100
        else:
            u,i,r = data
        interacted_items = user_interacted_items[u]
        not_interacted_items = set(all_movieIds) - set(interacted_items)
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
        test_items = selected_not_interacted + [i]
        
        if useTime:
            predB, pred = model(torch.tensor([u]*100).cuda(), 
                                                torch.tensor(test_items).cuda(), torch.tensor(test_times).cuda())
        else:
            predB, pred = model(torch.tensor([u]*100).cuda(), torch.tensor(test_items).cuda(), False) 
        predB = predB.cpu().detach().numpy().squeeze()
        # pred = pred.cpu().detach().numpy().squeeze()
        
        top10_items = [test_items[i] for i in np.argsort(predB)[::-1][0:10].tolist()]
        
        if i in top10_items:
            hits.append(1)
        else:
            hits.append(0)

        if includeRating:
            if useTime:
                predB, pred = model(torch.tensor(u).cuda(), torch.tensor(i).cuda(), torch.tensor(t).cuda())
            else:
                predB, pred = model(torch.tensor(u).cuda(), torch.tensor(i).cuda(), False)
            pred = pred.cpu().detach().numpy().squeeze()
            predmax = np.argmax(pred[1:]) + 1
            correctRatings.append(int(predmax == r))

    print("The Hit Ratio @ 10 is ", np.average(hits) * 100, "%", " at epoch ", epoch)
    if includeRating:
        print("Correct rating percentage: ", np.average(correctRatings) * 100, "%", " at epoch ", epoch)

if __name__ == "__main__":
   main(sys.argv[1:])