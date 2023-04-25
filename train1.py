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
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import KNNBasic
from surprise import SVDpp
from surprise.accuracy import rmse




def main(argv):
    num_epoch = 10
    filterbytime = False
    keptUsersRatio = 0.1
    numdays = timedelta(days = 356*3)
    includeRating = True
    useTime = False
    binary = True

    print("------------ params ------------")
    print("num_epoch:", num_epoch)
    print("includeRating:", includeRating)
    print("useTime:", useTime)
    print("filterbytime:", filterbytime)
    print("binary", binary)
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
    # timerange = ratings["timestamp"].max() + 1
    # ratings.rating *= 2
    if binary:
        ratings.rating = ratings.rating > 0
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
    #reader = Reader(rating_scale=(0.5, 5))
    #data = Dataset.load_from_df(train_ratings[['userId', 'movieId', 'rating', 'timestamp']], reader)
    if binary:
        reader = Reader(rating_scale=(0, 1))
        train_dataset = MovieLensTrainDataset(train_ratings, includeRating, useTime)
        df = pd.DataFrame({
            'user': [x[0].item() for x in train_dataset],
            'item': [x[1].item() for x in train_dataset],
            'rating': [x[2].item() for x in train_dataset],
        })
        data = Dataset.load_from_df(df, reader=reader)

    else:
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(train_ratings[['userId', 'movieId', 'rating']], reader)

    

    trainset = data.build_full_trainset()
    all_movieIds = ratings['movieId'].unique()


    # Train the SVD algorithm
    algo_svd = SVD()
    print("train svd")
    algo_svd.fit(trainset)

    # # Train the SVD++ algorithm
    algo_svdpp = SVDpp(n_factors=100, n_epochs=5, lr_all=0.005, reg_all=0.02, random_state=42)
    print("train svd++")
    algo_svdpp.fit(trainset)

    # Train the KNNBasic algorithm
    ucf = KNNBasic(sim_options={'user_based': True})
    print("train ucf")
    ucf.fit(trainset)

    icf = KNNBasic(sim_options={'user_based': False})
    print("train icf")
    icf.fit(trainset)

    # User-item pairs for testing
    test_user_item_set = set(zip(test_ratings['userId'], test_ratings['movieId'], ratings['timestamp'], test_ratings['rating']))
    # Dict of all items that are interacted with by each user
    user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()


    getTestAcc("Probabilistic Matrix Factorization", test_user_item_set, user_interacted_items, all_movieIds, algo_svd, 0, binary)
    getTestAcc("SVP++", test_user_item_set, user_interacted_items, all_movieIds, algo_svdpp, 0, binary)
    getTestAcc("User based Collaborative Filtering", test_user_item_set, user_interacted_items, all_movieIds, ucf, 0, binary)
    getTestAcc("Item based Collaborative Filtering", test_user_item_set, user_interacted_items, all_movieIds, icf, 0, binary)
    sys.stdout.flush()






def getTestAcc(method, test_user_item_set, user_interacted_items, all_movieIds, model, epoch, binary):
    hits = []
    correctRatings = []
    preds = []
    for (u,i,t,r) in (test_user_item_set):
        if binary:
            interacted_items = user_interacted_items[u]
            not_interacted_items = set(all_movieIds) - set(interacted_items)
            selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
            test_items = selected_not_interacted + [i]

            preds = []
            for j in range(len(test_items)):
                pred = model.predict(u, test_items[j]).est
                #print(pred)
                preds.append(pred)
                # iscor = ""
                # if j == len(test_items) - 1:
                #     iscor = "correct"
                # print(iscor, pred, r)
            top10_items = [test_items[i] for i in np.argsort(preds)[::-1][0:10].tolist()]
            hits.append(int(i in top10_items))
        else:
            pred = model.predict(u, i, t).est#, torch.tensor(t).cuda())
            pred = int(pred*2)/2
            
            correctRatings.append(int(pred == r))
            preds.append(pred)
    print("-------------------", method, "-------------------")
    if binary:
        print("The Hit Ratio @ 10 is ", np.average(hits) * 100, "%", " at epoch ", epoch)
    else:
        print("Correct rating percentage: ", np.average(correctRatings) * 100, "%", " at epoch ", epoch)


if __name__ == "__main__":
   main(sys.argv[1:])