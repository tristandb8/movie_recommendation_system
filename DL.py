import torch
from torch.utils.data import Dataset, DataLoader
from DL_ratings import RatingDataset
import numpy as np

class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training
    
    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds
    
    """

    def __init__(self, ratings, includeRating, useTime):
        self.includeRating = includeRating
        self.useTime = useTime
        #self.users, self.items, self.times, self.labels
        self.data = self.get_dataset(ratings)
        if self.useTime:
            self.users, self.items, self.times, self.labels = self.data
        else:
            self.users, self.items, self.labels = self.data
            
    

    def __len__(self):
        return len(self.users)
  
    def __getitem__(self, idx):
        if self.useTime:
            return self.users[idx], self.items[idx], self.times[idx], self.labels[idx]
        else:
            return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings):
        all_movieIds = ratings['movieId'].unique()
        users, items, labels = [], [], []
        if self.useTime:
            user_item_set = set(zip(ratings['userId'], ratings['movieId'], ratings['rating'], ratings['timestamp']))
            movieMeanViewTime = ratings.groupby('movieId')['timestamp'].mean().astype(int).to_dict()
            times = []
        else:
            user_item_set = set(zip(ratings['userId'], ratings['movieId'], ratings['rating']))

        num_negatives = 5
        for data in user_item_set:
            if self.useTime:
                u, i, r, t = data
                times.append(t)
            else:
                u, i, r = data
            users.append(u)
            items.append(i)
            
            if self.includeRating:
                labels.append(r)
            else:
                labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)

                if self.useTime:
                    times.append(movieMeanViewTime[negative_item])
                labels.append(0)
        if self.useTime:
            return torch.tensor(users), torch.tensor(items), torch.tensor(times), torch.tensor(labels)
        else:
            return torch.tensor(users), torch.tensor(items), torch.tensor(labels)