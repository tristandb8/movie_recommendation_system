import torch
from torch.utils.data import Dataset
import sys

class RatingDataset(Dataset):
    def __init__(self, ratings, all_movieIds, num_negatives):
        self.users_pos = torch.tensor(ratings['userId'].values, dtype=torch.long)
        self.items_pos = torch.tensor(ratings['movieId'].values, dtype=torch.long)
        self.all_movieIds = all_movieIds
        self.num_negatives = num_negatives
        
        # Get a list of unique user IDs
        self.unique_users = ratings['userId'].unique()
        num_users = len(self.unique_users)
        
        # Generate negative samples for all user-item pairs
        self.negatives = {}
        count = 0
        for user in self.unique_users:
            count += 1
            if (count % 100 == 0):
                print("Progress on RatingDataset init: ",  int(count/num_users *100), "%")
                sys.stdout.flush()
            items_neg = []
            user_item_set = set(zip(self.users_pos.tolist(), self.items_pos.tolist()))
            while len(items_neg) < num_negatives:
                negative_item = torch.randint(0, len(all_movieIds), (1,)).item()
                if (user, negative_item) not in user_item_set:
                    items_neg.append(negative_item)
            self.negatives[user] = items_neg

    def __len__(self):
        return len(self.users_pos)

    def __getitem__(self, idx):
        user_pos = self.users_pos[idx]
        item_pos = self.items_pos[idx]
        
        # Get the pre-generated negative samples for this user
        items_neg = self.negatives[user_pos.item()]

        # Return user-item-positive_item, user-item-negative_item pairs
        return (user_pos, item_pos, 1), [(user_pos, item_neg, 0) for item_neg in items_neg]
