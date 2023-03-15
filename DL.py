import torch
from torch.utils.data import Dataset, DataLoader
from DL_ratings import RatingDataset

class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training
    
    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds
    
    """

    def __init__(self, ratings, all_movieIds):
        num_negatives = 4
        ratingDS = RatingDataset(ratings, all_movieIds, num_negatives)
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds, ratingDS)
        

    def __len__(self):
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_movieIds, ratingDS):
        batch_size = 10000
        train_loader = DataLoader(ratingDS, batch_size=batch_size, shuffle=True)
        print(len(train_loader))
        users, items, labels = [], [], []
        # Iterate over the data in batches
        for batch_idx, (batch_pos, batch_neg) in enumerate(train_loader):
            if (batch_idx % 20 == 0):
                print(batch_idx / len(train_loader))
            # Process the batch here
            batch_users_pos = batch_pos[0].tolist()
            batch_items_pos = batch_pos[1].tolist()
            batch_labels_pos = batch_pos[2].tolist()
            
            # Add positive samples to the training data
            users += batch_users_pos
            items += batch_items_pos
            labels += batch_labels_pos
            
            # Add negative samples to the training data
            for batch_neg_sample in batch_neg:
                batch_users_neg = batch_neg_sample[0].tolist()
                batch_items_neg = batch_neg_sample[1].tolist()
                batch_labels_neg = batch_neg_sample[2].tolist()
                users += batch_users_neg
                items += batch_items_neg
                labels += batch_labels_neg

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)