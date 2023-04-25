


import torch
import torch.nn as nn
import pytorch_lightning as pl


class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)
    
        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_movieIds (list): List containing all movieIds (train + test)
    """
    
    def __init__(self, num_users, num_items, includeRating, num_months, useTime):

        super().__init__()
        embedding_out = 16
        self.useTime = useTime
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        if self.useTime:
            self.time_embedding = nn.Embedding(num_embeddings=num_months, embedding_dim=8)
            embedding_out += 8
        # self.includeRating = includeRating
        # self.rating_embedding = nn.Embedding(num_embeddings=6, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=embedding_out, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.includeRating = includeRating
        if self.includeRating:
            self.output = nn.Linear(in_features=32, out_features=11)
        self.outputB = nn.Linear(in_features=32, out_features=1)
        
    def forward(self, user_input, item_input, time_input):
        
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        if self.useTime:
            time_embedded = self.time_embedding(time_input)
        # Concat the two embedding layers
        if self.useTime:
            vector = torch.cat([user_embedded, item_embedded, time_embedded], dim=-1)
        else:
            vector = torch.cat([user_embedded, item_embedded], dim=-1)
        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        
        predB = nn.Sigmoid()(self.outputB(vector))
        if self.includeRating:  
            pred = nn.Sigmoid()(self.output(vector))
            return [predB, pred]


        return predB, 0
    


