# import necessary packages

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# load dataset

movies= pd.read_csv("ml_data/movies.dat", encoding="cp1252", delimiter="::", header=None, names=["movie_id", "title", "genres"])
ratings=  pd.read_csv("ml_data/ratings.dat", encoding="cp1252", delimiter="::", header=None, names=["user_id", "movie_id", "rating", "timestamp"])
users=  pd.read_csv("ml_data/users.dat", encoding="cp1252", delimiter="::", header=None, names=["user_id", "gender", "age", "occupation", "zip_code"])

# base dataset
base_dt= pd.merge(ratings,movies, on=["movie_id"], how="left")
base_dt= pd.merge(base_dt,users, on=["user_id"], how="left")

# encoding string columns
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

base_dt["userId"] = user_encoder.fit_transform(base_dt["user_id"])
base_dt["movieId"] = movie_encoder.fit_transform(base_dt["movie_id"])

num_users = base_dt["userId"].nunique()
num_movies = base_dt["movieId"].nunique()

# split train, test dataset
train_dt, test_dt= train_test_split(base_dt, test_size=0.2, random_state=731)

# setting modeling part
class nn_linear_bandit_model(nn.Module):

    def __init__(self, num_users, num_movies, embedding_dim=8, user_feature_dim=4, movie_feature_dim=4):
        super(nn_linear_bandit_model, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.user_linear = nn.Linear(user_feature_dim, embedding_dim)
        self.movie_linear = nn.Linear(movie_feature_dim, embedding_dim)

        self.mlp = nn.Sequential(

            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()

        )

    def forward(self, user_ids, movie_ids, user_features, movie_features):
        user_emb= self.user_embedding(user_ids) + self.user_feature_layer(user_features)
        movie_emb = self.movie_embedding(movie_ids) + self.movie_feature_layer(movie_features)
        x= torch.cat([user_emb, movie_emb], dim=-1)
        return self.mlp(x)


# setting train model
def train_neural_linear_bandit_model(model, samples, epochs=3, batch_size=32):
    optimizer= optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    model.train()

    loss_history = []  # save loss records

    for epoch in range(epochs):

        total_loss=0

        for i in range(0, len(samples), batch_size):
            batch= samples[i:i+batch_size]
            user_ids, movie_ids, user_features, movie_features, labels = zip(*batch)

            user_ids= torch.tensor(user_ids, dtype=torch.long)
            movie_ids= torch.tensor(movie_ids, dtype=torch.long)
            user_features= torch.tensor(user_features, dtype=torch.float32)
            movie_features= torch.tensor(movie_features, dtype=torch.float32)
            labels= torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

            optimizer.zero_grad()
            predictions= model(user_ids, movie_ids, user_features, movie_features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()

        print(f"Epoch {epoch}/{epochs}, Loss: {total_loss}")

    # Save the model
    torch.save(model.state_dict(), "neural_linear_bandit_model.pth")
    print("Model saved as neural_linear_bandit_model.pth")

    # Plot the loss history
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.grid()
    plt.show()

# param
embedding_dim= 8
user_feature_dim= 4
movie_feature_dim= 4

# train model
# model = nn_linear_bandit_model(num_users, num_movies, embedding_dim, user_feature_dim, movie_feature_dim)
# train_neural_linear_bandit_model(model, train_dt)


# debug
# print(base_dt.head())
# print(base_dt.info())
# print(len(base_dt))

# print(train_dt.head())
# print(train_dt.info())
# print(len(train_dt))

