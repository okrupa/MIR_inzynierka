from pathlib import Path
import streamlit as st

import numpy as np
import torch
from torch import nn
import tqdm
from torchmetrics import Accuracy

from music_fsl.util import dim_reduce, embedding_plot, batch_device
from music_fsl.backbone import Backbone
from music_fsl.protonet import PrototypicalNet
from music_fsl.train import FewShotLearner

from datasets import GivenData, EpisodeDataset, prepare_datasets

import csv
import uuid

#Klasa przeznaczona do dotrenowania wybranej sieci oraz predykcji zbioru danych
class PredictionPrototypicalNet():
    def __init__(self, 
            checkpoint_path: str,
            uploaded_files,
            uploaded_predict_set,
            n_way: int = 3, 
            n_support: int = 5,
            n_query:int = 15,
            n_episodes:int = 50, 
            sample_rate: int = 16000,
            
        ):
        self.checkpoint_path = checkpoint_path
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episodes = n_episodes

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.protonet = PrototypicalNet(Backbone(sample_rate))
        self.learner = FewShotLearner.load_from_checkpoint(self.checkpoint_path, protonet=self.protonet)
        self.learner.eval()
        self.learner = self.learner.to(self.DEVICE)

        self.dataset, self.dataset_query = prepare_datasets(uploaded_files, uploaded_predict_set)
        self.train_episodes = EpisodeDataset(
            dataset=self.dataset, 
            n_way=self.n_way, 
            n_support=self.n_support,
            n_query=self.n_query, 
            n_episodes=self.n_episodes
        )

        # support, query, episode_classlist = self.train_episodes.predict_subsets(124, self.dataset_query)

        self.metric = Accuracy(num_classes=self.n_way, average="samples")

    #Dotrenowanie modelu na wyznaczonej liczbie epok, następnie sprawdzenie średniej 
    def train(self):
        progress_bar = st.progress(0)
        optimizer = torch.optim.Adam(self.learner.parameters(), lr=self.learner.learning_rate)
        for episode_idx in range(self.n_episodes):
            support, query = self.train_episodes[episode_idx]

            # move all tensors to cuda if necessary
            batch_device(support, self.DEVICE)
            batch_device(query, self.DEVICE)

            # get the embeddings
            logits = self.learner.protonet(support, query)
            # compute the accuracy
            acc = self.metric(logits, query["target"])

            loss = self.learner.loss(logits, query["target"])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  

            st.write(f"Episode {episode_idx} // Accuracy: {acc.item():.2f}")
            progress_bar.progress((100/self.n_episodes)*(episode_idx+1)/100)

    def predict(self):
        # predict input data
        sig = nn.Sigmoid()
        support, query, episode_classlist = self.train_episodes.predict_subsets(125, self.dataset_query)

        # move all tensors to cuda if necessary
        batch_device(support, self.DEVICE)
        batch_device(query, self.DEVICE)

        # get the embeddings
        logits = self.learner.protonet(support, query)

        names = query["name"]

        outputs = sig(logits)
        _, predicted = torch.max(outputs, 1)
        # write each prediction and save to file
        new_title = f'<p style="font-family:serif; color:Black; font-size: 22px;">Prediction</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        path = f'.//outputs/{str(uuid.uuid4())}.csv'
        f = open(path, 'w')
        writer = csv.writer(f)
        writer.writerow(['file', 'instrument'])
        for name, prediction in zip (names, predicted):
            st.write(f"For file: {name} prediction is {episode_classlist[prediction]}")
            writer.writerow([name, episode_classlist[prediction]])
        f.close()
        with open(path) as f:
            st.download_button('Download prediction', f)
            
