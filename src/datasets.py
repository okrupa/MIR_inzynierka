import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy

from typing import Tuple, List, Dict
from collections import defaultdict

from music_fsl.data import ClassConditionalDataset
from music_fsl.backbone import Backbone
from music_fsl.protonet import PrototypicalNet


import librosa

import streamlit as st
import soundfile as sf
import io

class GivenData(ClassConditionalDataset):

    def __init__(self, 
            instruments: List[str] = None,
            duration: float = 1.0, 
            sample_rate: int = 16000,
            dataset_list: str = None,
            query_set: bool = False
        ):

        self.instruments = instruments  
        self.duration = duration
        self.sample_rate = sample_rate
        self.dataset_list = dataset_list
        self.query_set = query_set
        # load all tracks for this instrument
        self.tracks = []
        if query_set:
          for audio in dataset_list:
            self.tracks.append([audio, None, audio.name])
        
        else:
          for count, value in enumerate(instruments):
            for audio in dataset_list[count]:
              self.tracks.append([audio, value])

    @property
    def classlist(self) -> List[str]:
        return self.instruments

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        # cache it in self._class_to_indices 
        # so we don't have to recompute it every time
        if not hasattr(self, "_class_to_indices"):
            self._class_to_indices = defaultdict(list)
            for i, track in enumerate(self.tracks):
                self._class_to_indices[track[1]].append(i)

        return self._class_to_indices

    def load_excerpt(self, file, duration: float, sample_rate: int):
      """
      Load an excerpt of audio from a file.

      Returns a dictionary with the following keys:
          - audio: a torch.Tensor of shape (1, samples)
          - offset: the offset (in seconds) of the excerpt
          - duration: the duration (in seconds) of the excerpt
      """
      audio_bytes = file.getvalue()
      y, sr = librosa.load(path=io.BytesIO(audio_bytes))
      total_duration = librosa.get_duration(y=y, sr=sr)
      if total_duration < duration:
          raise ValueError(f"Audio file is too short"
                f"to extract an excerpt of duration {duration}")
      offset = random.uniform(0, total_duration - duration)
      audio, sr = librosa.load(path=io.BytesIO(audio_bytes), sr=sample_rate, 
                              offset=offset, duration=duration, 
                              mono=True)
      if audio.ndim == 1:
          audio = audio[None, :]
      return {
          "audio": torch.tensor(audio), 
          "offset": offset, 
          "duration": duration
      }
    def __getitem__(self, index) -> Dict:
        # load the track for this index
        track = self.tracks[index]

        # load the excerpt
        data = self.load_excerpt(track[0], self.duration, self.sample_rate)
        data["label"] = track[1]
        if self.query_set:
          data["name"] = track[2]
        return data

    def __len__(self) -> int:
        return len(self.tracks)

import random
import torch
import music_fsl.util as util

from typing import Tuple, Dict
class EpisodeDataset(torch.utils.data.Dataset):
    """
        A dataset for sampling few-shot learning tasks from a class-conditional dataset.

    Args:
        dataset (ClassConditionalDataset): The dataset to sample episodes from.
        n_way (int): The number of classes to sample per episode.
            Default: 5.
        n_support (int): The number of samples per class to use as support.
            Default: 5.
        n_query (int): The number of samples per class to use as query.
            Default: 20.
        n_episodes (int): The number of episodes to generate.
            Default: 100.
    """
    def __init__(self,
        dataset: ClassConditionalDataset, 
        n_way: int = 5, 
        n_support: int = 5,
        n_query: int = 20,
        n_episodes: int = 100,
    ):
        self.dataset = dataset

        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episodes = n_episodes
    
    def __getitem__(self, index: int) -> Tuple[Dict, Dict]:
        """Sample an episode from the class-conditional dataset. 

        Each episode is a tuple of two dictionaries: a support set and a query set.
        The support set contains a set of samples from each of the classes in the
        episode, and the query set contains another set of samples from each of the
        classes. The class labels are added to each item in the support and query
        sets, and the list of classes is also included in each dictionary.

        Yields:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the support
            set and the query set for an episode.
        """
        # seed the random number generator so we can reproduce this episode
        rng = random.Random(index)

        # sample the list of classes for this episode
        episode_classlist = rng.sample(self.dataset.classlist, self.n_way)

        # sample the support and query sets for this episode
        support, query = [], []
        for c in episode_classlist:
            # grab the dataset indices for this class
            all_indices = self.dataset.class_to_indices[c]
            # sample the support and query sets for this class
            indices = rng.sample(all_indices, self.n_support + self.n_query)
            items = [self.dataset[i] for i in indices]

            # add the class label to each item
            for item in items:
                item["target"] = torch.tensor(episode_classlist.index(c))

            # split the support and query sets
            support.extend(items[:self.n_support])
            query.extend(items[self.n_support:])

        # collate the support and query sets
        support = util.collate_list_of_dicts(support)
        query = util.collate_list_of_dicts(query)

        support["classlist"] = episode_classlist
        query["classlist"] = episode_classlist
        
        return support, query

    def __len__(self):
        return self.n_episodes

    def print_episode(self, support, query):
        """Print a summary of the support and query sets for an episode.

        Args:
            support (Dict[str, Any]): The support set for an episode.
            query (Dict[str, Any]): The query set for an episode.
        """
        print("Support Set:")
        print(f"  Classlist: {support['classlist']}")
        print(f"  Audio Shape: {support['audio'].shape}")
        print(f"  Target Shape: {support['target'].shape}")
        print()
        print("Query Set:")
        print(f"  Classlist: {query['classlist']}")
        print(f"  Audio Shape: {query['audio'].shape}")
        print(f"  Target Shape: {query['target'].shape}")

    def predict_subsets(self, index: int, query_dataset) -> Tuple[Dict, Dict]:
        """Sample an episode from the class-conditional dataset. 

        Each episode is a tuple of two dictionaries: a support set and a query set.
        The support set contains a set of samples from each of the classes in the
        episode, and the query set contains another set of samples from each of the
        classes. The class labels are added to each item in the support and query
        sets, and the list of classes is also included in each dictionary.

        Yields:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the support
            set and the query set for an episode.
        """
        # seed the random number generator so we can reproduce this episode
        rng = random.Random(index)

        # sample the list of classes for this episode
        episode_classlist = self.dataset.classlist

        # sample the support and query sets for this episode

        all_indices_query = query_dataset.class_to_indices[None]
        
        query = [query_dataset[i] for i in all_indices_query]
        support = []
        for c in episode_classlist:
            # grab the dataset indices for this class
            all_indices = self.dataset.class_to_indices[c]
            # sample the support and query sets for this class
            indices = rng.sample(all_indices, self.n_support)
            items = [self.dataset[i] for i in indices]

            # add the class label to each item
            for item in items:
                item["target"] = torch.tensor(episode_classlist.index(c))

            # split the support and query sets
            support.extend(items)

        # collate the support set
        support = util.collate_list_of_dicts(support)
        query = util.collate_list_of_dicts(query)
        support["classlist"] = episode_classlist
        
        return support, query, episode_classlist

def prepare_datasets(input_data, predict_set):
    instruments = []
    dataset_list = []
    for x in input_data:
        instruments.append(x[0])
        dataset_list.append(x[1])
    dataset = GivenData(
        instruments=instruments, 
        sample_rate=16000,
        dataset_list = dataset_list
      )

    dataset_query = GivenData(
        sample_rate=16000,
        dataset_list = predict_set,
        query_set=True
      )
    return dataset, dataset_query