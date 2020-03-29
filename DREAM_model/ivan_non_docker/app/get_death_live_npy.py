import pathlib
import csv
import collections as c

import sklearn as sk
import os
import sys
sys.path.extend(['/'])
if os.path.basename(os.getcwd()) != "app":
    os.chdir(os.getcwd() +'./app')
import time
import joblib as jl
import models as model_includes
import configs as model_configs
import numpy as np
import warnings
import random
from numpy import random
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
split_key = "id"
configs = model_configs.get_baseline_cv_configs()
config = list(configs.values())[0]
date_lags = config["date lags"]
data = model_includes.read_ehrdc_data(config["train path"])
person_items, uids_feats, uids_records = model_includes.get_grouped_features(data, config, uids_feats=None, key="id")
labels_individual = {k:v for k,v in zip(data["death"]["person_id"].copy(), data["death"]["label"].copy())}
death_individual ={k:v for k,v in labels_individual.items() if v == 1}
alive_individual = {k:v for k,v in labels_individual.items() if v == 0}
deathNumber = len(death_individual)
alive_person_translated = [person_items[uids_records[("person_id", k)]] for k in random.choice(list(alive_individual.keys()),deathNumber*5)]
death_person_translated = [person_items[uids_records[("person_id", k)]] for k in death_individual.keys()]
death_sparse = model_includes.get_dict_to_sparse(death_person_translated)
death_dense = death_sparse.todense()
np.save('death.npy',death_dense)
alive_sparse = model_includes.get_dict_to_sparse(alive_person_translated)
alive_dense = alive_sparse.todense()
np.save('alive.npy',alive_dense)
