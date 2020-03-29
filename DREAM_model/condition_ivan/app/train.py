import pandas as pd
import numpy as np
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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
t = time.time()
split_key = "id"
configs = model_configs.get_baseline_cv_configs()
config = list(configs.values())[0]
load_only=False
tt = time.time()
data = model_includes.read_ehrdc_data(config["train path"])
print("Data load time:" + str(time.time()-tt), flush=True)
if "train" in config and config["train"]:
    print("Running: " + config["model name"] + "," + config["cv split key"])
    if config["model name"] == "static demographic baseline":
        config_trained = model_includes.model_static_patient_train(data, data["death"]["label"], config)
        jl.dump(config_trained, config["model path"] + "config.joblib")
    elif config["model name"] == "static uid model selection":
        configs = model_configs.get_baseline_cv_configs()
        if load_only:
            x_train, x_test, y_train, y_test, keys_train, keys_test,data_sp = model_includes.preprocess_data(data, configs, split_key="id")
            #print("data_sp saved", flush = True)
            #np.save('data_sp.npy',data_sp)
        else:
            config_select, selected, perf, metrics_out, configs, uids = model_includes.model_sparse_feature_cv_train(data, configs, split_key=split_key)
            model_configs.pickle_nms(config_select, config["model path"] + "config.joblib" )
            #jl.dump(config_select, config["model path"] + "config.joblib")
            jl.dump(uids, config["scratch path"] + "uids.joblib")
            del config, configs, data, perf, selected, uids, config_select, metrics_out
print("Total time:" + str(time.time()-t), flush=True)
