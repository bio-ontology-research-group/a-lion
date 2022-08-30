#testing a model for a given test set (normal ROC-analisys)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import os
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from numpy import linalg as LA
import pkg_resources
pkg_resources.require("tensorflow==1.15.5")
import tensorflow as tf
import time
import multiG  
import modelR as model
from testerR import Tester
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import recall_score, precision_score, f1_score
import yaml
from tqdm import tqdm


params = {
    "model_path": "model_file",
    "data_path": "data_file",
    "topk": 10,
    "threshold": 0.5,
    "test_name": "test_R_1_5",

}



def generate_alignments():
    model_file = params["model_path"]
    data_file = params["data_path"]
    topk = params["topk"]
    threshold = params["threshold"]
    tester = Tester()
    tester.build(save_path = model_file, data_save_path = data_file)


    
    source_entities = list(tester.multiG.KG1.ents.keys())
    target_entities = list(tester.multiG.KG2.ents.keys())
    target_entities_vectors = tester.vec_e[2]
    mappings_index = {}
    count=0
    alignments = []
    for class_ in tqdm(source_entities, total = len(source_entities)):
        class_url = tester.multiG.KG1.ent_index2str(class_)
        vec_proj_class = tester.projection(class_, source = 1)
        rst = tester.kNN_with_names(vec_proj_class, target_entities_vectors, topk)
        for i in range(topk):
            if(rst[i][1]<threshold):
                alignments.append(class_url, rst[i][0], "=", 1.0)
    return alignments

if __name__ == "__main__":
    config_file = sys.argv[1]

    lst = generate_alignments(config_file)
