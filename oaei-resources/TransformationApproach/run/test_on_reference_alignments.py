#testing a model for a given test set (normal ROC-analisys)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import os
if not os.path.exists('../results'):
    os.makedirs('../results')
import os
if not os.path.exists('../results/detail_mR'):
    os.makedirs('../results/detail_mR')
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

def ranked_predicted_alignments(params = None, config_file = None):

    if params is None:
        with open(config_file, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

    model_file = params["model_file"]
    data_file = params["data_file"]
    reference_alignment_file = params["reference_alignment_file"]
    topk = params["topk"]
    threshold = params["threshold"]
    tester = Tester()
    tester.build(save_path = model_file, data_save_path = data_file)


    
    source_entities = list(tester.multiG.KG1.ents.keys())
    terget_entities = list(tester.multiG.KG2.ents.keys())
    terget_entities_vectors = tester.vec_e[2]
    mappings_index = {}
    count=0
    for class_ in source_entities:
        class_url = tester.multiG.KG1.ent_index2str(class_)
        vec_proj_class = tester.projection(class_, source = 1)
        rst = tester.kNN_with_names(vec_proj_class, terget_entities_vectors, topk)
        for i in range(topk):
            if(rst[i][1]<threshold):
                tuple_ = (class_url,rst[i][0])
                if(tuple_ not in mappings_index):
                    mappings_index[tuple_] = [0,1]
                    count+=1
                else:
                    mappings_index[tuple_][1] = 1
            else:
                continue

    with open(reference_alignment_file) as f:
        content = f.readlines()

    for p in content:
        line = p.split()
        tuple_ = (line[0], line[1])
        if(tuple_ not in mappings_index):
            mappings_index[tuple_] = [1,0]
            count+=1
        else:
            mappings_index[tuple_][0] = 1

    X = []
    Y = []
    for k in mappings_index:
        X.append(mappings_index[k][1])
        Y.append(mappings_index[k][0])

    print("recal",recall_score(X,Y) )
    print("precision", precision_score(X,Y))
    print("f1_score", f1_score(X,Y))


if __name__ == "__main__":
    config_file = sys.argv[1]

    lst = ranked_predicted_alignments(config_file = config_file)
