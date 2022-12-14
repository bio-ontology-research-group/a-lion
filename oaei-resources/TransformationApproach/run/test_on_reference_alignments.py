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
from generate_alignments import generate_alignments

def ranked_predicted_alignments(model_file, data_file, reference_alignment_file):

    pred_alignments = generate_alignments(model_file, data_file)

    mappings_index = {}

    for al in pred_alignments:
        tuple_ = (al[0], al[1])
        
        mappings_index[tuple_] = [0,1]


    with open(reference_alignment_file) as f:
        content = f.readlines()

    for p in content:
        line = p.split()
        tuple_ = (line[0], line[1])
        if(tuple_ not in mappings_index):
            mappings_index[tuple_] = [1,0]
            
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
