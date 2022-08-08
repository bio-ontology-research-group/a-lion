import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve,auc, precision_recall_curve, average_precision_score
import numpy as np
import sys


def evaluate(input_file):

    mappings_index = {}
    count = 0


    with open(input_file) as f:
        content = f.readlines()[1:]

    for p in content:
        line = p.split()
        tuple_ = (line[0], line[1])
        score = float(line[2])
        if(tuple_ not in mappings_index):
    	    mappings_index[tuple_] = [0,score]
    	    count+=1
        else:
    	    mappings_index[tuple_][1] = score




    with open("../data/refrence.tsv") as f:
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

    fpr, tpr, thresholds = roc_curve(Y, X, pos_label=1)
    avg_prec = average_precision_score(Y,X, pos_label = 1)
    return avg_prec, auc(fpr, tpr)
