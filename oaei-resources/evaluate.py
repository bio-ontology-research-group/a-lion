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
        X.append(mappings_index[k][1]) #pred
        Y.append(mappings_index[k][0]) #true


    #print(X,Y)
    

    fpr, tpr, thresholds = roc_curve(Y, X, pos_label=1)
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    roc_auc = auc(fpr, tpr)
    best_threshold = thresholds[ix]

    avg_prec = average_precision_score(Y,X, pos_label = 1)
    X = [1 if x>=best_threshold else 0 for x in X]
    
    

    print("best_threshold", best_threshold)
    print("recal",recall_score(Y,X) )
    print("precision", precision_score(Y,X))
    print("f1_score", f1_score(Y,X))    

    return avg_prec, roc_auc
