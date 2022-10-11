from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import tensorflow as tf
import tempfile
from KG import KG
from multiG import multiG   # we don't import individual things in a model. This is to make auto reloading in Notebook happy
import modelR as model
from trainerR import Trainer
from OntoL import Onto2KG , LexicalMatch, removeInconsistincyAlignmnets
import yaml
from sklearn.metrics import recall_score, precision_score, f1_score

from testerR import Tester
#print(tf.version)
from tqdm import tqdm

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





params = {
    "embedding_size": 128,
    "epochs": 401,
    "batch_k": 64,
    "batch_a": 16,
    "a1": 1,
    "L1": 1,
    "lr": 0.01,
    "margin": 1,
    "AM_folds": 10,
    "topk": 10,
    "threshold": 0.1,

}



def generate_alignments(model_file, data_file):
    #model_file = params["model_path"]
    #data_file = params["data_path"]
    topk = params["topk"]
    threshold = params["threshold"]
    tester = Tester()
    tester.build(save_path = model_file, data_save_path = data_file)
    
    source_entities = list(tester.multiG.KG1.ents.keys())
    target_entities = list(tester.multiG.KG2.ents.keys())

    min_entities = min(len(source_entities), len(target_entities))//2
    #min_entities = 10
    target_entities_vectors = tester.vec_e[2]
        
    alignments = []

    acceptable_alignments = False

    
    while not acceptable_alignments:
        for class_ in source_entities:#tqdm(source_entities, total = len(source_entities)):
            class_url = tester.multiG.KG1.ent_index2str(class_)
            vec_proj_class = tester.projection(class_, source = 1)
            rst = tester.kNN_with_names(vec_proj_class, target_entities_vectors, topk)
            for i in range(topk):
                if(rst[i][1]<threshold):
                    if class_ in source_entities:
                        source_entities.remove(class_)

                    #print(class_url, rst[i][0], rst[i][1])
                    alignments.append([class_url, rst[i][0], "=", 1-rst[i][1]])

        if (len(alignments) >= min_entities) or threshold > 1.0 :
            acceptable_alignments = True
        else:
            print(f"Not enough alignments, trying higher threshold. Num of aligns: {len(alignments)}. Min entities: {min_entities}")
            
            threshold += 0.1
            print(f"")


    tester = Tester()
    tester.build(save_path = model_file, data_save_path = data_file)
    predictions = tester.predicted_alignments(5 ,0.1)
    ls = removeInconsistincyAlignmnets(source, target, predictions)
    print(ls)
    print("-------------------")
    print(f"Predictions found: {len(predictions)}")
    print("-------------------")
    print(f"Inconsistencies found: {len(ls)}")

    
    #shutil.rmtree("data/")
    ls = set([(x[0], x[1], "=", 1.0)  for x in ls])
    alignments = [tuple(x) for x in alignments]
    print(f"Number of original alignments: {len(alignments)}")
    alignments = list(set(alignments) - ls)
    print(f"Number of alignments after removing inconsistencies: {len(alignments)}")
    return alignments


def train_model(source_owl, target_owl):

    
    ####### Params accessing ###########
    this_dim = params["embedding_size"]
    #model_path = params["model_path"]
    #data_path = params["data_path"]

    model_tmp_file = tempfile.NamedTemporaryFile(delete=False)
    model_path = model_tmp_file.name

    data_tmp_file = tempfile.NamedTemporaryFile(delete=False)
    data_path = data_tmp_file.name
    
    batch_k = params["batch_k"]
    batch_a = params["batch_a"]
    a1 = params["a1"]
    L1 = params["L1"]
    lr = params["lr"]
    margin = params["margin"]
    AM_folds = params["AM_folds"]
    epochs = params["epochs"]
    #####################################

    ##### Knowledge Graph Building #####
    KG1 = Onto2KG(source_owl)
    KG2 = Onto2KG(target_owl)
    ####################################

    this_data = multiG(KG1, KG2)

    lexical_alignments_file = tempfile.NamedTemporaryFile(delete=False)
    lexical_alignments_file_name = lexical_alignments_file.name
    #lexical_alignments_file_name = "data/lexical_alignments"
    #if not os.path.exists(lexical_alignments_file_name+ ".json"):
    print("Computing lexical alignments")
    alignments, min_alignments = LexicalMatch(source_owl, target_owl, lexical_alignments_file_name)

    AM_folds = min_alignments #int(min_alignments//20)+1
    print("AM Folds: ", AM_folds)

    
    #print("Loading lexical alignments from file")
    #this_data.load_align_json(lexical_alignments_file_name+ ".json")
    this_data.load_align_list(alignments)
        

    m_train = Trainer()
    m_train.build(
        this_data,
        dim=this_dim,
        batch_sizeK=batch_k,
        batch_sizeA=batch_a,
        a1=a1, a2=0.5,
        m1=margin,
        save_path = model_path,
        multiG_save_path = data_path,
        L1=L1)
    
    m_train.train_MTransE(source_owl,
                          target_owl,
                          epochs=epochs,
                          save_every_epoch=10,
                          lr=lr, a1=a1,
                          a2=0.5,
                          m1=margin,
                          AM_fold=AM_folds,
                          half_loss_per_epoch=150)
    
    return model_path, data_path
    
if __name__ == "__main__":
    
    print("GPU Available: ", tf.test.is_gpu_available())
    


    if len(sys.argv) > 1: #from command line
        print(sys.argv)
        source = sys.argv[1]
        target = sys.argv[2]
        reference = sys.argv[3]
                    
        model_file, data_file = train_model(source, target)
        ranked_predicted_alignments(model_file, data_file, reference)
        
            
    else:
        print("ERROR: Calling from command line requires passing config file as first argument: python training_modelR.py config.yaml")
        

    



