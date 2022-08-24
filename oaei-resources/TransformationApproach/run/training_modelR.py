from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import tensorflow as tf

from KG import KG
from multiG import multiG   # we don't import individual things in a model. This is to make auto reloading in Notebook happy
import modelR as model
from trainerR import Trainer
from OntoL import Onto2KG , LexicalMatch
import yaml
#print(tf.version)


def get_embeddings(params, source_owl=None, target_owl=None):

    this_dim = params["embedding_size"]
    model_path = params["model_path"]
    data_path = params["data_path"]

    if source_owl is None:
        source_owl = params["source"]
    if target_owl is None:
        target_owl = params["target"]

    batch_k = params["batch_k"]
    batch_a = params["batch_a"]
    a1 = params["a1"]
    L1 = params["L1"]
    lr = params["lr"]
    margin = params["margin"]
    AM_folds = params["AM_folds"]
    
    KG1 = Onto2KG(source_owl,"source")
    KG2 = Onto2KG(target_owl,"target")
            
    this_data = multiG(KG1, KG2)
        
    this_data.load_align_json("alignments.json")
    
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
    
    m_train.train_MTransE(epochs=100,
                          save_every_epoch=100,
                          lr=lr, a1=a1,
                          a2=0.5,
                          m1=margin,
                          AM_fold=AM_folds,
                          half_loss_per_epoch=150)
    

if __name__ == "__main__":
    
    print("GPU Available: ", tf.test.is_gpu_available())
    


    if len(sys.argv) > 1: #from command line
        print(sys.argv)
        config_file = sys.argv[1]

        with open(config_file, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        get_embeddings(params)

    else:
        print("ERROR: Calling from command line requires passing config file as first argument: python training_modelR.py config.yaml")
        

    



