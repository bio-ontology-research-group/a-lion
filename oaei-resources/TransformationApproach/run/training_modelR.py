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

print("GPU Available: ", tf.test.is_gpu_available())

this_dim = 50


print(sys.argv)

if len(sys.argv) > 1:

    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    this_dim = params["embedding_size"]
    model_path = params["model_path"]
    data_path = params["data_path"]
    source = params["source"]
    target = params["target"]
    batch_k = params["batch_k"]
    batch_a = params["batch_a"]
    a1 = params["a1"]
    L1 = params["L1"]
    lr = params["lr"]
    margin = params["margin"]
    AM_folds = params["AM_folds"]
    

#alignf = LexicalMatch(source,target,'anatomy')
#this_data.load_align_list(alignf)

KG1 = Onto2KG(source,"source")
KG2 = Onto2KG(target,"target")
#alignments = LexicalMatch(source,target,"test")



print("------------------------------------------------------")
#import mowl
this_data = multiG(KG1, KG2)
#this_data.load_align_list(alignments)
# if alignemnt file exist use:
this_data.load_align_json("alignments.json")

m_train = Trainer()
# L1: if false L2norm is used
m_train.build(this_data, dim=this_dim, batch_sizeK=batch_k, batch_sizeA=batch_a, a1=a1, a2=0.5, m1=margin, save_path = model_path, multiG_save_path = data_path, L1=L1)
m_train.train_MTransE( epochs=100, save_every_epoch=100, lr=lr, a1=a1, a2=0.5, m1=margin, AM_fold=AM_folds, half_loss_per_epoch=150)



