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

#print(tf.version)

print("GPU Available: ", tf.test.is_gpu_available())

this_dim = 50

print(sys.argv)

if len(sys.argv) > 1:
    this_dim = int(sys.argv[1])
    model_path = sys.argv[2]	#saving the model with this name
    data_path = sys.argv[3]		#saving name for the graph
    source = sys.argv[4]			#in form h	t	r\n
    target =  sys.argv[5]
    batch_k = int(sys.argv[6])
    batch_a = int(sys.argv[7])
    a1 = float(sys.argv[8])
    L1 = int(sys.argv[9])
    lr = float(sys.argv[10])
    margin = float(sys.argv[11])
    AM_folds = int(sys.argv[12])


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



