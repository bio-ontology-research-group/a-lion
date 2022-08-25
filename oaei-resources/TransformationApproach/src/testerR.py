''' Module for held-out test.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import heapq as HP
import sys

import multiG  
import modelR as model
import trainerR as trainer

# This class is used to load and combine a TF_Parts and a Data object, and provides some useful methods for training
class Tester(object):
    def __init__(self):
        self.tf_parts = None
        self.multiG = None
        self.vec_e = {}
        self.vec_r = {}
        self.mat = np.array([0])
        # below for test data
        self.test_align = np.array([0])
        self.test_align_rel = []
        self.aligned = {1: set([]), 2: set([])}
        # L1 to L2 map
        self.lr_map = {}
        self.lr_map_rel = {}
        # L2 to L1 map
        self.rl_map = {}
        self.rl_map_rel = {}
        self.sess = None
    
    def build(self, save_path = 'this-model.ckpt', data_save_path = 'this-data.bin'):
        self.multiG = multiG.multiG()
        self.multiG.load(data_save_path)
        self.tf_parts = model.TFParts(num_rels1=self.multiG.KG1.num_rels(),
                                 num_ents1=self.multiG.KG1.num_ents(),
                                 num_rels2=self.multiG.KG2.num_rels(),
                                 num_ents2=self.multiG.KG2.num_ents(),
                                 dim=self.multiG.dim,
                                 #batch_sizeK=self.batch_sizeK,
                                 #batch_sizeA=self.batch_sizeA,
                                 L1=self.multiG.L1)
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, save_path)  # load it
        value_ht1, value_r1, value_ht2, value_r2, value_M= sess.run([self.tf_parts._ht1_norm, self.tf_parts._r1, self.tf_parts._ht2_norm, self.tf_parts._r2, self.tf_parts._M])  # extract values.
        self.vec_e[1] = np.array(value_ht1)
        self.vec_e[2] = np.array(value_ht2)
        self.vec_r[1] = np.array(value_r1)
        self.vec_r[2] = np.array(value_r2)
        self.mat = np.array(value_M)

   
    # by default, return head_mat
    def get_mat(self):
        return self.mat
    
    def ent_index2vec(self, e, source):
        assert (source in set([1, 2]))
        return self.vec_e[source][int(e)]

    def rel_index2vec(self, r, source):
        assert (source in set([1, 2]))
        return self.vec_r[source][int(r)]

    def ent_str2vec(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        this_index = KG.ent_str2index(str)
        if this_index == None:
            return None
        return self.vec_e[source][this_index]
    
    def rel_str2vec(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        this_index = KG.rel_str2index(str)
        if this_index == None:
            return None
        return self.vec_r[source][this_index]
    
    class index_dist:
        def __init__(self, index, dist):
            self.dist = dist
            self.index = index
            return
        def __lt__(self, other):
            return self.dist > other.dist
                
    def ent_index2str(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        return KG.ent_index2str(str)
    
    def rel_index2str(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        return KG.rel_index2str(str)

    def ent_str2index(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        return KG.ent_str2index(str)
    
    def rel_str2index(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        return KG.rel_str2index(str)
    
    # input must contain a pool of vecs. return a list of indices and dist
    def kNN(self, vec, vec_pool, topk=10, self_id=None, except_ids=None, limit_ids=None):
        q = []
        for i in range(len(vec_pool)):
            #skip self
            if i == self_id or ((not except_ids is None) and i in except_ids):
                continue
            if (not limit_ids is None) and i not in limit_ids:
                continue
            dist = LA.norm(vec - vec_pool[i], ord=(1 if self.multiG.L1 else 2))
            if len(q) < topk:
                HP.heappush(q, self.index_dist(i, dist))
            else:
                #indeed it fetches the biggest
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist > dist:
                    HP.heapreplace(q, self.index_dist(i, dist) )
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, (item.index, item.dist))
        return rst

    def kNN_with_names(self, vec, vec_pool, topk=10, self_id=None, except_ids=None, limit_ids=None):
        #print("limited ",limit_ids)
        q = []
        for i in range(len(vec_pool)):
            #skip self
            if i == self_id or ((not except_ids is None) and i in except_ids):
                continue
            if (not limit_ids is None) and self.multiG.KG2.ent_index2str(i) not in limit_ids:
                continue
            else:
                #print(i, self.multiG.KG2.ent_index2str(i), type(self.multiG.KG2.ent_index2str(i)))
                dist = LA.norm(vec - vec_pool[i], ord=(1 if self.multiG.L1 else 2))
                if len(q) < topk:
                    HP.heappush(q, self.index_dist(self.multiG.KG2.ent_index2str(i), dist))
                else:
                    #indeed it fetches the biggest
                    tmp = HP.nsmallest(1, q)[0]
                    if tmp.dist > dist:
                        HP.heapreplace(q, self.index_dist(self.multiG.KG2.ent_index2str(i), dist) )
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, (item.index, item.dist))
        return rst


    # input must contain a pool of vecs. return a list of indices and dist
    def NN(self, vec, vec_pool, self_id=None, except_ids=None, limit_ids=None):
        min_dist = sys.maxint
        rst = None
        for i in range(len(vec_pool)):
            #skip self
            if i == self_id or ((not except_ids is None) and i in except_ids):
                continue
            if (not limit_ids is None) and i not in limit_ids:
                continue
            dist = LA.norm(vec - vec_pool[i], ord=(1 if self.multiG.L1 else 2))
            if dist < min_dist:
                min_dist = dist
                rst = i
        return (rst, min_dist)
        
    # input must contain a pool of vecs. return a list of indices and dist. rank an index in a vec_pool from 
    def rank_index_from(self, vec, vec_pool, index, self_id = None, except_ids=None, limit_ids=None):
        dist = LA.norm(vec - vec_pool[index], ord=(1 if self.multiG.L1 else 2))
        rank = 1
        for i in range(len(vec_pool)):
            if i == index or i == self_id or ((not except_ids is None) and i in except_ids):
                continue
            if (not limit_ids is None) and i not in limit_ids:
                continue
            if dist > LA.norm(vec - vec_pool[i], ord=(1 if self.multiG.L1 else 2)):
                rank += 1
        return rank

    # Change if AM changes
    def projection(self, e, source):
        assert (source in set([1, 2]))
        vec_e = self.ent_index2vec(e, source)
        #return np.add(np.dot(vec_e, self.mat), self._b)
        return np.dot(vec_e, self.mat)

    def projection_rel(self, r, source):
        assert (source in set([1, 2]))
        vec_r = self.rel_index2vec(r, source)
        #return np.add(np.dot(vec_e, self.mat), self._b)
        return np.dot(vec_r, self.mat)

    def projection_vec(self, vec, source):
        assert (source in set([1, 2]))
        #return np.add(np.dot(vec_e, self.mat), self._b)
        return np.dot(vec, self.mat)
    
    # Currently supporting only lan1 to lan2
    def projection_pool(self, ht_vec):
        #return np.add(np.dot(ht_vec, self.mat), self._b)
        return np.dot(ht_vec, self.mat)