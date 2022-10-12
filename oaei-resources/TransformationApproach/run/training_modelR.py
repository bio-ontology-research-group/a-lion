from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import click as ck
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
# print(tf.version)
from tqdm import tqdm

def ranked_predicted_alignments(model_file, data_file, reference_alignment_file, source, target, topk, min_threshold, max_threshold, root=None):

    pred_alignments = generate_alignments(model_file, data_file, source, target, topk, min_threshold, max_threshold, root=root)

    mappings_index = {}

    for al in pred_alignments:
        tuple_ = (al[0], al[1])

        mappings_index[tuple_] = [0, 1]

    with open(reference_alignment_file) as f:
        content = f.readlines()

    for p in content:
        line = p.split()
        tuple_ = (line[0], line[1])
        if(tuple_ not in mappings_index):
            mappings_index[tuple_] = [1, 0]

        else:
            mappings_index[tuple_][0] = 1

    X = []
    Y = []
    for k in mappings_index:
        X.append(mappings_index[k][1])
        Y.append(mappings_index[k][0])

    recall = recall_score(X, Y)
    precision = precision_score(X, Y)
    f1 = f1_score(X, Y)
    print("recal", recall)
    print("precision", precision)
    print("f1_score", f1)

    
    return recall, precision, f1


def generate_alignments(model_file, data_file, source, target, topk, min_threshold, max_threshold, root=None):
    #model_file = params["model_path"]
    #data_file = params["data_path"]
    if root is None or not isinstance(root, str):
        raise TypeError("Parameter outfile_path must be of type stra")

    tester = Tester()
    tester.build(save_path=model_file, data_save_path=data_file)

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
            vec_proj_class = tester.projection(class_, source=1)
            rst = tester.kNN_with_names(vec_proj_class, target_entities_vectors, topk)
            for i in range(topk):
                if(rst[i][1] < min_threshold):
                    if class_ in source_entities:
                        source_entities.remove(class_)

                    #print(class_url, rst[i][0], rst[i][1])
                    alignments.append([class_url, rst[i][0], "=", 1.0])

        if (len(alignments) >= min_entities) or min_threshold > max_threshold:
            acceptable_alignments = True
        else:
            print(f"Not enough alignments, trying higher threshold. Num of aligns: {len(alignments)}. Min entities: {min_entities}")

            min_threshold += 0.1
            print(f"")

    tester = Tester()
    tester.build(save_path=model_file, data_save_path=data_file)
    predictions = tester.predicted_alignments(5 , 0.1)
    ls = removeInconsistincyAlignmnets(source, target, predictions)
    print(ls)
    print("-------------------")
    print(f"Predictions found: {len(predictions)}")
    print("-------------------")
    print(f"Inconsistencies found: {len(ls)}")

    # shutil.rmtree("data/")
    ls = set([(x[0], x[1], "=", 1.0)  for x in ls])
    alignments = [tuple(x) for x in alignments]
    print(f"Number of original alignments: {len(alignments)}")
    alignments = list(set(alignments) - ls)
    print(f"Number of alignments after removing inconsistencies: {len(alignments)}")

    outfile_path = os.path.join(root, "predicted_alignments.txt")
    with open(outfile_path, "w") as f:
        f.write("SrcEntity\tTgtEntity\tScore\n")
        for src, dst, _, score in alignments:
            f.write(f"{src}\t{dst}\t{score}\n")

    return alignments


def train_model(source_owl,
                target_owl,
                embedding_size,
                epochs,
                batch_k,
                batch_a,
                a1,
                L1,
                lr,
                margin,
                AM_folds,
                root=None):

    ####### Params accessing ###########
    this_dim = embedding_size
    #model_path = params["model_path"]
    #data_path = params["data_path"]

    model_path = os.path.join(root, "modelbin")
    data_path = os.path.join(root, "databin")

    #####################################

    ##### Knowledge Graph Building #####
    KG1 = Onto2KG(source_owl)
    KG2 = Onto2KG(target_owl)
    ####################################

    this_data = multiG(KG1, KG2)

    lexical_alignments_file_name = os.path.join(root, "lexical_alignments.txt")
    #lexical_alignments_file_name = "data/lexical_alignments"
    # if not os.path.exists(lexical_alignments_file_name+ ".json"):
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
        save_path=model_path,
        multiG_save_path=data_path,
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


@ck.command()
@ck.option('--source', "-s", type=str, help='Source ontology file')
@ck.option('--target', "-t", type=str, help='Target ontology file')
@ck.option('--reference', "-r", type=str, help='Reference alignment file')
@ck.option('--aim', "-aim", default="all", type=str)
@ck.option("--embedding-size", "-size", default=128, type=int)
@ck.option("--epochs", "-e", default=401, type=int)
@ck.option("--batch-k", "-bsk", default=64, type=int)
@ck.option("--batch-a", "-bsa", default=16, type=int)
@ck.option("--a1", "-a1", default=1., type=int)
@ck.option("--l1", "-l1", default=1, type=int)
@ck.option("--lr", "-lr", default=0.01, type=float)
@ck.option("--margin", "-m", default=1, type=float)
@ck.option("--am-folds", "-am", default=10, type=int)
@ck.option("--topk", "-k", default=10, type=int)
@ck.option("--min-threshold", "-minth", default=0.1, type=float)
@ck.option("--max-threshold", "-maxth", default=0.9, type=float)
def main(source,
         target,
         reference,
         aim,
         embedding_size,
         epochs,
         batch_k,
         batch_a,
         a1,
         l1,
         lr,
         margin,
         am_folds,
         topk,
         min_threshold,
         max_threshold):

    # print all the click arguments
    print("-----------------------")
    print("Current configuration:")
    print("source: ", source)
    print("target: ", target)
    print("reference: ", reference)
    print("aim: ", aim)
    print("embedding_size: ", embedding_size)
    print("epochs: ", epochs)
    print("batch_k: ", batch_k)
    print("batch_a: ", batch_a)
    print("a1: ", a1)
    print("l1: ", l1)
    print("lr: ", lr)
    print("margin: ", margin)
    print("am_folds: ", am_folds)
    print("topk: ", topk)
    print("min_threshold: ", min_threshold)
    print("max_threshold: ", max_threshold)
    print("-----------------------")
    

    if not os.path.exists(source):
        raise FileNotFoundError(f"Source file {source} not found")
    if not os.path.exists(target):
        raise FileNotFoundError(f"Target file {target} not found")
    if not os.path.exists(reference):
        raise FileNotFoundError(f"Reference file {reference} not found")

    root = "data/"
    source_prefix = source.split("/")[-1].split(".owl")[0]
    target_prefix = target.split("/")[-1].split(".owl")[0]
    root += f"{source_prefix}_{target_prefix}_emb{embedding_size}_e{epochs}_bsk{batch_k}_bsa{batch_a}_a1{a1}_L1{l1}_lr{lr}_m{margin}_AM{am_folds}_k{topk}_th{min_threshold}/"
    # create root dir if not exists
    if not os.path.exists(root):
        os.makedirs(root)

    if aim in ("train", "all"):
        model_file, data_file = train_model(source,
                                            target,
                                            embedding_size,
                                            epochs,
                                            batch_k,
                                            batch_a,
                                            a1,
                                            l1,
                                            lr,
                                            margin,
                                            am_folds,
                                            root=root)

    if aim in ("predict", "all"):
        model_file = os.path.join(root, "modelbin")
        data_file = os.path.join(root, "databin")
        recall, precision, f1 = ranked_predicted_alignments(model_file,
                                    data_file,
                                    reference,
                                    source,
                                    target,
                                    topk,
                                    min_threshold,
                                    max_threshold,
                                    root=root)

        with open("data/hpo_results.txt", "a") as f:
            f.write(f"{source_prefix} {target_prefix} {embedding_size} {epochs} {batch_k} {batch_a} {a1} {l1} {lr} {margin} {am_folds} {topk} {min_threshold} {recall} {precision} {f1}\n")

    else:
        raise ValueError(f"Aim {aim} not recognized")


if __name__ == "__main__":

    print("GPU Available: ", tf.test.is_gpu_available())
    main()
