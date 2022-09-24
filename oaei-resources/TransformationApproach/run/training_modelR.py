from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import tempfile
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score
import click as ck

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# we don't import individual things in a model.
# This is to make auto reloading in Notebook happy
from multiG import multiG
from trainerR import Trainer
from OntoL import Onto2KG, LexicalMatch, removeInconsistincyAlignmnets

from testerR import Tester
# print(tf.version)


def ranked_predicted_alignments(model_file, data_file,
                                reference_alignment_file,
                                source,
                                target,
                                files_with_header=False,
                                root=None):

    pred_alignments = generate_alignments(model_file, data_file,
                                          source, target, root=root)

    mappings_index = {}

    for al in pred_alignments:
        tuple_ = (al[0], al[1])

        mappings_index[tuple_] = [0, 1]

    with open(reference_alignment_file) as f:
        if files_with_header:
            content = f.readlines()[1:]
        else:
            content = f.readlines()

    for p in content:
        line = p.split()
        tuple_ = (line[0], line[1])
        if (tuple_ not in mappings_index):
            mappings_index[tuple_] = [1, 0]

        else:
            mappings_index[tuple_][0] = 1

    X = []
    Y = []
    for k in mappings_index:
        X.append(mappings_index[k][1])
        Y.append(mappings_index[k][0])

    print("recal", recall_score(X, Y))
    print("precision", precision_score(X, Y))
    print("f1_score", f1_score(X, Y))


params = {
    "embedding_size": 128,
    "epochs": 21,  # 401,
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


def generate_alignments(model_file, data_file, source, target, root=None):
    if root is None or not isinstance(root, str):
        raise TypeError("Parameter outfile_path must be of type stra")

    topk = params["topk"]
    threshold = params["threshold"]
    tester = Tester()
    tester.build(save_path=model_file, data_save_path=data_file)

    source_entities = list(tester.multiG.KG1.ents.keys())
    target_entities = list(tester.multiG.KG2.ents.keys())

    min_entities = min(len(source_entities), len(target_entities))//2
    target_entities_vectors = tester.vec_e[2]

    alignments = []

    acceptable_alignments = False

    while not acceptable_alignments:
        for class_ in tqdm(source_entities, total=len(source_entities)):
            class_url = tester.multiG.KG1.ent_index2str(class_)
            vec_proj_class = tester.projection(class_, source=1)
            rst = tester.kNN_with_names(vec_proj_class,
                                        target_entities_vectors, topk)
            for i in range(topk):
                if (rst[i][1] < threshold):
                    if class_ in source_entities:
                        source_entities.remove(class_)

                    alignments.append([class_url, rst[i][0], "=", 1.0])

        if (len(alignments) >= min_entities) or threshold > 1.0:
            acceptable_alignments = True
        else:
            print(f"Not enough alignments, trying higher threshold. \
Num of aligns: {len(alignments)}. Min entities: {min_entities}")

            threshold += 0.1
            print("")

    tester = Tester()
    tester.build(save_path=model_file, data_save_path=data_file)
    predictions = tester.predicted_alignments(5, 0.1)

    ls = removeInconsistincyAlignmnets(source, target, predictions)
    print(ls)
    print("-------------------")
    print(f"Predictions found: {len(predictions)}")
    print("-------------------")
    print(f"Inconsistencies found: {len(ls)}")

    ls = set([(x[0], x[1], "=", 1.0) for x in ls])
    alignments = [tuple(x) for x in alignments]
    print(f"Number of original alignments: {len(alignments)}")
    alignments = list(set(alignments) - ls)
    print(f"Number of alignments after removing inconsistencies: \
{len(alignments)}")

    outfile_path = os.path.join(root, "predicted_alignments.txt")
    with open(outfile_path, "w") as f:
        f.write("SrcEntity\tTgtEntity\tScore\n")
        for src, dst, _, score in alignments:
            f.write(f"{src}\t{dst}\t{score}\n")

    return alignments


def train_model(source_owl, target_owl, root=None):

    # Params accessing ###########
    this_dim = params["embedding_size"]
    # model_path = params["model_path"]
    # data_path = params["data_path"]

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

    # Knowledge Graph Building #####
    KG1 = Onto2KG(source_owl)
    KG2 = Onto2KG(target_owl)
    ####################################

    this_data = multiG(KG1, KG2)

    if root is None:
        lexical_alignments_file = tempfile.NamedTemporaryFile(delete=False)
        lexical_alignments_file_name = lexical_alignments_file.name
    else:
        lexical_alignments_file_name = os.path.join(root,
                                                    "lexical_alignments.txt")
    # lexical_alignments_file_name = "data/lexical_alignments"
    # if not os.path.exists(lexical_alignments_file_name+ ".json"):
    print("Computing lexical alignments")
    alignments, min_alignments = LexicalMatch(source_owl,
                                              target_owl,
                                              lexical_alignments_file_name)

    AM_folds = min_alignments  # int(min_alignments//20)+1
    print("AM Folds: ", AM_folds)

    # print("Loading lexical alignments from file")
    # this_data.load_align_json(lexical_alignments_file_name+ ".json")
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
@ck.option('--mode', '-m', type=ck.Choice(["semisupervised", "unsupervised"]),
           help='Mode')
def main(source, target, reference, mode):

    if not os.path.exists(source):
        raise FileNotFoundError(f"Source file {source} not found")
    if not os.path.exists(target):
        raise FileNotFoundError(f"Target file {target} not found")
    if not os.path.exists(reference):
        raise FileNotFoundError(f"Reference file {reference} not found")

    root = "data/"
    source_prefix = source.split("/")[-1].split(".")[0]
    target_prefix = target.split("/")[-1].split(".")[0]
    root += f"{source_prefix}_{target_prefix}_{mode}/"
    # create root dir if not exists
    if not os.path.exists(root):
        os.makedirs(root)

    model_file, data_file = train_model(source, target, root=root)
    ranked_predicted_alignments(model_file,
                                data_file,
                                reference,
                                source,
                                target,
                                files_with_header=True,
                                root=root)


if __name__ == "__main__":
    print("GPU Available: ", tf.test.is_gpu_available())
    main()
