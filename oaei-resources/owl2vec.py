import mowl
mowl.init_jvm("5g")
from mowl.datasets.base import PathDataset
from mowl.projection.owl2vec_star.model import OWL2VecStarProjector
from mowl.projection.edge import Edge
from mowl.walking.deepwalk.model import DeepWalk
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from evaluate import evaluate
import pickle as pkl
from tqdm import tqdm

path_id_entity_src = "id_entity_src.pkl"
path_id_entity_tar = "id_entity_tar.pkl"
path_train_triples = "train_triples.pkl"
path_model = "path_model"

def getID():

    with open(path_id_entity_src, "rb") as f:
        id_entity_src = pkl.load(f)
    with open(path_id_entity_tar, "rb") as f:
        id_entity_tar = pkl.load(f)

    model = Word2Vec.load(path_model)
    vector = model.wv

    source_vecs = {x:vector[x] for _, x in id_entity_src.items()}
    target_vecs = {x:vector[x] for _, x in id_entity_tar.items()}
        
    return id_entity_src, id_entity_tar, source_vecs, target_vecs


def getList():
    with open(path_id_entity_src, "rb") as f:
        id_entity_src = pkl.load(f)
    with open(path_id_entity_tar, "rb") as f:
        id_entity_tar = pkl.load(f)

    src_list = list(set(id_entity_src.values()))
    tar_list = list(set(id_entity_tar.values()))
    return src_list, tar_list

def getAligns(soure_list, target_list, threshold):
    print("####running getAligns---------------------------------")
    sim_dict = {}
    vec_alignments = []
    ent_ids_source,ent_ids_target,source_vecs,target_vecs = getID()
    for ent1 in tqdm(soure_list):
        source_vec = source_vecs[ent1]
        for ent2 in target_list:
            target_vec = target_vecs[ent2]
            simility = np.dot(target_vec, source_vec) / (np.linalg.norm(target_vec) * np.linalg.norm(source_vec))
            sim_dict[ent1 + '\t' + ent2] = simility
            if simility >= threshold: 
                vec_alignments.append((ent1, ent2))
    return sim_dict, vec_alignments


def get_threshold(Y, X):
    fpr, tpr, thresholds = roc_curve(Y, X, pos_label=1)
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    roc_auc = auc(fpr, tpr)
    best_threshold = thresholds[ix]
    return best_threshold

def compute_embeddings(source_path, target_path):


    source_ds = PathDataset(source_path, None,None)
    target_ds = PathDataset(target_path, None, None)

    source_only_cls = source_ds.classes
    target_only_cls = target_ds.classes


    
    projector = OWL2VecStarProjector(bidirectional_taxonomy = True, include_literals = True)
    source_graph = projector.project(source_ds.ontology)
    target_graph = projector.project(target_ds.ontology)


    source_graph_ents, _ = Edge.getEntitiesAndRelations(source_graph)
    target_graph_ents, _ = Edge.getEntitiesAndRelations(target_graph)

    source_graph_ents = [x for x in source_graph_ents if x in source_only_cls]
    target_graph_ents = [x for x in target_graph_ents if x in target_only_cls]
    id_entity_src = {i: name for i,name in enumerate(source_graph_ents)}
    id_entity_tar = {i: name for i,name in enumerate(target_graph_ents)}

    with open(path_id_entity_src, "wb") as f:
        pkl.dump(id_entity_src, f)

    with open(path_id_entity_tar, "wb") as f:
        pkl.dump(id_entity_tar, f)



    
    source_target_graph = source_graph + target_graph    
    walker = DeepWalk(
        40,
        30,
        0.1,
        workers = 14)
    walker.walk(source_target_graph)

    corpus = LineSentence(walker.outfile)

    w2v_model = Word2Vec(
        corpus,
        sg=1,
        min_count=1,
        vector_size=50,
        window = 7,
        epochs = 40,
        workers = 14)

    w2v_model.save(path_model)
    
if __name__ == "__main__":
    source_owl = "../data/mouse.owl"
    target_owl = "../data/human.owl"

    compute_embeddings(source_owl, target_owl)

    w2v_model = Word2Vec.load(path_model)
    vectors = w2v_model.wv
    
    source_ds = PathDataset(source_owl, None,None)
    target_ds = PathDataset(target_owl, None, None)

    source_only_cls = source_ds.classes
    target_only_cls = target_ds.classes

    source_class_with_embs = [c for c in source_only_cls if c in vectors]
    target_class_with_embs = [c for c in target_only_cls if c in vectors]

    source_emb = []
    for src_cls in source_class_with_embs:
        source_emb.append(vectors[src_cls])

    target_emb = []
    for targ_cls in target_class_with_embs:
        target_emb.append(vectors[targ_cls])
    
    source_emb_np = np.array(source_emb)
    target_emb_np = np.array(target_emb)
    scores = cosine_similarity(source_emb_np, target_emb_np)

    source_cls_to_id = {v:k for k,v in enumerate(source_class_with_embs)}
    target_cls_to_id = {v:k for k,v in enumerate(target_class_with_embs)}

    with open("owl2vec_scores.tsv", "w") as f:
        f.write("First_Ontology_Class\tSecond_Ontology_Class\tScore\tRelation\n")
        for src_cls in source_class_with_embs:
            for tar_cls in target_class_with_embs:
                src_idx = source_cls_to_id[src_cls]
                tar_idx = target_cls_to_id[tar_cls]
                f.write(f"{src_cls}\t{tar_cls}\t{scores[src_idx, tar_idx]}\t=\n")

            
    avg_prec, auc = evaluate("owl2vec_scores.tsv")
    print(avg_prec, auc)
