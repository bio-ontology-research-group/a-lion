import mowl
mowl.init_jvm("5g")
from mowl.datasets.base import PathDataset
from mowl.projection.owl2vec_star.model import OWL2VecStarProjector
from mowl.walking.deepwalk.model import DeepWalk
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from evaluate import evaluate
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
import tempfile

def objective(config):
    human_ds = PathDataset("../data/human.owl",None,None)
    mouse_ds = PathDataset("../data/mouse.owl", None, None)

    human_only_cls = human_ds.classes
    mouse_only_cls = mouse_ds.classes

    projector = OWL2VecStarProjector(bidirectional_taxonomy = True, include_literals = True)
    mouse_graph = projector.project(mouse_ds.ontology)
    human_graph = projector.project(human_ds.ontology)

    mouse_human_graph = mouse_graph + human_graph
    walker = DeepWalk(config["num_walks"], config["walk_length"], config["alpha"], workers = 16)
    walker.walk(mouse_human_graph)

    corpus = LineSentence(walker.outfile)

    w2v_model = Word2Vec(
        corpus,
        sg=1,
        min_count=1,
        vector_size=config["wv_vector_size"],
        window = config["window"],
        epochs = config["epochs"],
        workers = 16)

    vectors = w2v_model.wv

    mouse_class_with_embs = [c for c in mouse_only_cls if c in vectors]
    human_class_with_embs = [c for c in human_only_cls if c in vectors]
    mouse_emb = []
    for mcls in mouse_class_with_embs:
        mouse_emb.append(vectors[mcls])
    human_emb = []
    for hcls in human_class_with_embs:
        human_emb.append(vectors[hcls])

    mouse_emb_np = np.array(mouse_emb)
    human_emb_np = np.array(human_emb)

    scores = cosine_similarity(mouse_emb_np, human_emb_np)

    human_cls_to_id = {v:k for k,v in enumerate(human_class_with_embs)}
    mouse_cls_to_id = {v:k for k,v in enumerate(mouse_class_with_embs)}

    tmpfile = tempfile.getTemporaryFile()
    tmpfile = tmpfile.getName()
    
    with open(tmpfile, "w") as f:
    f.write("First_Ontology_Class\tSecond_Ontology_Class\tScore\tRelation\n")
    for mcls in mouse_class_with_embs:
        for hcls in human_class_with_embs:
            midx = mouse_cls_to_id[mcls]
            hidx = human_cls_to_id[hcls]
            f.write(f"{mcls}\t{hcls}\t{scores[midx, hidx]}\t=\n")


    avg_prec, auc = evaluate(tmpfile)
    tune.report(auc=auc)
    print(avg_prec, auc)



search_space = {
    "num_walks": tune.choice([10, 50, 100, 200]),
    "walk_length": tune.choice([10, 20, 50, 100]),
    "alpha": tune.choice([0.0, 0.1, 0.2]),
    "wv_vector_size": tune.choice([25, 50, 100, 150]),
    "window": tune.choice([5,7,9]),
    "epochs": tune.choice([10, 20, 40, 60])
}

algo = OptunaSearch()

analysis = tune.run(
    objective,
    metric="auc",
    mode="max",
    search_alg=algo,
    stop={"training_iteration": 1},
    config=search_space,
)

