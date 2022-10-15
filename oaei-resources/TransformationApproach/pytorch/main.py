import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import click as ck
import os
from tqdm import tqdm
import pickle as pkl
from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_curve
import random
import numpy as np
import mowl
mowl.init_jvm("10g")

from mowl.datasets import PathDataset
from mowl.projection import DL2VecProjector
from mowl.projection.edge import Edge

from alion import ALiOn
from data import KGDataset, AlignmentDataset, AllDataset
from onto import lexical_match, remove_inconsistent_alignments

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_data(source, target, embedding_size, batch_size_kg, batch_size_alignment, margin, norm, root):
    # Generate Lexical Alignments
    lexical_alignments = lexical_match(source, target, root)
    
    
    
    source_dataset = PathDataset(source)
    target_dataset = PathDataset(target)

    if os.path.exists(os.path.join(root, "source_graph.pkl")) and os.path.exists(os.path.join(root, "target_graph.pkl")):
        print("Loading graphs from disk")
        with open(os.path.join(root, "source_graph.pkl"), "rb") as f:
            source_kg = pkl.load(f)
        with open(os.path.join(root, "target_graph.pkl"), "rb") as f:
            target_kg = pkl.load(f)
    else:
        print("Generating Knowledge Graphs")
        projector = DL2VecProjector(bidirectional_taxonomy=False)

        source_kg = projector.project(source_dataset.ontology)
        target_kg = projector.project(target_dataset.ontology)

        with open(os.path.join(root, "source_graph.pkl"), "wb") as f:
            pkl.dump(source_kg, f)
        with open(os.path.join(root, "target_graph.pkl"), "wb") as f:
            pkl.dump(target_kg, f)

    # Generate Datasets
    source_classes = source_dataset.classes.as_str
    target_classes = target_dataset.classes.as_str

    extra_rels = ["http://subclassof", "http://superclassof", "http://equivalentto"]
    source_relations = source_dataset.object_properties.as_str + extra_rels
    target_relations = target_dataset.object_properties.as_str + extra_rels
    #source_classes, source_relations = Edge.get_entities_and_relations(source_kg)
    #target_classes, target_relations = Edge.get_entities_and_relations(target_kg)

    source_class_to_id = {c: i for i, c in enumerate(source_classes)}
    target_class_to_id = {c: i for i, c in enumerate(target_classes)}
    source_relation_to_id = {r: i for i, r in enumerate(source_relations)}
    target_relation_to_id = {r: i for i, r in enumerate(target_relations)}

    source_kg = [(source_class_to_id[e.src], source_relation_to_id[e.rel], source_class_to_id[e.dst]) for e in source_kg]
    target_kg = [(target_class_to_id[e.src], target_relation_to_id[e.rel], target_class_to_id[e.dst]) for e in target_kg]

    alignments = [(source_class_to_id[s], target_class_to_id[d]) for s, d in lexical_alignments]

    source_dataset = KGDataset(source_kg)
    target_dataset = KGDataset(target_kg)
    alignment_dataset = AlignmentDataset(alignments)

    all_dataset = AllDataset(source_dataset, target_dataset, alignment_dataset)
    
    # Generate Dataloaders
    source_dataloader = DataLoader(source_dataset, batch_size=batch_size_kg, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size_kg, shuffle=True)
    alignment_dataloader = DataLoader(alignment_dataset, batch_size=batch_size_alignment, shuffle=True)
    all_dataloader = DataLoader(all_dataset, batch_size=batch_size_kg, shuffle=True)
    
    model = ALiOn(len(source_classes),
                  len(target_classes),
                  len(source_relations),
                  len(target_relations),
                  embedding_size,
                  embedding_size,
                  margin,
                  norm)


    return model, source_dataloader, target_dataloader, source_class_to_id, target_class_to_id, len(source_classes), len(target_classes), alignment_dataloader, all_dataloader
    
def train(model,
          source_dataloader,
          target_dataloader,
          num_source_classes,
          num_target_classes,
          alignment_dataloader,
          all_dataloader,
          epochs,
          learning_rate,
          margin,
          device,
          root):

    criterion_kg = nn.MarginRankingLoss(margin=margin)
    criterion_alignment = nn.L1Loss()
    criterion_alignment = nn.MSELoss()
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    best_loss = float("inf")

    model = model.to(device)
    model.train()
    for epoch in tqdm(range(epochs)):
        # Train Source KGyes
        epoch_loss = 0
        if True:
            kg_loss = 0
            alignment_loss = 0
            for i, (source_batch, target_batch, source_al, target_al) in enumerate(all_dataloader):
                # Source batch
                source_batch = source_batch.to(device)
                neg_source_batch = th.randint(0, num_source_classes, (source_batch.shape[0], 1), device=device)
                neg_source_batch = th.cat((source_batch[:, :-1], neg_source_batch), dim=1)

                pos_source_distances, neg_source_distances = model(source_batch, neg_source_batch, "source")
                target = -th.ones_like(pos_source_distances)
                source_loss = criterion_kg(pos_source_distances, neg_source_distances, target)

                # Target batch
                target_batch = target_batch.to(device)
                neg_target_batch = th.randint(0, num_target_classes, (target_batch.shape[0], 1), device=device)
                neg_target_batch = th.cat((target_batch[:, :-1], neg_target_batch), dim=1)
                pos_target_distances, neg_target_distances = model(target_batch, neg_target_batch, "target")
                target = -th.ones_like(pos_target_distances)
                target_loss = criterion_kg(pos_target_distances, neg_target_distances, target)
                kg_loss += source_loss + target_loss

                # Alignment batch
                source_al = source_al.to(device)
                target_al = target_al.to(device)
                preds, targets = model(source_al, target_al, "alignment")
                loss = criterion_alignment(preds, targets)
                alignment_loss += 10*loss
                
                epoch_loss = kg_loss + alignment_loss

            kg_loss /= (i + 1)
            alignment_loss /= (i + 1)
            epoch_loss /= (i + 1)
            optimizer.zero_grad()
            epoch_loss.backward()
            optimizer.step() 
                        
                
        else:
            source_loss = 0        
            for batch in source_dataloader:
                # Generate negatives
                negative_samples = th.randint(0, num_source_classes, (batch.shape[0], 1))
                negative_samples = th.cat((batch[:, :-1], negative_samples), dim=1)
                
                batch = batch.to(device)
                negative_samples = negative_samples.to(device)
                positive_distances, negative_distances = model(batch, negative_samples, 'source')
                target = -th.ones_like(positive_distances)
                loss = criterion_kg(positive_distances, negative_distances, target)
                source_loss += loss
            
            # Train Target KG
            target_loss = 0
            for batch in target_dataloader:
                negative_samples = th.randint(0, num_target_classes, (batch.shape[0], 1))
                negative_samples = th.cat((batch[:, :-1], negative_samples), dim=1)
                
                batch = batch.to(device)
                negative_samples = negative_samples.to(device)
                positive_distances, negative_distances = model(batch, negative_samples, 'target')
                target = -th.ones_like(positive_distances)
                loss = criterion_kg(positive_distances, negative_distances, target)
                target_loss += loss

        

            # Train Alignment
            alignment_loss = 0
            for source, target in alignment_dataloader:
                source = source.to(device)
                target = target.to(device)
                preds, targets = model(source, target, "alignment")
                alignment_loss += criterion_alignment(preds, targets)
                

            total_triples = len(source_dataloader.dataset) + len(target_dataloader.dataset) + len(alignment_dataloader.dataset)

            num_source_triples = len(source_dataloader.dataset)
            num_target_triples = len(target_dataloader.dataset)
            num_alignment_triples = len(alignment_dataloader.dataset)

            max_num_triples = max(num_source_triples, num_target_triples, num_alignment_triples)

            weight_source = max_num_triples / num_source_triples
            weight_target = max_num_triples / num_target_triples
            weight_alignment = 10* max_num_triples / num_alignment_triples

            kg_loss = weight_source * source_loss + weight_target * target_loss
            alignment_loss = weight_alignment * alignment_loss
            loss = kg_loss + alignment_loss

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}\t KG Loss: {kg_loss}\t Alignment Loss: {alignment_loss}")

        if epoch_loss.detach().item() < best_loss:
            best_loss = epoch_loss.detach().item()
            th.save(model.state_dict(), f"{root}/model.pt")

    print(f"Traingin Complete. Best Loss: {best_loss}")


def test(model,source_owl, target_owl, source_class_to_id, target_class_to_id, reference, device, root):

    source_id_to_class = {v: k for k, v in source_class_to_id.items()}
    target_id_to_class = {v: k for k, v in target_class_to_id.items()}
    predictions = dict()
    criterion = nn.L1Loss(reduction="none")
    criterion = nn.MSELoss(reduction="none")

    
    model.load_state_dict(th.load(f"{root}/model.pt"))
    model = model.to(device)
    
    model.eval()
    with th.no_grad():
        for src in tqdm(source_id_to_class):
            src_th = th.tensor([src], dtype=th.long, device=device)
            trgts = th.tensor(list(target_id_to_class.keys()), dtype=th.long, device=device)

            preds, targets = model(src_th, trgts, "alignment")
            scores = 1- th.sigmoid(th.sum(criterion(preds, targets), dim=1))
            
            scores = scores.cpu().numpy()

            # Get top k
            k = 10
            top_k = np.argsort(scores)[-k:]
            #for trg in top_k:
            #    predictions[(source_id_to_class[src], target_id_to_class[trg])] = (0, scores[trg])

            for target, score in zip(list(target_class_to_id), scores):
                predictions[(source_id_to_class[src], target)] = (0, score)


    # Normalize scores
    #unnormalized_scores = [score for _, score in predictions.values()]
    #normalized_scores = (unnormalized_scores - np.min(unnormalized_scores)) / (np.max(unnormalized_scores) - np.min(unnormalized_scores))
    #for i, (_, score) in enumerate(predictions.values()):
    #    predictions[tuple(predictions.keys())[i]] = (0, normalized_scores[i])

                
    non_equivalence_in_reference = 0
    with open(reference, "r") as f:
        reference = f.readlines()
        reference = [r.strip().split("\t") for r in reference]
    
        for r in reference:
            if (r[0], r[1]) in predictions:
                if r[3] == "=":
                    score = predictions[(r[0], r[1])][1]
                    predictions[(r[0], r[1])] = (1, score)
                else:
                    non_equivalence_in_reference += 1
                    score = predictions[(r[0], r[1])][1]
                    predictions[(r[0], r[1])] = (1, 0)
            else:
                if r[0].startswith("http"):
                    print("Missing:", r[0], r[1])
                    predictions[(r[0], r[1])] = (1, 0)
                else:
                    print(f"Bad format: {r[0]} - {r[1]}")

    print(f"Non-equivalence in reference: {non_equivalence_in_reference}")
    
    trues = [v[0] for v in predictions.values()]
    scores = [v[1] for v in predictions.values()]

    fpr, tpr, thresholds = roc_curve(trues, scores)
    auc_score = auc(fpr, tpr)
    print(f"AUC: {auc_score}")

    # Get the best threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    positive_preds = dict()
    for i, (pred_alignment) in enumerate(predictions.keys()):
        true_score = predictions[pred_alignment][0]
        pred_score = predictions[pred_alignment][1]
        if pred_score > thresholds[ix]:
            positive_preds[pred_alignment] = (true_score, 1)
        elif true_score == 1:
            positive_preds[pred_alignment] = (true_score, 0)
            

    inconsistent_alignments = remove_inconsistent_alignments(source_owl, target_owl, list(positive_preds.keys()))

    print(f"Number of inconsistent alignments: {len(inconsistent_alignments)}")

    positive_preds = {k: v for k, v in positive_preds.items() if k not in inconsistent_alignments}
            
    modified_trues = [p[0] for p in positive_preds.values()]
    modified_preds = [p[1] for p in positive_preds.values()]

    
    print(sum(modified_trues), sum(modified_preds))
    recall = recall_score(modified_trues, modified_preds)
    precision = precision_score(modified_trues, modified_preds)
    f1 = f1_score(modified_trues, modified_preds)
        
    return recall, precision, f1




@ck.command()
@ck.option('--source', '-s', type=ck.Path(exists=True, dir_okay=False))
@ck.option('--target', '-t', type=ck.Path(exists=True, dir_okay=False))
@ck.option('--reference', '-r', type=ck.Path(exists=False, dir_okay=False))
@ck.option('--embedding-size', '-e', type=int, default=100)
@ck.option('--batch-size-kg', '-bkg', type=int, default=32)
@ck.option('--batch-size-alignment', '-bal', type=int, default=32)
@ck.option('--epochs', '-ep', type=int, default=400)
@ck.option('--learning-rate', '-lr', type=float, default=0.001)
@ck.option('--margin', '-m', type=float, default=1.0)
@ck.option('--norm', '-n', type=int, default=1)
@ck.option('--device', '-d', type=str, default='cpu')
@ck.option('--aim', '-aim', type=ck.Choice(['all', "train", "test"]), default='all')
@ck.option('--seed', '-seed', type=int, default=42)
@ck.option('--root', "-root", type=ck.Path(exists=True, dir_okay=True))
def main(source, target, reference, embedding_size, batch_size_kg, batch_size_alignment, epochs, learning_rate, margin, norm, device, aim, seed, root):


    #Print the parameters
    print("------------------------------------")
    print("Configuration:")
    print("Source: ", source)
    print("Target: ", target)
    print("Reference: ", reference)
    print("Embedding size: ", embedding_size)
    print("Batch size KG: ", batch_size_kg)
    print("Batch size alignment: ", batch_size_alignment)
    print("Epochs: ", epochs)
    print("Learning rate: ", learning_rate)
    print("Margin: ", margin)
    print("Norm: ", norm)
    print("Device: ", device)
    print("Aim: ", aim)
    print("Seed", seed)
    print("------------------------------------")

    seed_everything(seed)
    
    if not root.endswith("/"):
        root = root + "/"
    
    source_prefix = source.split("/")[-1].split(".owl")[0]
    target_prefix = target.split("/")[-1].split(".owl")[0]
    post_root = f"{source_prefix}_{target_prefix}_emb{embedding_size}_bskg{batch_size_kg}_bsal{batch_size_alignment}_ep{epochs}_lr{learning_rate}_m{margin}"

    root = root + post_root + "/"
    if not os.path.exists(root):
        os.makedirs(root)

    print("Loading data...")
    model, source_dataloader, target_dataloader, source_class_to_id, target_class_to_id, num_source_classes, num_target_classes, alignment_dataloader, all_dataloader = load_data(source, target, embedding_size, batch_size_kg, batch_size_alignment, margin, norm, root)
    
    if aim in ['all', 'train']:

        #Train the model
        print("Training the model...")
        train(model, source_dataloader, target_dataloader, num_source_classes, num_target_classes,  alignment_dataloader, all_dataloader, epochs, learning_rate, margin, device, root)

        
    if aim in ["all", "test"]:
        #Test the model
        print("Testing the model...")
        recall, precision, f1 = test(model, source, target, source_class_to_id, target_class_to_id, reference, device, root)

        print(f"Recall: {recall}\t Precision: {precision}\t F1: {f1}")
        
if __name__ == '__main__':
    main()
