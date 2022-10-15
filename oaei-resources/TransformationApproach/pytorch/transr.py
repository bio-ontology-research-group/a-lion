import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize


class TransR(nn.Module):

    def __init__(self, entity_count, relation_count, norm=1, dim_e=100, dim_r = 100, margin=1.0, device = 'cpu'):
        super().__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.norm = norm
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.entities_emb = self._init_enitity_emb()
        self.relations_emb, self.relation_projection_emb = self._init_relation_emb()
        self.criterion = nn.MarginRankingLoss(margin=
                                              margin, reduction='none')

        self.device = device
        
    def _init_enitity_emb(self):
        entities_emb = nn.Embedding(self.entity_count, self.dim_e)
        nn.init.xavier_uniform_(entities_emb.weight.data)
        return entities_emb

    def _init_relation_emb(self):
        relations_emb = nn.Embedding(self.relation_count, self.dim_r)
        nn.init.xavier_uniform_(relations_emb.weight.data)

        relation_projection_emb = nn.Embedding(self.relation_count, self.dim_e * self.dim_r)
        nn.init.xavier_uniform_(relation_projection_emb.weight.data)
        return relations_emb, relation_projection_emb

    def forward(self,
                positive_triples: torch.LongTensor,
                negative_triples: torch.LongTensor):
                                                
        """Return model losses based on the input.
        :param positive_triples: triples of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :param negative_triples: triples of negatives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: tuple of the model loss, positive triples loss component, negative triples loss component
        """

        # Normalize embeddings
        self.entities_emb.weight.data = normalize(self.entities_emb.weight.data, p=self.norm, dim=1)
        self.relations_emb.weight.data = normalize(self.relations_emb.weight.data, p=self.norm, dim=1)

        assert positive_triples.shape[1] == 3
        assert negative_triples.shape[1] == 3

        positive_distances = self._distance(positive_triples)
        negative_distances = self._distance(negative_triples)

        return positive_distances, negative_distances

    def predict(self, triples: torch.LongTensor):
        """Calculated dissimilarity score for given triples.
        :param triples: triples in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triples
        """
        return self._distance(triples)


    def _distance(self, triples, project = False):
        """Triples should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        assert triples.size()[1] == 3

        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]


        heads = self._project(heads, relations)
        tails = self._project(tails, relations)
        relations = self.relations_emb(relations)

        # Normalize projected entities
        heads = normalize(heads, p=self.norm, dim=1)
        tails = normalize(tails, p=self.norm, dim=1)
        
        distances = (heads + relations - tails).norm(p=self.norm, dim=1)
                   
        return distances

    def _project(self, entities, relation_ids):
        """Project entities to the relation space."""
        projection = self.relation_projection_emb(relation_ids)
        projection = projection.view(-1, self.dim_e, self.dim_r)
        entities = self.entities_emb(entities)
        return torch.matmul(entities, projection)

