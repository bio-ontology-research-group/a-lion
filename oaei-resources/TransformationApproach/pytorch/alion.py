import torch as th
import torch.nn as nn
from transr import TransR
from torch.nn.functional import normalize

class ALiOn(nn.Module):
    def __init__(self,
                 num_entities_source,
                 num_entities_target,
                 num_relations_source,
                 num_relations_target,
                 dim_e,
                 dim_r,
                 margin,
                 norm):

        super(ALiOn, self).__init__()
        
        self.num_entities_source = num_entities_source
        self.num_entities_target = num_entities_target
        self.num_relations_source = num_relations_source
        self.num_relations_target = num_relations_target
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.margin = margin
        self.norm = norm
        self.source_model = TransR(num_entities_source,
                                   num_relations_source,
                                   norm=self.norm,
                                   dim_e=self.dim_e,
                                   dim_r=self.dim_r,
                                   margin=self.margin)

        self.target_model = TransR(num_entities_target,
                                   num_relations_target,
                                   norm=self.norm,
                                   dim_e=self.dim_e,
                                   dim_r=self.dim_r,
                                   margin=self.margin)

        self.fc = nn.Linear(self.dim_e, self.dim_e)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, data_a, data_b, mode):

        
        self.source_model.entities_emb.weight.data = normalize(self.source_model.entities_emb.weight.data, p=self.norm, dim=1)
        self.target_model.entities_emb.weight.data = normalize(self.target_model.entities_emb.weight.data, p=self.norm, dim=1)

        
        if mode == 'source':
            return self.source_model(data_a, data_b)
        elif mode == 'target':
            return self.target_model(data_a, data_b)
        elif mode == 'alignment':
            source_ents = self.source_model.entities_emb(data_a)
            target_ents = self.target_model.entities_emb(data_b)

            target_preds = self.fc(target_ents)  # normalize(self.fc(target_ents), p=self.norm, dim=1)
            return target_preds, target_ents
        else:
            raise ValueError(f'Invalid mode: {mode}')
            
