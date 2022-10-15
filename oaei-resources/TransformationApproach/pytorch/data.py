from torch.utils.data import Dataset
import numpy as np

class KGDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return np.array(self.triples[idx])


class AlignmentDataset(Dataset):
    def __init__(self, alignments):
        self.alignments = alignments

    def __len__(self):
        return len(self.alignments)

    def __getitem__(self, idx):
        return self.alignments[idx]
