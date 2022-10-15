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


class AllDataset(Dataset):
    def __init__(self, source_triples, target_triples, alignments):
        self.source_triples = source_triples
        self.target_triples = target_triples
        self.alignments = alignments

        self.max_len = max(len(source_triples), len(target_triples), len(alignments))


    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        max_len = self.max_len
        if max_len == len(self.source_triples):
            source = self.source_triples[idx]
            target = self.target_triples[idx % len(self.target_triples)]
            alignment = self.alignments[idx % len(self.alignments)]
        elif max_len == len(self.target_triples):
            source = self.source_triples[idx % len(self.source_triples)]
            target = self.target_triples[idx]
            alignment = self.alignments[idx % len(self.alignments)]
        else:
            source = self.source_triples[idx % len(self.source_triples)]
            target = self.target_triples[idx % len(self.target_triples)]
            alignment = self.alignments[idx]

        return np.array(source), np.array(target), alignment[0], alignment[1]
