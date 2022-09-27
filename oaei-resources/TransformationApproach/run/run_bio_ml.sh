#  OMIM-DOID
#time python training_modelR.py -s data/MONDO/equiv_match/ontos/omim.owl -t data/MONDO/equiv_match/ontos/ordo.owl -r data/MONDO/equiv_match/refs/omim2ordo/unsupervised/test.tsv -m unsupervised

time python training_modelR.py -s data/MONDO/equiv_match/ontos/omim.owl -t data/MONDO/equiv_match/ontos/ordo.owl -r data/MONDO/equiv_match/refs/omim2ordo/semi_supervised/test.tsv -m semisupervised -tr data/MONDO/equiv_match/refs/omim2ordo/semi_supervised/train.tsv
#NCIT-DOID
#SNOMED-FMA (Body)
#SNOMED-NCIT (Pharm)
#SNOMED-NCIT (Neoplas)
