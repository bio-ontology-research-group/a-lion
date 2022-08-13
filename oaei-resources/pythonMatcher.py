import logging
import sys
from AlignmentFormat import serialize_mapping_to_tmp_file
import amd_baseline as amd
from amd_baseline import getTrainList
from owl2vec import compute_embeddings, getID, getList, getAligns
import alion
import pickle as pkl
from tqdm import tqdm
import urllib.request

def match_rdflib(source_graph, target_graph, input_alignment):
    # a very simple label matcher:
    alignment = []

    label_to_uri = defaultdict(list)
    for s, p, o in source_graph.triples((None, RDFS.label, None)):
        if isinstance(s, URIRef):
            label_to_uri[str(o)].append(str(s))

    for s, p, o in target_graph.triples((None, RDFS.label, None)):
        if isinstance(s, URIRef) and str(o) in label_to_uri:
            for one_uri in label_to_uri[str(o)]:
                alignment.append((one_uri, str(s), "=", 1.0))
    return alignment
    # return [('http://one.de', 'http://two.de', '=', 1.0)]


def get_file_from_url(location):
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname, urlopen

    if location.startswith("file:"):
        return open(url2pathname(unquote(urlparse(location).path)))
    else:
        return urlopen(location)




def alignmentMatch(source_url, target_url):
    print("####running alignmentMatch---------------------------------")
    relation = '='
    alignments = []
    train = []
    

    aligns = []
    ent12ent2 = {}
    compute_embeddings(source_url, target_url)
    threshold = 0.4
    ent_ids_source,ent_ids_target,source_vecs,target_vecs = getID()
    source_list,target_list = getList()
    sim_dict, vec_alignments = getAligns(source_list, target_list, threshold)
    train = getTrainList(source_url, target_url)
    

    path_id_entity_src = "id_entity_src.pkl" # In AMD example the file was ent_ids_source.txt
    path_id_entity_tar = "id_entity_tar.pkl" # In AMD example the file was ent_ids_target.txt

    with open(path_id_entity_src, "rb") as f:
        sourceent2id = pkl.load(f)
    with open(path_id_entity_tar, "rb") as f:
        targetent2id = pkl.load(f)
        
    #with open('ent_ids_source.txt','r') as f:
    #    sourceent2id = {line.strip().split('\t')[0]: line.strip().split('\t')[1] for line in f.readlines()}
    #with open('ent_ids_target.txt','r') as f:
    #    targetent2id = {line.strip().split('\t')[0]: line.strip().split('\t')[1] for line in f.readlines()}
    
    # with open('train.txt','r') as f:
    #     for line in f.readlines():
    #         pair = line.strip().split('\t')
    #         train.append((int(pair[0]), int(pair[1])))
    #     f.close()  
    
    for align in tqdm(vec_alignments):
        source, target = align[0], align[1]
        if source is None or target is None:
            print("Found None is aligns")
            continue
        simility = sim_dict[source + '\t' + target]
        aligns.append((source,target))
        alignments.append((source,target,relation,round(simility)))

    for i in range(len(train)):
        a = sourceent2id.get(str(train[i][0]))
        b = targetent2id.get(str(train[i][1]))
        if a is None or b is None:
            print("Found None in train list")
            continue
        if (a,b) not in aligns:
            alignments.append((a,b,relation,1.0))  
    #print(alignments)

    return alignments

    
def match(source_url, target_url, input_alignment_url):
    logging.info("Python matcher info: Match " + source_url + " to " + target_url)



    urllib.request.urlretrieve(source_url, "source.owl")
    urllib.request.urlretrieve(target_url, "target.owl")

    source_file = "source.owl"
    target_file = "target.owl"

    
    resulting_alignment = alignmentMatch(source_file, target_file)
    with open("final_align.pkl","wb") as f:
        pkl.dump(resulting_alignment, f)
    # in case you have the results in a pandas dataframe, make sure you have the columns
    # source (uri), target (uri), relation (usually the string '='), confidence (floating point number)
    # in case relation or confidence is missing use: df["relation"] = '='  and  df["confidence"] = 1.0
    # then select the columns in the right order (like df[['source', 'target', 'relation', 'confidence']])
    # because serialize_mapping_to_tmp_file expects an iterbale of source, target, relation, confidence
    # and then call .itertuples(index=False)
    # example: alignment_file_url = serialize_mapping_to_tmp_file(df[['source', 'target', 'relation', 'confidence']].itertuples(index=False))

    alignment_file_url = serialize_mapping_to_tmp_file(resulting_alignment)
    return alignment_file_url



def main(argv):
    if len(argv) == 2:
        print(match(argv[0], argv[1], None))
    elif len(argv) >= 3:
        if len(argv) > 3:
            logging.error("Too many parameters but we will ignore them.")
        print(match(argv[0], argv[1], argv[2]))
    else:
        logging.error(
            "Too few parameters. Need at least two (source and target URL of ontologies"
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO
    )
    main(sys.argv[1:])

#print(match())
