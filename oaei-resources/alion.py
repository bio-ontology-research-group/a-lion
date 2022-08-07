def alionMatch(source_url, target_url):
    print("####running alignmentMatch---------------------------------")
    relation = '='
    alignments = []
    train = []
    aligns = []
    ent12ent2 = {}
    #    getFile(source_url, target_url)
    #    embedding_run()

    
    threshold = 0.95  
    ent_ids_source,ent_ids_target,source_vecs,target_vecs = getID()
    source_list,target_list = getList()
    sim_dict, vec_alignments = getAligns(source_list, target_list, threshold)

    with open('ent_ids_source.txt','r') as f:
        sourceent2id = {line.strip().split('\t')[0]: line.strip().split('\t')[1] for line in f.readlines()}
    with open('ent_ids_target.txt','r') as f:
        targetent2id = {line.strip().split('\t')[0]: line.strip().split('\t')[1] for line in f.readlines()}
    
    with open('train.txt','r') as f:
        for line in f.readlines():
            pair = line.strip().split('\t')
            train.append((int(pair[0]), int(pair[1])))
        f.close()  
    
    for align in tqdm(vec_alignments):
        source, target = align[0], align[1]
        simility = sim_dict[source + '\t' + target]
        aligns.append((source,target))
        alignments.append((source,target,relation,round(simility)))

    for i in range(len(train)):
        a = sourceent2id.get(str(train[i][0]))
        b = targetent2id.get(str(train[i][1]))
        if (a,b) not in aligns:
            alignments.append((a,b,relation,1.0))  
    #print(alignments)

    return alignments
