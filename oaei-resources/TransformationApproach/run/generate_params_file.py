def generate_params_file():
    
    source_owl="data/Testing_anatomy/mouse.owl"
    target_owl="data/Testing_anatomy/human.owl"
    reference="data/Testing_anatomy/reference.tsv"

    # params are in the following order:
    # source target reference emb_size epochs batch_k batch_a lr margin top_k

    with open("params.txt", "w") as f:
        for emb_size in [64, 128, 256]:
            for epochs in [401, 601]:
    	        for batch_k in [32, 64, 128]:
                    for batch_a in [8, 16, 32]:
            	        for lr in [0.1, 0.01, 0.001]:
                            for margin in [0.5, 1.0, 1.5]:
                                for topk in [10, 20, 30]:
                                    f.write(f"{source_owl} {target_owl} {reference} {emb_size} {epochs} {batch_k} {batch_a} {lr} {margin} {topk}\n")


def generate_params_file_fixed_no_max_threshold():
    source_owl="data/Testing_anatomy/mouse.owl"
    target_owl="data/Testing_anatomy/human.owl"
    reference="data/Testing_anatomy/reference.tsv"
    emb_size=128
    epochs=401
    batch_k=32
    batch_a=32
    lr=0.01
    margin=1.5
    topk=20
    min_threshold=0.1
    
    for max_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        with open(f"params_max_threshold.txt", "a") as f:
            f.write(f"{source_owl} {target_owl} {reference} {emb_size} {epochs} {batch_k} {batch_a} {lr} {margin} {topk} {min_threshold} {max_threshold}\n")
    
    
    
if __name__ == "__main__":
    #generate_params_file()
    generate_params_file_fixed_no_max_threshold()
