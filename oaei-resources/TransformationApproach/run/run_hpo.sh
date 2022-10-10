SOURCE_OWL=data/Testing_anatomy/mouse.owl
TARGET_OWL=data/Testing_anatomy/human.owl
REFERENCE=data/Testing_anatomy/reference.tsv

# for loop
for emb_size in 64 128 256
do
    for epochs in 401 601
    do
	for batch_k in 32 64 128
	do
	    for batch_a in 8 16 32
	    do
		for lr in 0.1 0.01 0.001
		do
		    for margin in 0.5 1.0 1.5
		    do
			for topk in 10 20 30
			do
			    # Run the code
			    python training_modelR.py --source $SOURCE_OWL --target $TARGET_OWL --reference $REFERENCE -size $emb_size --epochs $epochs --batch-k $batch_k --batch-a $batch_a --lr $lr --margin $margin --topk $topk
			done
		    done
		done
	    done
	done
    done
done

			    
			       
