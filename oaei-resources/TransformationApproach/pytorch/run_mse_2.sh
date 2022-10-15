base=/home/zhapacfp/Github/a-lion/oaei-resources/TransformationApproach/pytorch/data/test_mse/secondTestCase/

source=$base/MaterialInformation.owl
target=$base/MatOnto.owl
reference=$base/RefAlign2.tsv
embedding_size=100
batch_size_kg=32
batch_size_alignment=32
epochs=400
learning_rate=0.001
margin=1.0
device='cuda'
aim='all'
seed=42
root=$base/testtorch2/

python main.py --source $source --target $target --reference $reference --embedding-size $embedding_size --batch-size-kg $batch_size_kg --batch-size-alignment $batch_size_alignment --epochs $epochs --learning-rate $learning_rate --margin $margin --device $device --aim $aim --seed $seed --root $root

