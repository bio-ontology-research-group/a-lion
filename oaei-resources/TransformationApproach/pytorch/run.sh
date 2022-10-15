base=/home/zhapacfp/Github/a-lion/oaei-resources/TransformationApproach/pytorch/data/test_mse/firstTestCase/

source=$base/MaterialInformation-Reduced.owl
target=$base/MatOnto.owl
reference=$base/RefAlign1.tsv
embedding_size=128
batch_size_kg=32
batch_size_alignment=32
epochs=400
learning_rate=0.0001
margin=0.5
device='cuda'
aim='test'
seed=42
root=$base/testtorch/

python main.py --source $source --target $target --reference $reference --embedding-size $embedding_size --batch-size-kg $batch_size_kg --batch-size-alignment $batch_size_alignment --epochs $epochs --learning-rate $learning_rate --margin $margin --device $device --aim $aim --seed $seed --root $root

