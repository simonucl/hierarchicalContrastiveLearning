python train_deepfashion.py --data ./data/bgc/ \
 --train-listfile dev_data.jsonl \
 --val-listfile dev_data.jsonl \
 --class-map-file value_dict.pt \
 --learning_rate 0.5 --temp 0.1 \
 --cosine \
 --gpu 0 \
 --batch-size 20 \
 --device cuda:0 \
#  --model ./models/

#  python train_deepfashion.py --data ./data/bgc/ --train-listfile train_data.jsonl --val-listfile dev_data.jsonl --class-map-file value_dict.pt --learning_rate 0.5 --temp 0.1 --world-size 1 --rank 0 --cosine --gpu 0 --batch-size 20