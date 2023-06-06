python train_deepfashion.py --data ./deep_fashion/ \
 --train-listfile train_listfile.json \
 --val-listfile val_listfile.json \
 --class-map-file class_map.json \
 --repeating-product-file repeating_product_ids.csv \
 --num-classes 17 \
 --learning_rate 0.5 --temp 0.1 \
 --ckpt ./model/hmlc_dataset_resnet50_lr_0.5_decay_0.1_bsz_256_loss_hmce_trial_5/checkpoint_0100.pth.tar \
 --world-size 1 --rank 0 --cosine \
 --start_epoch 100 \
#  --ckpt /pretrained_model/ \

 # python train_deepfashion.py --data ./deep_fashion/ \
#     --train-listfile train_listfile.json \
#     --val-listfile val_listfile.json \
#     --class-map-file class_map.json \
#     --repeating-product-file repeating_product_ids.csv \
#     --num-classes 17 \
#     --learning_rate 0.5 --temp 0.1 \
#     --ckpt ./pretrained/resnet50-19c8e357.pth \
#     --world-size 1 --rank 0 --cosine \
#     --gpu 0 \
