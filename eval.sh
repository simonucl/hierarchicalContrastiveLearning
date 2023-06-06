	python eval_deepfashion.py --data ./deep_fashion/ \
	--epochs 100 \
	--train-listfile train_listfile.json --val-listfile val_listfile.json \
	--test-listfile test_listfile.json \
	--class-map-file class_map.json \
	--num-classes 17 \
	--learning_rate 0.5 \
    --repeating-product-file repeating_product_ids.csv \
	--ckpt ./model/hmlc_dataset_resnet50_lr_0.5_decay_0.1_bsz_256_loss_hmce_trial_5/checkpoint_0100.pth.tar