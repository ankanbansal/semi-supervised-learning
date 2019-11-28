CUDA_VISIBLE_DEVICES=6 python main.py --log_dir \
./temp/logs/ \
    --type cls_clust_reg --reg_distance_type cosine --data_dir \
/vulcan/scratch/ankan/Amazon/data/ \
    --resume temp/checkpoints/run_0/checkpoint_cls_MEL_reg_epoch_98.pth.tar\
    --val_on True \
    --save_dir \
./temp/checkpoints/ \
    --batch_size 128 \
    --num_workers 0 \
    --delta 0.5 \
    --runs 1 \
    --learning_rate 0.1 \
    --lr_step 40 \
    --epochs 100 \
    --print_freq 100 \
    --temp_file ./temp/sup_indices.json \
    --sup_indices_file ./temp/sup_indices.json
