CUDA_VISIBLE_DEVICES=1 python main.py --log_dir \
./sup_4k_tot_50k/logs_MEL_reg/feat/ratio0.7/alpha0.8_delta0.4/ \
    --type cls_MEL_reg --reg_distance_type Euclidean --data_dir \
/fs/janus-scratch/ankan/Rainforest/data/ \
    --val_on True \
    --save_dir \
./sup_4k_tot_50k/checkpoints_MEL_reg/feat/ratio0.7/alpha0.8_delta0.4/ \
    --batch_size 128 \
    --num_workers 0 \
    --delta 0.4 \
    --runs 5 \
    --lr_step 40 \
    --epochs 100 \
    --print_freq 100
