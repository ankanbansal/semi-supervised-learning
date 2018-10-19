CUDA_VISIBLE_DEVICES=6 python main.py --log_dir \
./sup_4k_tot_50k/logs_MEL_reg/sm/ratio0.7/cosine/alpha0.8_delta0.5/ \
    --type cls_MEL_reg --reg_distance_type cosine --data_dir \
/vulcan/scratch/ankan/Amazon/data/ \
    --val_on True \
    --save_dir \
./sup_4k_tot_50k/checkpoints_MEL_reg/sm/ratio0.7/cosine/alpha0.8_delta0.5/ \
    --batch_size 128 \
    --num_workers 0 \
    --delta 0.5 \
    --runs 5 \
    --lr_step 40 \
    --epochs 100 \
    --print_freq 100
