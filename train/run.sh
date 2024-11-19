
# 不能有 = 号，否则会报错，不能有 逗号，否则会报错
python ./modified_model/train/train.py \
    --data_file /media/zkl/zkl_T7/preprocession/observe_prediction_512 \
    --data_save_path ./modified_model/datas \
    --model_save_path ./modified_model/checkpoints \
    --save_epoch_freq 1 \
    --epochs 600 \
    --start_epoch 0 \
    --batch_size 32 \
    --lrs 0.01 \
    --loss_iter_log 2 \
    --plot_test_fig True \
    --seed 123 \
    --warm_up True \
    --warmup_steps 20 \
    --lr_min 1e-8 \
    --scheduler cosine \
    --optimizer adam \
    --device cuda \
    --e_layer 1 \
    --e_layers 1  \
    --pred_len 512 \
    --output_attention False \
    --enc_in 1 \
    --d_models 128 \
    --d_model 128 \
    --embed fixed \
    --freq h \
    --dropout 0.3 \
    --activation glue \
    --exp_setting 2 \
    --c_out 1 \
    --n_heads 3 \
    --factor 3\
    --ratio 0.7\
    --resume True \
    --saving_fig_log_freq 50


















































