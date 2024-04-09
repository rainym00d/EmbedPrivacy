# Sample Data
`./sample_data` is a sample of the wikipedia, which is embeded with bge-large-en. It contains three parts: train, valid and test. Each part has two attribues: text and embedding. The number of three parts are 1000, 100 and 100 respectively.

# Sample Model Checkpoint

Due to all model files are too large, we release one of the sample to help understanding the code. This checkpoint is trained with the `wikipedia_xl`, the base model is gpt2, and the embedding model is `bge-large-en`. You can download it from [here](https://huggingface.co/rainym00d/bge_large_en-wikipedia_xl-gpt2)

# Train

Below is a training example. You can change the arguments to adapt to suit your situation.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    deepspeed src/main.py \
        --model_name_or_path gpt2 \
        --from_pretrained true \
        --dataset_name_or_path wikipedia \
        --dataset_save_dir {your_data_path} \
        --train_num {your_data_num} \
        --embedding_model bge-large-en \
        --embedding_dim 1024 \
        --output_dir {your_output_dir} \
        --overwrite_output_dir true \
        --num_train_epochs 200 \
        --warmup_ratio 0.1 \
        --learning_rate 5e-5 \
        --per_device_train_batch_size 150 \
        --per_device_eval_batch_size 100 \
        --eval_steps 1000 \
        --save_steps 1000 \
        --logging_steps 100 \
        --run_name {your_output_name} \
        --deepspeed ./ds_config/ds_config_stage2.json
```

# Test

Below is a test example. You can change the arguments to adapt to suit your situation.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    deepspeed src/main.py \
        --model_name_or_path {your_trained_model_name} \
        --cache_dir {your_cache_dir} \
        --from_pretrained true \
        --test_dataset_name_or_path {your_dataset_name} \
        --dataset_save_dir {your_dataset_dir} \
        --embedding_model e5-large-v2 \
        --embedding_dim 1024 \
        --output_dir {your_output_dir} \
        --overwrite_output_dir false \
        --per_device_eval_batch_size 50 \
        --bf16 true \
        --do_train false \
        --do_predict true \
        --use_wandb false \
        --do_sample true
```

# Attribute inference attack test

If the similarity model and the target model are different, you can refer below example.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    deepspeed src/pred.py \
        --model_name_or_path {your_trained_model_name} \
        --cache_dir {your_cache_dir} \
        --from_pretrained true \
        --test_dataset_name_or_path {your_dataset_name} \
        --dataset_save_dir {your_dataset_dir}  \
        --embedding_model e5-large-v2 \
        --embedding_dim 1024 \
        --output_dir {your_output_dir} \
        --overwrite_output_dir false \
        --per_device_eval_batch_size 50 \
        --bf16 true \
        --do_train false \
        --do_predict true \
        --use_wandb false \
        --do_sample true

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    deepspeed src/attr_e5.py \
        --dataset_list {your_dataset_name} \
        --dataset_save_dir "{your_output_dir}/generated_predictions" \
        --output_dir {your_output_dir}
```

If the similarity model is the same as the target model, you can refer below example.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    deepspeed src/attr_same_embedding_model_bge.py \
        --dataset_list {your_dataset_name} \
        --dataset_save_dir "{your_output_dir}/generated_predictions" \
        --output_dir {your_output_dir}
```