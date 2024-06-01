# Random Projections for Efficient Gradient Low Rank Approximation


### Install experiment dependencies

```bash
pip install -r exp_requirements.txt
```

## Benchmark 1: Pre-Training LLaMA on C4 dataset

```bash
# LLaMA-60M
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.005 \
    --galore_scale 0.25 \
    --rank 128 \
    --update_proj_gap 50 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer galore_adamw \
    --proj_type gaussian 
```

```bash
# LLaMA-130M
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 256 \
    --update_proj_gap 50 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer galore_adamw \
    --proj_type gaussian 
```

Currently per-layer weight updates technique is only supported for single GPU training (`--single_gpu`) without using `nn.parallel.DistributedDataParallel`. We are working on supporting multi-GPU training with per-layer weight updates.

## Benchmark 2: Fine-Tuning RoBERTa on GLUE tasks
```bash
python run_glue.py \
    --model_name_or_path roberta-base \
    --task_name mrpc \
    --enable_galore \
    --lora_all_modules \
    --max_length 512 \
    --seed=1234 \
    --lora_r 4 \
    --galore_scale 4 \
    --per_device_train_batch_size 16 \
    --update_proj_gap 500 \
    --learning_rate 1e-5 \
    --num_train_epochs 30 \
    --proj_type gaussian \
    --output_dir results/ft/roberta_base/mrpc