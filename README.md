# HOW TO PRETRAIN WAV2VEC ON YOUR OWN DATASETS
Now you can pre-train Wav2vec 2.0 model on your dataset, push it into the Huggingface hub, and finetune it on downstream tasks with just a few lines of code. Follow the below instruction on how to use it.
<a name = "documentation" ></a>
### Documentation
Most of the codes originated from [run_wav2vec2_pretraining_no_trainer.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py) from Huggingface but are more friendly and easier to use, visualize and control the training process. </br>
### New supporting features include:
- [x] Train on your own local datasets
- [x] Visualization with Tensorboard 
- [x] Resume training (optimizer, a learning rate scheduler, gradient scaler,...) from the latest checkpoint.


<a name = "installation" ></a>
### Installation
```
pip install -r requirements.txt
```

<a name = "usage" ></a>
### Usage
1. Prepare your dataset
    - Your dataset can be in <b>.txt</b> or <b>.csv</b> format.
    - Only <b>PATH</b> column is compulsory, the others (eg: DURATION, TRANSCRIPT, ...) are not nescessary. <b>PATH</b> contains the paths to your stored audio files. Depending on your dataset location, it can be either absolute paths or relative paths.
    - Check out our [data_example.csv](examples/data_example.csv) file for more information.

3. Run
    - Train:
        ```
        CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
        --multi_gpu \
        --num_machines="1" \
        --num_processes="2" \
        --mixed_precision="fp16" \
        --num_cpu_threads_per_process="16" \
        run_wav2vec2_pretraining_no_trainer.py \
            --train_datasets data/train_clean_100.tsv data/train_clean_360.tsv data/train_other_500.tsv \
            --val_datasets data/dev_clean.tsv data/dev_other.tsv \
            --train_cache_file_name="cache/train_960h.arrow" \
            --validation_cache_file_name="cache/validation.arrow" \
            --audio_column_name="PATH" \
            --model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
            --output_dir="/wav2vec2-pretrained-960h" \
            --max_train_steps="200000" \
            --num_warmup_steps="32000" \
            --gradient_accumulation_steps="8" \
            --learning_rate="0.005" \
            --weight_decay="0.01" \
            --max_duration_in_seconds="20.0" \
            --min_duration_in_seconds="2.0" \
            --logging_steps="1" \
            --saving_steps="50" \
            --per_device_train_batch_size="8" \
            --per_device_eval_batch_size="8" \
            --adam_beta1="0.9" \
            --adam_beta2="0.98" \
            --adam_epsilon="1e-06" \
            --gradient_checkpointing \
        ```
    - Resume: Same as Train, but with an additional argument
        ```
        --resume
        ```

<a name = "logs" ></a>
### Logs and Visualization
The logs during the training will be stored, and you can visualize it using TensorBoard by running this command:
```
# specify the <name> in config.json
tensorboard --logdir ~/<output_dir>/logs

# specify a port 8080
tensorboard --logdir ~/<output_dir>/logs --port 8080
```