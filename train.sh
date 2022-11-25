CUDA_VISIBLE_DEVICES="0" accelerate launch \
	--multi_gpu \
	--num_machines="1" \
	--num_processes="1" \
	--mixed_precision="fp16" \
	--num_cpu_threads_per_process="12" \
	run_wav2vec2_pretraining_no_trainer.py \
		--train_datasets \ 
			data/train_clean_100.tsv \
			data/train_clean_360.tsv \
			data/train_other_500.tsv \
		--val_datasets \
			data/dev_clean.tsv \
			data/dev_other.tsv \
		--audio_column_name="path" \
		--duration_column_name="duration" \
		--separator="\t" \
		--model_name_or_path="facebook/wav2vec2-base" \
		--load_from_pretrained \
		--output_dir="/data1/speech/khanhld/wav2vec-pretraining-hpc3/wav2vec2_pretraining-production" \
		--max_train_steps="300000" \
		--num_warmup_steps="90000" \
		--gradient_accumulation_steps="8" \
		--learning_rate="0.005" \
		--weight_decay="0.01" \
		--max_duration_in_seconds="15.6" \
		--min_duration_in_seconds="0.5" \
		--logging_steps="1" \
		--saving_steps="100" \
		--per_device_train_batch_size="16" \
		--per_device_eval_batch_size="8" \
		--adam_beta1="0.9" \
		--adam_beta2="0.98" \
		--adam_epsilon="1e-06" \
		--gradient_checkpointing

