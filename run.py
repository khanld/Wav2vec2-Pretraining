import argparse
import math
import os
import pandas as pd
import numpy as np
import soundfile as sf

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm

import datasets
import torch
from torch.utils.data import Dataset
from datasets import DatasetDict, concatenate_datasets, load_dataset, IterableDatasetDict
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from scheduler import WarmupLR

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from huggingface_hub import Repository
from transformers import (
    SchedulerType,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    get_scheduler,
    set_seed,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from transformers.utils import get_full_repo_name
import time
from accelerate import DistributedDataParallelKwargs


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--train_datasets",
        nargs="+",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--val_datasets",
        nargs="+",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--separator",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training.",
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--saving_steps",
        type=int,
        default=500,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--audio_column_name",
        type=str,
        default="path",
        help="Column in the dataset that contains speech file path. Defaults to 'audio'",
    )
    parser.add_argument(
        "--duration_column_name",
        type=str,
        default="duration",
        help="Column in the dataset that contains speech file path. Defaults to 'audio'",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--load_from_pretrained",
        action="store_true",
        help="Whether to load pretrained model from model_name_or_path."
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_gumbel_temperature",
        type=float,
        default=2.0,
        help="Maximum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--min_gumbel_temperature",
        type=float,
        default=0.5,
        help="Minimum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--gumbel_temperature_decay", type=float, default=0.999995, help="Decay of gumbel temperature during training."
    )
    parser.add_argument(
        "--max_duration_in_seconds",
        type=float,
        default=5.0,
        help="Filter out audio files that are longer than `max_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--min_duration_in_seconds",
        type=float,
        default=3.0,
        help="Filter out audio files that are shorter than `min_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=None,
        help=(
            "If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the"
            " use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer",
    )

    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


class CustomDataset(Dataset):
    def __init__(self, files, sep, sr, audio_column_name, duration_column_name, min_duration, max_duration):
        self.sep = sep
        self.sr = sr
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_column_name = audio_column_name
        self.duration_column_name = duration_column_name
        self.data = self.load_ds(files)
    
    def load_ds(self, all_files):
        li = []
        for filename in all_files:
            df = pd.read_csv(filename, sep=self.sep, engine="python")
            li.append(df)
        data = pd.concat(li, axis=0, ignore_index=True)

        if self.duration_column_name in data.columns:
            data = data[data[self.duration_column_name] >= self.min_duration]
            print("Mean duration: ", data[self.duration_column_name].mean())
        return data

    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        batch = {}
        batch["input_values"] = sf.read(item[self.audio_column_name])[0]

        if len(batch["input_values"])//self.sr > self.max_duration:
            start = np.random.randint(0, len(batch["input_values"]) - self.max_duration * self.sr)
            batch["input_values"] = batch["input_values"][start : start + int(self.max_duration * self.sr)]

        return batch


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask")
        )
        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch


def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)


def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def main():
    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], step_scheduler_with_optimizer=False)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # set up tensorboard if available
        writer = SummaryWriter(args.output_dir + '/logs', max_queue=5, flush_secs=30)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

                
    accelerator.wait_for_everyone()

    # Download data
    train_dataset = CustomDataset(
        args.train_datasets, 
        sep=args.separator, 
        audio_column_name=args.audio_column_name, 
        duration_column_name=args.duration_column_name,
        sr=16000, 
        min_duration=args.min_duration_in_seconds, 
        max_duration=args.max_duration_in_seconds)

    val_dataset = CustomDataset(
        args.val_datasets, 
        sep=args.separator, 
        audio_column_name=args.audio_column_name, 
        duration_column_name=args.duration_column_name,
        sr=16000, 
        min_duration=args.min_duration_in_seconds, 
        max_duration=args.max_duration_in_seconds)


    # Load feature_extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
    # only normalized-inputs-training is supported
    if not feature_extractor.do_normalize:
        raise ValueError(
            "Training is only supported for normalized inputs. Make sure ``feature_extractor.do_normalize == True``"
        )

    # Load model config
    config = Wav2Vec2Config.from_pretrained(args.model_name_or_path)
    # if not config.do_stable_layer_norm or config.feat_extract_norm != "layer":
    #     raise ValueError(
    #         "PreTraining is only supported for ``config.do_stable_layer_norm=True`` and"
    #         " ``config.feat_extract_norm='layer'"
    #     )


    # initialize random model
    model = Wav2Vec2ForPreTraining(config)
    if args.load_from_pretrained is not None:
        try:
            model = model.from_pretrained(args.model_name_or_path)
        except:
            print("!!!!! Warning: Pretrained model may not exist. Start training from Scratch")

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Define data collator, optimizer and scheduler
    data_collator = DataCollatorForWav2Vec2Pretraining(
        model=model, feature_extractor=feature_extractor, pad_to_multiple_of=args.pad_to_multiple_of
    )

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=16
    )

    eval_dataloader = DataLoader(
        val_dataset,
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size,
        num_workers=16
    )

    # Optimizer
    optimizer =  torch.optim.AdamW(
        list(model.parameters()),
        lr=args.learning_rate,
        betas=[args.adam_beta1, args.adam_beta2],
        eps=args.adam_epsilon,
    )
    lr_scheduler = WarmupLR(
        optimizer=optimizer,
        warmup_steps=args.num_warmup_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    if args.resume:
        print("******Resume checkpoint******")
        accelerator.load_state(args.output_dir)
        checkpoint = torch.load(os.path.join(args.output_dir, 'latest_checkpoint.pt'), 
                                map_location="cpu")


    # Train
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)


    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch


    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        print("Number of training data: ", len(train_dataset))
        print("total_batch_size: ", total_batch_size)
        print("num_update_steps_per_epoch: ", num_update_steps_per_epoch)
        print("num_train_epochs: ", args.num_train_epochs)

    # Only show the progress bar once on each machine.
    completed_steps = checkpoint['completed_steps'] + 1 if args.resume else 0
    starting_epoch = checkpoint['epoch'] if args.resume else 0
    progress_bar = tqdm(initial = completed_steps, total = args.max_train_steps, disable=not accelerator.is_local_main_process)

    print(f"******STARTING AT EPOCH {starting_epoch} - STEP {completed_steps}******")


    for epoch in range(starting_epoch, args.num_train_epochs):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch}: ")
        model.train()
        for step, batch in enumerate(train_dataloader):
            # compute num of losses
            num_losses = batch["mask_time_indices"].sum()
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
            )
            percent_masked = num_losses / sub_attention_mask.sum()

            # forward
            outputs = model(**batch)

            # divide loss by gradient accumulation steps since gradients
            # are accumulated for multiple backward passes in PyTorch
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            # make sure that `num_losses` is summed for distributed training
            # and average gradients over losses of all devices
            if accelerator.state.num_processes > 1:
                num_losses = accelerator.gather(num_losses).sum()
                gradient_multiplier = accelerator.state.num_processes / num_losses
                multiply_grads(model.module.parameters(), gradient_multiplier)
            else:
                multiply_grads(model.parameters(), 1 / num_losses)

            # update step
            if (step + 1) % args.gradient_accumulation_steps == 0:

                # compute grad norm for monitoring
                scale = (
                    accelerator.scaler._scale.item()
                    if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                    else 1
                )
                if accelerator.state.num_processes > 1:
                    grad_norm = get_grad_norm(model.module.parameters(), scale)
                else:
                    grad_norm = get_grad_norm(model.parameters(), scale)

                # update parameters
                optimizer.step()
                optimizer.zero_grad()

                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                elif accelerator.is_local_main_process:
                    progress_bar.write(
                        f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                    )

                # update gumbel temperature
                gumbel_temperature = max(
                    args.max_gumbel_temperature * args.gumbel_temperature_decay**completed_steps,
                    args.min_gumbel_temperature,
                )
                if hasattr(model, "module"):
                    model.module.set_gumbel_temperature(gumbel_temperature)
                else:
                    model.set_gumbel_temperature(gumbel_temperature)

                progress_bar.update(1)
                completed_steps += 1

            # Log all results
            if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0:
                loss.detach()
                outputs.contrastive_loss.detach()
                outputs.diversity_loss.detach()
                cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)
                cosine_sim = cosine_sim[batch["mask_time_indices"].to(torch.bool)].mean()

                if accelerator.state.num_processes > 1:
                    loss = accelerator.gather(loss).sum()
                    outputs.contrastive_loss = accelerator.gather(outputs.contrastive_loss).sum()
                    outputs.diversity_loss = accelerator.gather(outputs.diversity_loss).sum()
                    percent_masked = accelerator.gather(percent_masked).sum()
                    cosine_sim = accelerator.gather(cosine_sim).mean()

                train_logs = {
                    "step": torch.tensor((step + 1) // args.gradient_accumulation_steps, dtype=torch.int32),
                    "loss": (loss * args.gradient_accumulation_steps) / num_losses,
                    "contrast_loss": outputs.contrastive_loss / num_losses,
                    "div_loss": outputs.diversity_loss / num_losses,
                    "%_mask_idx": percent_masked / accelerator.num_processes,
                    "ppl": outputs.codevector_perplexity,
                    "lr": torch.tensor(lr_scheduler.get_lr()),
                    "temp": torch.tensor(gumbel_temperature),
                    "grad_norm": torch.tensor(grad_norm),
                    "cosine_sim": cosine_sim * 100
                }
                log_str = ""
                for k, v in train_logs.items():
                    log_str += "| {}: {:.3e}".format(k, v.item())

                if accelerator.is_local_main_process:
                    progress_bar.write(log_str)
                    for k, v in train_logs.items():
                        writer.add_scalar('TRAIN' + '/' + k, v, completed_steps)
                    

            # save model every `args.saving_steps` steps
            if (step + 1) % (args.gradient_accumulation_steps * args.saving_steps) == 0:
                if (args.push_to_hub and epoch < args.num_train_epochs - 1) or args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                            args.output_dir + f'/saved_model/epoch_{epoch}', is_main_process=accelerator.is_main_process, save_function=accelerator.save
                        )
                    if accelerator.is_main_process:
                        feature_extractor.save_pretrained(args.output_dir + f'/saved_model/epoch_{epoch}')
                        print("****Saving checkpoint*****")
                        state_dict = {
                            "completed_steps": completed_steps,
                            "epoch": epoch
                        }
                        torch.save(state_dict, os.path.join(args.output_dir, "latest_checkpoint.pt"))
                    accelerator.save_state(args.output_dir)

                if (args.push_to_hub and epoch < args.num_train_epochs - 1) and accelerator.is_main_process:
                    repo.push_to_hub(
                        commit_message=f"Training in progress step {completed_steps}",
                        blocking=False,
                        auto_lfs_prune=True,
                    )

            # if completed steps > `args.max_train_steps` stop
            if completed_steps >= args.max_train_steps:
                break

        print("******END OF EPOCH******")
        # Validate!
        model.eval()

        # init logs
        val_logs = {
            "val_loss": 0,
            "val_contrastive_loss": 0,
            "val_diversity_loss": 0,
            "val_num_losses": 0,
        }
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch.pop("sub_attention_mask", None)
                outputs = model(**batch)

            val_logs["val_loss"] += outputs.loss
            val_logs["val_contrastive_loss"] += outputs.contrastive_loss
            val_logs["val_diversity_loss"] += outputs.diversity_loss
            val_logs["val_num_losses"] += batch["mask_time_indices"].sum()

        # sum over devices in multi-processing
        if accelerator.num_processes > 1:
            val_logs = {k: accelerator.gather(v).sum() for k, v in val_logs.items()}

        val_logs = {k: v / val_logs["val_num_losses"] for k, v in val_logs.items()}

        log_str = ""
        for k, v in val_logs.items():
            log_str += "| {}: {:.3e}".format(k, v.item())

        if accelerator.is_local_main_process:
            progress_bar.write(log_str)
            for k, v in val_logs.items():
                writer.add_scalar('VALIDATION' + '/' + k, v, epoch)

            

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                    args.output_dir + f'/saved_model/epoch_{epoch}', is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
            if accelerator.is_main_process:
                feature_extractor.save_pretrained(args.output_dir + f'/saved_model/epoch_{epoch}')
                print("****Saving checkpoint*****")
                state_dict = {
                    "completed_steps": completed_steps,
                    "epoch": epoch
                }
                torch.save(state_dict, os.path.join(args.output_dir, "latest_checkpoint.pt"))

            accelerator.save_state(args.output_dir)
            if accelerator.is_main_process:
                if args.push_to_hub:
                    repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


if __name__ == "__main__":
    main()
