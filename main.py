from torch.nn.utils.rnn import pad_sequence
from utility import (data_selection, 
custom_collator, 
preprocess_function,
finetune_model_eval,
)
import torch
import argparse

from metric import SimAOU

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset


# Load Qwen3 tokenizer and model
model_name = "Qwen/Qwen3-0.6B"
model_path = "/home/students/wli/UniHeidelberg/semster2/final_projects/models/Qwen3-0.6B-Base"
# model_path = model_name  #uncommment this for colab for kaggle environment 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "auto")
tokenizer.padding_side = "left"

# Data collator (handles padding dynamically in batch)
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Make sure tokenizer has pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def main(parser):
    args = parser.parse_args()

    # Load dataset based on data_type
    if args.data_type == "sst2":
        dataset = load_dataset("glue", "sst2")
        train_dataset = dataset["train"]
        test_dataset = dataset["validation"]
    elif args.data_type == "sst5":
        dataset = load_dataset("SetFit/sst5")
        train_dataset = dataset["train"]
        test_dataset = dataset["validation"]
        # Add idx column for sst5 to track original indices
        test_dataset = test_dataset.map(lambda example, idx: {**example, "idx": idx}, with_indices=True)
    else:
        raise ValueError(f"Unsupported data_type: {args.data_type}")
    



    # Dispatch
    if args.command == "data" and getattr(args, "subcommand", None) in ["select", "dataselect"]:
        ideal_seed, candidates, ideal_demo = data_selection(
            model,
            tokenizer,
            train_dataset,
            test_dataset,
            data_type=args.data_type,
            num_data_points=args.num_data_points,
            seed_max=args.seed_max,
            batch_size=args.batch_size,
        )
        print(f"Best seed: {ideal_seed}")
        print(f"Selected candidate idxs (subset): {candidates[:10]}{'...' if len(candidates) > 10 else ''}")
        return
    elif args.command == "dataselect":
        ideal_seed, candidates, ideal_demo = data_selection(
            model,
            tokenizer,
            train_dataset,
            test_dataset,
            data_type=args.data_type,
            num_data_points=args.num_data_points,
            seed_max=args.seed_max,
            batch_size=args.batch_size,
        )
        print(f"Best seed: {ideal_seed}")
        print(f"Selected candidate idxs (subset): {candidates[:10]}{'...' if len(candidates) > 10 else ''}")
        return
    elif args.command  == "finetune_eval":
        finetune_model_eval(model, tokenizer,train_dataset,test_dataset, num_data_points = args.num_data_points, best_seed = args.seed, data_type = args.data_type)
    elif args.command == "SimAOU":
        SimAOU(model, tokenizer, train_dataset, test_dataset,  best_seed = args.seed, data_type = args.data_type, num_data_points = args.num_data_points)

    # Default behavior when no command is provided
    parser.print_help()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-form attention CLI")
    subparsers = parser.add_subparsers(dest="command")
    # parser.add_argument("--data-type", type=str, default="sst2", help="Dataset type identifier")
    # Level-1: data
    p_data = subparsers.add_parser("data", help="Data-related commands")
    subparsers_data = p_data.add_subparsers(dest="subcommand")

    # Level-2: select (aka dataselect)
    p_select = subparsers_data.add_parser("select", aliases=["dataselect"], help="Run data selection routine")
    p_select.add_argument("--num-data-points", type=int, default=32, help="Number of training examples to sample per seed")
    p_select.add_argument("--seed-max", type=int, default=100, help="Number of random seeds to try")
    p_select.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    p_select.add_argument("--data-type", type=str, default="sst2", help="Dataset type identifier")

    # Also support a single-word shortcut command for convenience
    p_ds = subparsers.add_parser("dataselect", help="Shortcut for data select")
    p_ds.add_argument("--num-data-points", type=int, default=32)
    p_ds.add_argument("--seed-max", type=int, default=100)
    p_ds.add_argument("--batch-size", type=int, default=32)
    p_ds.add_argument("--data-type", type=str, default="sst2")

    p_train = subparsers.add_parser("finetune_eval", help = "fine tune model for one epoch")
    p_train.add_argument("--num-data-points", type=int, default=32, help="Number of training examples to fine-tune on")
    p_train.add_argument("--seed", type=int, default=0, help="Random seed for sampling training data")
    p_train.add_argument("--data-type", type=str, default="sst2", help="Dataset type identifier")

    p_SimAOU = subparsers.add_parser("SimAOU", help = "Calcuate SimAOU")
    p_SimAOU.add_argument("--data-type", type=str, default="sst2", help="Dataset type identifier")
    p_SimAOU.add_argument("--num-data-points", type=int, default=32, help="Number of training examples to sample per seed")
    p_SimAOU.add_argument("--seed", type=int, default=0, help="Random seed for sampling training data") 



    main(parser)
