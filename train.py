import argparse
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import wandb

from steerer import FixedVectorSteerer, AdapterSteerer
from custom_trainer import StepwiseTrainer

import sys
sys.path.append('./')
from utils import *
from inference import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--intervene-last", action ='store_true', help="If set, intervene on only the last tokens")
    parser.add_argument("--no-adapter", action ='store_true', help="If set, train fixed vectors instead of adapte module")
    parser.add_argument("--adapter-rank", type=int, default=8, help="Rank of adapter modules (if using adapters)")
    parser.add_argument("--submodules", action ='store_true', help="If set, train on submodules")
    parser.add_argument("--linear", action ='store_true', help="If set, disable nonlinearity in adapters")
    parser.add_argument("--nonlinearity", type=str, default="silu")
    parser.add_argument("--do-eval", action ='store_true', help="If set, do eval in the same script after training, and do not save model")
    parser.add_argument("--test-only", action ='store_true', help="If set, only eval on test set")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    default_epochs = epochs_per_dataset.get(dataset_name, 1)
    train_epochs = args.epochs if args.epochs is not None else default_epochs
    dataset_class = dataset_dict[dataset_name]

    print("=" * 20)
    print(f"Model Name: {args.model_name}")
    print(f"Train Epochs: {train_epochs}")
    print(f"Learning Rate: {args.lr}")

    model_name = args.model_name
    if "gemma" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Freeze base model
    for p in base_model.parameters():
        p.requires_grad_(False)

    # We want to steer layers 10 and 15, and only at MLP and attention outputs
    if args.submodules:
        targets=("mlp", "attn")
    else:
        targets=("block")
    if args.no_adapter:
        steered_model = FixedVectorSteerer(
            base_model,
            layers_to_steer="all",
            targets=targets,
            apply_to="all" if not args.intervene_last else "last",
        )
    else:
        steered_model = AdapterSteerer(
            base_model,
            layers_to_steer="all",
            targets=targets,
            apply_to="all" if not args.intervene_last else "last",
            rank=args.adapter_rank if hasattr(args, 'adapter_rank') else 8,
            activation=None if args.linear else args.nonlinearity,
        )

    # Check trainable params: should be just the deltas
    print("Trainable params:")
    for n, p in steered_model.named_parameters():
        if p.requires_grad:
            print(n, p.shape)
    print("Percentage trainable params:", sum(p.numel() for p in steered_model.parameters() if p.requires_grad)/sum(p.numel() for p in base_model.parameters())*100, "%")

    # --- Dataset & preprocessing (toy example) ---

    if args.no_adapter:
        project_str = "fixedvec_all" if not args.intervene_last else "fixedvec_last"
        project_str = f"{project_str}_linear" if args.linear else f"{project_str}_nonlinear"
        project_name = f"{dataset_name}_model/{model_name.split('/')[-1]}_{project_str}/lr{args.lr}_bs{args.bs}_ep{train_epochs}_warmup{args.warmup_ratio}_wd{args.weight_decay}"
    else:
        project_str = "adapter_all" if not args.intervene_last else "adapter_last"
        project_str = f"{project_str}_linear" if args.linear else f"{project_str}_nonlinear_{args.nonlinearity}"
        project_name = f"{dataset_name}_model/{model_name.split('/')[-1]}_{project_str}/lr{args.lr}_bs{args.bs}_ep{train_epochs}_warmup{args.warmup_ratio}_wd{args.weight_decay}_rank{args.adapter_rank}"

    if "gemma" in model_name.lower():
        train_dataset, eval_dataset = dataset_class(mode="train").load_tokenized_dataset_chat(tokenizer=tokenizer)
    else:
        train_dataset, eval_dataset = dataset_class(mode="train").load_tokenized_dataset(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./{project_name}",
        per_device_train_batch_size=args.bs,
        num_train_epochs=train_epochs,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        remove_unused_columns=False,
        # Ensure training loss gets logged to wandb frequently for short runs
        logging_strategy="steps",
        logging_steps=10,
        report_to=["wandb"],
        # Tied weights (embed_tokens/lm_head) trip safetensors; save as torch instead
        save_safetensors=False,
        save_total_limit=1,
        save_strategy="no",         
        group_by_length=True,
    )

    if args.intervene_last:
        trainer = StepwiseTrainer(
            model=steered_model,
            args=training_args,
            train_dataset=train_dataset,
        )
    else:
        trainer = Trainer(
            model=steered_model,
            args=training_args,
            train_dataset=train_dataset,
        )
        

    trainer.train()
    if not args.do_eval:
        trainer.save_model(project_name)
    else:
        dataset_obj = dataset_class(mode="eval")
        if "gemma" in model_name.lower():
            datasets = dataset_obj.load_raw_dataset_chat(tokenizer, return_test=True) 
        else:
            datasets = dataset_obj.load_raw_dataset(return_test=True)
        validation_dataset = datasets["validation"]
        test_dataset = datasets["test"]
        eval_fn = dataset_obj.evaluate

        output_file = os.path.join("outputs", project_name, "validation.json")
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        if not args.test_only:
            print("=" * 20, "VALIDATION SET", "=" * 20)
            print("\n")
            evaluate_model(validation_dataset, eval_fn, steered_model, tokenizer, output_file, max_new_tokens=max_new_tokens_per_dataset.get(dataset_name, 100))

        output_file = os.path.join("outputs", project_name, "test.json")
        print("=" * 20, "TEST SET", "=" * 20)
        print("\n")
        evaluate_model(test_dataset, eval_fn, steered_model, tokenizer, output_file, max_new_tokens=max_new_tokens_per_dataset.get(dataset_name, 100))
        print("=" * 20)
        print("\n")
