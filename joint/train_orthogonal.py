import sys
sys.path.append('../')
from utils import *
from inference import evaluate_model

sys.path.append('./')
from joint_steerer import AdapterSteerer

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
import wandb
import torch

import os
import argparse


def _get_layers(model):
    try:
        return model.model.layers
    except Exception:
        if hasattr(model.model, "text_model"):
            return model.model.text_model.layers
        return model.model.model.layers


def _merged_weight(module, adapter_name=None):
    base_w = module.base_layer.weight if hasattr(module, "base_layer") else module.weight
    if hasattr(module, "get_delta_weight"):
        name = adapter_name or getattr(module, "active_adapter", None) or "default"
        if isinstance(name, (list, tuple)):
            name = name[0] if name else "default"
        return base_w + module.get_delta_weight(name)
    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
        name = adapter_name or getattr(module, "active_adapter", None) or "default"
        if isinstance(name, (list, tuple)):
            name = name[0] if name else "default"
        if name in module.lora_A and name in module.lora_B:
            A = module.lora_A[name].weight
            B = module.lora_B[name].weight
            scaling = module.scaling[name] if isinstance(module.scaling, dict) else module.scaling
            delta = B @ A
            if scaling is not None:
                delta = delta * scaling
            return base_w + delta
    return base_w


def _lora_delta(module, adapter_name=None):
    if hasattr(module, "get_delta_weight"):
        name = adapter_name or getattr(module, "active_adapter", None) or "default"
        if isinstance(name, (list, tuple)):
            name = name[0] if name else "default"
        return module.get_delta_weight(name)
    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
        name = adapter_name or getattr(module, "active_adapter", None) or "default"
        if isinstance(name, (list, tuple)):
            name = name[0] if name else "default"
        if name in module.lora_A and name in module.lora_B:
            A = module.lora_A[name].weight
            B = module.lora_B[name].weight
            scaling = module.scaling[name] if isinstance(module.scaling, dict) else module.scaling
            delta = B @ A
            if scaling is not None:
                delta = delta * scaling
            return delta
    return None

def orthogonalize_mlp_adapters(steered_model):
    base_model = steered_model.base_model
    layers = _get_layers(base_model)

    with torch.no_grad():
        for layer_idx, layer in enumerate(layers):
            if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
                continue

            down_proj = layer.mlp.down_proj
            # A = _merged_weight(down_proj)
            B = down_proj.lora_B.default.weight
            if args.submodules:
                key = f"layer{layer_idx}_mlp"
            else:
                key = f"layer{layer_idx}_block"
            if key not in steered_model.adapters:
                continue
            adapter = steered_model.adapters[key]
            A = adapter.down.weight.data
            _, _, Vh = torch.linalg.svd(B.transpose(0, 1), full_matrices=False)
            V = Vh.transpose(-2, -1)
            proj = torch.eye(V.shape[0], device=V.device, dtype=V.dtype) - (V @ V.T)
            adapter.down.weight.data = adapter.down.weight.data @ proj


class OrthogonalizeCallback(TrainerCallback):
    def __init__(self, freq):
        self.freq = freq

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.freq == "init":
            orthogonalize_mlp_adapters(model)

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if self.freq == "epoch":
            orthogonalize_mlp_adapters(model)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if self.freq == "step":
            orthogonalize_mlp_adapters(model)
            return
        if isinstance(self.freq, int) and self.freq > 0:
            if state.global_step % self.freq == 0:
                orthogonalize_mlp_adapters(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--bs", type=int, default=8)

    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--submodules", action="store_true")

    parser.add_argument("--lora-rank", type=int, default=8, help="Rank (r) for LoRA adapters.")
    parser.add_argument("--adapter-rank", type=int, default=8, help="Rank of adapter modules (if using adapters)")
    parser.add_argument("--linear", action ='store_true', help="If set, disable nonlinearity in adapters")
    parser.add_argument("--nonlinearity", type=str, default="silu")
    parser.add_argument("--lora-lr", type=float, default=None, help="Override learning rate for LoRA params")
    parser.add_argument("--adapter-lr", type=float, default=None, help="Override learning rate for adapter params")
    parser.add_argument(
        "--orthogonal-freq",
        type=str,
        default="100",
        help="step, epoch, init, or integer steps (e.g. 1000)",
    )

    args = parser.parse_args()
    dataset_name = args.dataset_name
    default_epochs = epochs_per_dataset.get(dataset_name, 5)
    train_epochs = args.epochs if args.epochs is not None else default_epochs
    dataset_class = dataset_dict[dataset_name]

    model_name = args.model_name
    
    if "gemma" in model_name.lower():
        base_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to("cuda")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    lora_config = LoraConfig(
        r=args.lora_rank,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_config).to("cuda")
    

    steered_model = AdapterSteerer(
        model,
        layers_to_steer="all",
        targets=("mlp", "attn") if args.submodules else ("block",),
        apply_to="all",
        rank=args.adapter_rank if hasattr(args, 'adapter_rank') else 8,
        activation=None if args.linear else args.nonlinearity,
    )
    base_total_params = sum(p.numel() for p in base_model.parameters())
    lora_trainable_params = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and "lora_" in name
    )
    adapter_trainable_params = sum(
        p.numel() for p in steered_model.adapters.parameters() if p.requires_grad
    )
    # if base_total_params > 0:
    lora_pct = 100 * lora_trainable_params / base_total_params
    adapter_pct = 100 * adapter_trainable_params / base_total_params
    print(f"LoRA trainable params: {lora_trainable_params} ({lora_pct:.4f}% of base)")
    print(f"Adapter trainable params: {adapter_trainable_params} ({adapter_pct:.4f}% of base)")

    if "gemma" in model_name.lower():
        train_dataset, eval_dataset = dataset_class(mode="train").load_tokenized_dataset_chat(tokenizer=tokenizer)
    else:
        train_dataset, eval_dataset = dataset_class(mode="train").load_tokenized_dataset(tokenizer=tokenizer)

    sft_arguments = TrainingArguments(
        output_dir=f'model_{dataset_name}/{model_name.split("/")[-1]}_dual',
        # overwrite_output_dir=True,
        learning_rate=args.lr,
        num_train_epochs=train_epochs,
        per_device_train_batch_size=args.bs,       # Lower batch size to reduce memory usage.
        gradient_accumulation_steps=2,         # Accumulate gradients to effectively use a batch size of 2.
        per_device_eval_batch_size=1,
        save_total_limit=1,
        logging_steps=10,
        logging_first_step=True,
        # report_to=["wandb"],
        remove_unused_columns=False,
        warmup_ratio=args.warmup_ratio if hasattr(args, 'warmup_ratio') else 0.03,
        lr_scheduler_type=args.scheduler if hasattr(args, 'scheduler') else "cosine",
        weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.01,
        group_by_length=True,
        save_strategy="no",       
    )
    if args.orthogonal_freq.isdigit():
        orthogonal_freq = int(args.orthogonal_freq)
    else:
        orthogonal_freq = args.orthogonal_freq
    trainer = Trainer(
        model=steered_model,
        args=sft_arguments,
        train_dataset=train_dataset,
        callbacks=[OrthogonalizeCallback(orthogonal_freq)],
    )
    assert any(p.requires_grad for p in trainer.model.parameters()), "No trainable params!"

    print("Percentage trainable params:", sum(p.numel() for p in steered_model.parameters() if p.requires_grad)/sum(p.numel() for p in model.parameters())*100, "%")
    assert any(p.requires_grad for p in trainer.model.parameters())
    model.config.use_cache = False
    trainer.train()

    dataset_obj = dataset_class(mode="eval")
    if "gemma" in model_name.lower():
        datasets = dataset_obj.load_raw_dataset_chat(tokenizer, return_test=True) # @TODO: add chat support for gemma
    else:
        datasets = dataset_obj.load_raw_dataset(return_test=True) # @TODO: add chat support for gemma
    validation_dataset = datasets["validation"]
    test_dataset = datasets["test"]
    eval_fn = dataset_obj.evaluate
    s = 'submodules' if args.submodules else 'block'
    project_name = f'orth_outputs/{dataset_name}/{model_name.split("/")[-1]}_dual_lora{args.lora_rank}_adapter{args.adapter_rank}_lr{args.lora_lr}_{s}'

    if args.save_model:
        final_dir = f'orth_model_{dataset_name}/{model_name.split("/")[-1]}_dual_lora{args.lora_rank}_adapter{args.adapter_rank}_lr{args.lr}_epochs{train_epochs}'
        # Save LoRA-wrapped base model and adapter steerer weights separately.
        trainer.model.base_model.save_pretrained(final_dir)
        adapter_path = os.path.join(final_dir, "adapter_steerer.pt")
        torch.save(trainer.model.adapters.state_dict(), adapter_path)
        print(f"Saved LoRA model to {final_dir} and adapter steerer to {adapter_path}")
    if not args.test_only:
        output_file = os.path.join("outputs", project_name, "validation.json")
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        print("=" * 20, "VALIDATION SET", "=" * 20)
        print("\n")
        evaluate_model(validation_dataset, eval_fn, steered_model, tokenizer, output_file, max_new_tokens=max_new_tokens_per_dataset.get(dataset_name, 100))
    output_file = os.path.join("outputs", project_name, "test.json")
    print("=" * 20, "TEST SET", "=" * 20)
    print("\n")
    evaluate_model(test_dataset, eval_fn, steered_model, tokenizer, output_file, max_new_tokens=max_new_tokens_per_dataset.get(dataset_name, 100))
    
    print("=" * 20)
    print("\n")

    
