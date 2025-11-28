import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import numpy as np

from steerer import FixedVectorSteerer, AdapterSteerer
import sys
sys.path.append('./')
from utils import *

def evaluate_model(evaluation_dataset, eval_fn, steered_model, tokenizer, output_file, max_new_tokens=100):
    assert eval_fn is not None, "eval_fn must be provided"
    assert output_file is not None, "output_file must be provided"

    prompts = [evaluation_dataset[i][0] for i in range(len(evaluation_dataset))]
    continuations = [evaluation_dataset[i][1] for i in range(len(evaluation_dataset))]
    labels = [evaluation_dataset[i][2] for i in range(len(evaluation_dataset))]

    eval_all = []
    outputs_all = []
    
    for i in tqdm(range(len(prompts))):
        inputs = tokenizer(prompts[i], 
            return_tensors="pt",
            add_special_tokens=True,
            padding=False,
            truncation=False).to("cuda")
        collected = []
        with torch.no_grad():
            out_ = steered_model.generate(**inputs, 
                        max_new_tokens=max_new_tokens, 
                        do_sample=False,
                        )
        generated_text = tokenizer.decode(out_[0], skip_special_tokens=True).split(prompts[i])[-1].strip()

        eval_result = eval_fn(generated_text, labels[i])
        eval_all.append(eval_result)
        outputs_all.append({
            'prompt': prompts[i], 
            'output': generated_text,
            "label": labels[i],
            "eval": eval_result
        })
    log_output(outputs_all, output_file)
    print("Evaluation completed. Results saved to", output_file)
    print("Accuracy = {:.3f}".format(np.mean(eval_all)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output-file", type=str, required=False)
    parser.add_argument("--no-adapter", action ='store_true', help="If set, train adapter modules instead of fixed vectors")
    parser.add_argument("--adapter-rank", type=int, default=8, help="Rank of adapter modules (if using adapters)")
    parser.add_argument("--intervene-last", action ='store_true', help="If set, intervene on ALL tokens, not just last token")
    parser.add_argument("--submodules", action ='store_true', help="If set, adapters a submodules")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model_name
    dataset_name = args.dataset_name


    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    layers_to_steer = "all"
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
        )

    # 4. Load checkpoint state dict
    ckpt_bin = os.path.join(args.ckpt_path, "pytorch_model.bin")
    if not os.path.exists(ckpt_bin):
        # if you used safetensors instead:
        from safetensors.torch import load_file
        ckpt_bin = os.path.join(args.ckpt_path, "model.safetensors")
        state_dict = load_file(ckpt_bin)
    else:
        state_dict = torch.load(ckpt_bin, map_location="cpu")

    ckpt_bin = os.path.join(args.ckpt_path, "pytorch_model.bin")
    if os.path.exists(ckpt_bin):
        state_dict = torch.load(ckpt_bin, map_location="cpu")
    else:
        # safetensors fallback
        from safetensors.torch import load_file
        ckpt_bin = os.path.join(args.ckpt_path, "model.safetensors")
        state_dict = load_file(ckpt_bin)

    missing, unexpected = steered_model.load_state_dict(state_dict, strict=False)

    steered_model.to(device)
    steered_model.eval()

    dataset_class = dataset_dict[dataset_name](mode="eval")
    if "gemma" in model_name.lower():
        datasets = dataset_class.load_raw_dataset_chat(tokenizer=tokenizer, return_test=True)
    else:
        datasets = dataset_class.load_raw_dataset(return_test=True) # @TODO: add chat support for gemma
    evaluation_dataset = datasets[args.split]

    if args.no_adapter:
        project_str = "fixedvec_all" if not args.intervene_last else "fixedvec_last"
    else:
        project_str = "adapter_all" if not args.intervene_last else "adapter_last"

    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(project_str, dataset_name, "/".join(args.ckpt_path.split('/')[:-1]), f"inference_{args.split}.json")
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    eval_fn = dataset_class.evaluate
    evaluate_model(evaluation_dataset, eval_fn, steered_model, tokenizer, output_file, max_new_tokens=max_new_tokens_per_dataset.get(dataset_name, 100))
    

if __name__ == "__main__":
    main()
