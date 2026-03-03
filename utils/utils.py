import json
import os
import re

import numpy as np
import torch
from tqdm import tqdm

from .dataloaders import *
from .dataloaders_it import *
from .dataloaders_LRA import *

dataset_dict = {
    "ASDiv": ASDiv,
    "BoolQ": BoolQ,
    "Winogrande": Winogrande,
    "ListOps": ListOps,
    "GSM8K": GSM8K,
    "MNLI": MNLI,
    "SVAMP": SVAMP,
    "MAWPS": MAWPS,
    "ARC": ARC_Challenge,
    "AQuA": AQuA

}

epochs_per_dataset = {
    "BoolQ": 1,
    "Winogrande": 1,
    "ListOps": 1,
    "GSM8K": 3,
    "MAWPS": 3,
    "AQuA": 1
}

max_new_tokens_per_dataset = {
    "BoolQ": 10,
    "Winogrande": 5,
    "ListOps": 10,
    "GSM8K": 100,
    "MAWPS": 20,
    "AQuA": 256
}


def log_output(answers, output_file):
    directory_path = os.path.dirname(output_file)
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
    ans_file = open(os.path.join(output_file), "w")
    for answer in answers:
        ans_file.write(json.dumps(answer) + "\n")
        ans_file.flush()
    ans_file.close()


def get_lm_head_output(inputs, model, n_steps):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    modules_to_peek = ["lm_head"]

    seq = input_ids.clone()
    attn = attention_mask.clone()
    collected = []  # list of [B, V] per step

    for step in range(n_steps):
        with torch.no_grad():
            out = model.generate(
                inputs=seq.to("cuda"),
                attention_mask=attn.to("cuda"),
                do_sample=False,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,  # <-- get per-step logits
            )
        scores = torch.stack(out.scores, dim=1)
        last_logits = scores
        collected.append(last_logits.detach().cpu())

        # Greedy (deterministic) next token
        next_tok = last_logits.argmax(dim=-1)  # [B]
        seq = torch.cat([seq.to("cuda"), next_tok], dim=1)
        attn = torch.cat(
            [
                attn.to("cuda"),
                torch.ones((attn.size(0), 1), dtype=model.dtype, device=model.device),
            ],
            dim=1,
        )
    return torch.vstack(collected).squeeze()


def evaluate_model(
    evaluation_dataset, eval_fn, model, tokenizer, output_file, max_new_tokens=100
):
    assert eval_fn is not None, "eval_fn must be provided"
    assert output_file is not None, "output_file must be provided"

    prompts = [evaluation_dataset[i][0] for i in range(len(evaluation_dataset))]
    continuations = [evaluation_dataset[i][1] for i in range(len(evaluation_dataset))]
    labels = [evaluation_dataset[i][2] for i in range(len(evaluation_dataset))]

    eval_all = []
    outputs_all = []

    for i in tqdm(range(len(prompts))):
        inputs = tokenizer(
            prompts[i],
            return_tensors="pt",
            add_special_tokens=True,
            padding=False,
            truncation=False,
        ).to("cuda")
        out_ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated_text = tokenizer.decode(out_[0], skip_special_tokens=True).split(
            prompts[i]
        )[-1]
        eval_result = eval_fn(generated_text, labels[i])
        eval_all.append(eval_result)
        outputs_all.append(
            {
                "prompt": prompts[i],
                "output": generated_text,
                "label": labels[i],
                "eval": eval_result,
            }
        )
    total_accuracy = np.mean(eval_all)
    outputs_all.insert(0, {"total_accuracy": total_accuracy})
    log_output(outputs_all, output_file)
    print("Evaluation completed. Results saved to", output_file)
    print("Accuracy = {:.3f}".format(total_accuracy))
    return total_accuracy


def _parse_numeric_answer(text):
    if not text:
        return None

    num_pat = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")
    s = text.strip()
    matches = list(re.finditer(r"(?i)answer\s*:\s*(.+)", s))
    if matches:
        ans_raw = matches[-1].group(1).strip()
        mnum = num_pat.search(ans_raw)
        if mnum:
            try:
                return float(mnum.group(0).replace(",", ""))
            except ValueError:
                return None

    nums = list(num_pat.finditer(s))
    if nums:
        try:
            return float(nums[-1].group(0).replace(",", ""))
        except ValueError:
            return None

    return None


def parse_eval_results_jsonl(json_path):
    total_accuracy = None
    rows = []

    with open(json_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "total_accuracy" in record:
                total_accuracy = record["total_accuracy"]
                continue
            prediction = _parse_numeric_answer(record.get("output", ""))
            rows.append({**record, "prediction": prediction})

    return {"total_accuracy": total_accuracy, "rows": rows}