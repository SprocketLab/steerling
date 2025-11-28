from .dataloaders import BaseDatasetLoader
import re

from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import numpy as np

class ListOps(BaseDatasetLoader):
    def __init__(self, mode, ops_len=64):
        self.mode = mode
        assert ops_len in [32, 64, 128]
        self.raw_dict = load_dataset(f"fengyang0317/listops-{str(ops_len)}")

    def build_prompt(self, ex):
        LISTOPS_PROMPT_TMPL = (
            "Evaluate the value of the following nested list expression.\n"
            "Return only the final numeric result inside <answer>…</answer>.\n\n"
            "Expression:\n{expression}\n"
        )
        return LISTOPS_PROMPT_TMPL.format(
            expression=ex["Source"]
        )
    
    def get_answer_string(self, ex):
        answer_key = str(ex["Target"])
        return f"<answer>{answer_key}</answer>"
    
    def load_tokenized_dataset(self, tokenizer, max_len: int = 512, return_test: bool = False):
        def build_tokenized_features(ex):
            prompt = self.build_prompt(ex)
            completion = self.get_answer_string(ex)
            return self._tokenize_prompt_completion(tokenizer, prompt, completion, max_len)
        return self._map_tokenized_splits(build_tokenized_features, return_test)
    
    def load_raw_dataset(self, return_test: bool = False):
        def build_raw_features(ex):
            prompt = self.build_prompt(ex)
            answer_key = str(ex["Target"])
            completion = self.get_answer_string(ex)
            return self._build_raw_example(prompt, completion, answer_key)
        
        return self._map_raw_splits(build_raw_features, return_test)

    def evaluate(self, output, label):
        if "You are a helpful assistant." in output:
            output = output.split("\nmodel\n")[-1].strip().rstrip()
        m = re.search(r"<answer>(.*?)</answer>", (output or "").strip(), flags=re.S|re.I)

        if not m:
            return False
        try:
            return int(m.group(1).strip()) == int(label)
        except:
            return False

    def load_raw_dataset_chat(self, tokenizer, return_test=False):
        def build_raw_features(ex):
            prompt  = self.build_prompt(ex)
            completion = self.get_answer_string(ex)

            # Base conversation (system + user)
            base_msgs = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user",   "content": [{"type": "text", "text": prompt}]},
            ]

            # Render only the prefix (system+user) and add a model turn prompt
            prefix = tokenizer.apply_chat_template(
                base_msgs, tokenize=False, add_generation_prompt=True
            )

            # Completion is the assistant response we want to learn
            return [prefix, completion, ex.get("Target")]

        return self._map_raw_chat_splits(build_raw_features, return_test) 
    
    def load_tokenized_dataset_chat(self, tokenizer, max_len=256, return_test=False):
        def build_tokenized_features(ex):
            prompt  = self.build_prompt(ex)
            completion = self.get_answer_string(ex)

            # Common messages (no assistant content)
            base_msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ]

            return self._tokenize_chat_completion(tokenizer, base_msgs, completion, max_len)

        return self._map_tokenized_splits(build_tokenized_features, return_test)
