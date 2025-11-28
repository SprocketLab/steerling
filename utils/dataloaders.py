import re
import math

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import numpy as np

@dataclass
class SFTDataCollator:
    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100
    padding: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].size(1)
        label_tensor = torch.full((len(labels), max_len), self.label_pad_token_id, dtype=torch.long)
        for i, label in enumerate(labels):
            label_tensor[i, :len(label)] = torch.tensor(label, dtype=torch.long)
        batch["labels"] = label_tensor
        return batch

class BaseDatasetLoader(ABC):
    """Abstract interface that keeps dataset loaders interchangeable."""

    @abstractmethod
    def __init__(self, mode: str = "eval", **kwargs):
        assert mode in ["train", "eval"], "Mode must be 'train' or 'eval'"
        self.mode = mode

    @abstractmethod
    def load_tokenized_dataset_chat(self, tokenizer, max_len: int = 256, return_test: bool = False, **kwargs):
        """Return chat-style tokenized dataset splits."""
        raise NotImplementedError

    @abstractmethod
    def load_tokenized_dataset(self, tokenizer, max_len: int = 256, return_test: bool = False, **kwargs):
        """Return plain tokenized dataset splits."""
        raise NotImplementedError

    @abstractmethod
    def load_raw_dataset_chat(self, tokenizer, return_test: bool = False, **kwargs):
        """Return raw chat-formatted dataset splits."""
        raise NotImplementedError

    @abstractmethod
    def load_raw_dataset(self, return_test: bool = False, **kwargs):
        """Return raw prompt/completion pairs."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, output, label) -> bool:
        """Evaluate a model output against the reference label."""
        raise NotImplementedError

    # --- Shared helpers for plain prompt/completion datasets ---
    def _tokenize_prompt_completion(self, tokenizer, prompt: str, completion: str, max_len: int):
        """Tokenize prompt+completion pairs with prefix masking for training."""
        text_full = prompt + completion
        text_prefix = prompt  # mask this part

        if self.mode == "train":
            full = tokenizer(
                text_full,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = full["input_ids"].squeeze().tolist()
            attention_mask = full["attention_mask"].squeeze().tolist()

            prefix_ids = tokenizer(
                text_prefix,
                truncation=True,
                max_length=max_len,
            )["input_ids"]

            labels = input_ids.copy()
            cut = min(len(labels), len(prefix_ids))
            pad_id = tokenizer.pad_token_id
            labels = [(-100 if (i < cut or tok == pad_id) else tok) for i, tok in enumerate(input_ids)]
        else:
            full = tokenizer(
                text_prefix,
                truncation=False,
                padding=False,
                return_tensors="pt",
            )
            input_ids = full["input_ids"].squeeze().tolist()
            attention_mask = full["attention_mask"].squeeze().tolist()
            labels = input_ids.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _build_raw_example(self, prompt: str, completion: str, label_value):
        """Return the common [prompt, completion, label] structure."""
        return [prompt, completion, label_value]

    def _map_tokenized_splits(self, build_fn, return_test: bool):
        """Map a tokenization fn over train/val/test with common column removal."""
        train_ds = self.raw_dict["train"]
        valid_ds = self.raw_dict["validation"]
        test_ds  = self.raw_dict["test"]

        cols_to_remove = train_ds.column_names
        train_ds = train_ds.map(build_fn, remove_columns=cols_to_remove)
        valid_ds = valid_ds.map(build_fn, remove_columns=cols_to_remove)
        test_ds  = test_ds.map(build_fn,  remove_columns=cols_to_remove)

        return (train_ds, valid_ds, test_ds) if return_test else (train_ds, valid_ds)

    def _map_raw_chat_splits(self, build_fn, return_test: bool):
        """Map a chat raw-builder across splits and package dict output."""
        train_data = [build_fn(ex) for ex in self.raw_dict["train"]]
        eval_data  = [build_fn(ex) for ex in self.raw_dict["validation"]]
        if return_test:
            test_data = [build_fn(ex) for ex in self.raw_dict["test"]]
            return {"train": train_data, "validation": eval_data, "test": test_data}
        return {"train": train_data, "validation": eval_data}

    def _map_raw_splits(self, build_fn, return_test: bool):
        """Map a raw-builder across splits and package dict output."""
        train_data = [build_fn(ex) for ex in self.raw_dict["train"]]
        eval_data  = [build_fn(ex) for ex in self.raw_dict["validation"]]
        if return_test:
            test_data = [build_fn(ex) for ex in self.raw_dict["test"]]
            return {"train": train_data, "validation": eval_data, "test": test_data}
        return {"train": train_data, "validation": eval_data}

    def _tokenize_chat_completion(self, tokenizer, base_msgs, completion: str, max_len: int):
        """Tokenize chat messages, masking the prefix and supervising assistant completion."""
        if self.mode == "train":
            # Include assistant content for teacher forcing
            messages = base_msgs + [{"role": "assistant", "content": completion}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False)

            # Prefix without assistant content; add generation prompt for masking
            prefix_rendered = tokenizer.apply_chat_template(
                base_msgs, tokenize=False, add_generation_prompt=True
            )

            full = tokenizer(
                rendered,
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )

            input_ids      = full["input_ids"]
            attention_mask = full["attention_mask"]

            prefix_ids = tokenizer(
                prefix_rendered, truncation=True, max_length=max_len
            )["input_ids"]

            labels = input_ids.copy()
            cut    = min(len(labels), len(prefix_ids))
            pad_id = tokenizer.pad_token_id
            labels = [(-100 if (i < cut or tok == pad_id) else tok)
                    for i, tok in enumerate(input_ids)]
        else:
            # Eval: no assistant content; add generation prompt
            rendered = tokenizer.apply_chat_template(
                base_msgs, tokenize=False, add_generation_prompt=True
            )
            full = tokenizer(
                rendered,
                truncation=False,
                padding=False,
                return_attention_mask=True,
            )
            input_ids      = full["input_ids"]
            attention_mask = full["attention_mask"]
            labels = [-100] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    

# class StrategyQA(BaseDatasetLoader):
#     def __init__(self, mode='eval'):
#         super().__init__(mode)
#         ds = load_dataset("ChilleD/StrategyQA")
#         tmp = ds["test"].shuffle(seed=42).train_test_split(test_size=0.50, seed=42) # 50 val, 50 test
#         self.raw_dict = {
#             "train": ds["train"],
#             "validation": tmp["train"],
#             "test": tmp["test"],
#         }
    
#     def build_prompt(self, ex):
#         STRATEGY_QA_PROMPT_TMPL = (
#             "Answer the question with a true/false based on the given passage.\n"
#             "Passage: {passage}\n"
#             "Question: {question}\n"
#         )
#         return STRATEGY_QA_PROMPT_TMPL.format(
#             passage=ex['facts'],
#             question=ex['question']
#         )
    
#     def get_answer_string(self, ex):
#         answer_key = str(ex["answer"])
#         return f"Answer: {answer_key}."
    
#     def load_tokenized_dataset(self, tokenizer, max_len: int = 256, return_test: bool = False):
#         def build_tokenized_features(ex):
#             prompt = self.build_prompt(ex)
#             completion = self.get_answer_string(ex)
#             return self._tokenize_prompt_completion(tokenizer, prompt, completion, max_len)
#         return self._map_tokenized_splits(build_tokenized_features, return_test)
    
#     def load_raw_dataset(self, return_test: bool = False):
#         def build_raw_features(ex):
#             prompt = self.build_prompt(ex)
#             answer_key = str(ex["answer"])
#             completion = self.get_answer_string(ex)
#             return self._build_raw_example(prompt, completion, answer_key)
        
#         return self._map_raw_splits(build_raw_features, return_test)

#     def evaluate(self, output, label):
#         s = output.strip().lower()
#         answer_key = str(label).strip().lower()

#         # Try to extract from "Answer: (C)" or similar
#         pattern = r"(?i)answer\s*:\s*(true|false)"
#         m_ans = re.search(pattern, s)
#         if m_ans:
#             return m_ans.group(1).lower() == answer_key
#         return False
    
#     def load_raw_dataset_chat(self, tokenizer, return_test=False):
#         pass
    
#     def load_tokenized_dataset_chat(self, tokenizer, max_len=256, return_test=False):
#         pass

class BoolQ(BaseDatasetLoader):
    def __init__(self, mode='eval'):
        super().__init__(mode)
        ds = load_dataset("google/boolq")
        # Option A: 80/10/10["test"].shuffle(seed=42)
        tmp = ds["validation"].shuffle(seed=42).train_test_split(test_size=0.20, seed=42) # 80 val, 20 test
        self.raw_dict = {
            "train": ds["train"],
            "validation": tmp["train"],
            "test": tmp["test"],
        }
    
    def build_prompt(self, ex):
        BOOLQ_PROMPT_TMPL = (
            "Answer the question with a true/false based on the given passage.\n"
            "Passage: {passage}\n"
            "Question: {question}\n"
        )
        return BOOLQ_PROMPT_TMPL.format(
            passage=ex['passage'],
            question=ex['question']
        )
    
    def get_answer_string(self, ex):
        answer_key = str(ex["answer"])
        return f"Answer: {answer_key}."
    
    def load_tokenized_dataset(self, tokenizer, max_len: int = 256, return_test: bool = False):
        def build_tokenized_features(ex):
            prompt = self.build_prompt(ex)
            completion = self.get_answer_string(ex)
            return self._tokenize_prompt_completion(tokenizer, prompt, completion, max_len)
        return self._map_tokenized_splits(build_tokenized_features, return_test)
            
    
    def load_raw_dataset(self, return_test: bool = False):
        def build_raw_features(ex):
            prompt = self.build_prompt(ex)
            answer_key = str(ex["answer"])
            completion = self.get_answer_string(ex)
            return self._build_raw_example(prompt, completion, answer_key)
        
        return self._map_raw_splits(build_raw_features, return_test)

    def evaluate(self, output, label):
        s = output.strip().lower()
        answer_key = str(label).strip().lower()

        # 1. Try the strict format: "Answer: true/false"
        pattern_full = r"answer\s*:\s*(true|false)"
        m = re.search(pattern_full, s)
        if m:
            return m.group(1) == answer_key

        # 2. Fallback: find the FIRST standalone true/false
        #    \b ensures we don't match "truth" or "falsehood"
        pattern_simple = r"\b(true|false)\b"
        m2 = re.search(pattern_simple, s)
        if m2:
            return m2.group(1) == answer_key

        # 3. No recognizable answer found
        return False

    
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
            return [prefix, completion, str(ex.get("answer"))]

        return self._map_raw_chat_splits(build_raw_features, return_test) 

# class QASC(BaseDatasetLoader):
#     def __init__(self, mode = 'eval'):
#         super().__init__(mode)
#         ds = load_dataset("allenai/qasc")
#         # Option A: 80/10/10["test"].shuffle(seed=42)
#         tmp = ds["validation"].shuffle(seed=42).train_test_split(test_size=0.20, seed=42) # 50 val, 50 test
#         self.raw_dict = {
#             "train": ds["train"],
#             "validation": tmp["train"],
#             "test": tmp["test"],
#         }
    
#     def build_prompt(self, ex):
#         QASC_PROMPT_TMPL = (
#             "Answer the multiple-choice question using the provided facts.\n"
#             "Fact 1: {fact1}\n"
#             "Fact 2: {fact2}\n"
#             "Question: {formatted_question}\n"
#         )
#         return QASC_PROMPT_TMPL.format(
#             fact1=ex['fact1'],
#             fact2=ex['fact2'],
#             formatted_question=ex['formatted_question']
#         )
    
    
#     def get_answer_string(self, ex):
#         answer_key = str(ex["answerKey"])
#         choices = ex["choices"]
#         choices_text = choices["text"]
#         choices_label = choices["label"]
#         choices_label = np.array([str(l) for l in choices_label])

#         answer_index = np.argwhere(choices_label == answer_key).flatten()[0]
#         answer_text = choices_text[answer_index]
#         return f"Answer: ({answer_key}) {answer_text}."

#     def load_tokenized_dataset(self, tokenizer, max_len: int = 256, return_test: bool = False):
#         def build_tokenized_features(ex):
#             prompt = self.build_prompt(ex)
#             completion = self.get_answer_string(ex)
#             return self._tokenize_prompt_completion(tokenizer, prompt, completion, max_len)
#         return self._map_tokenized_splits(build_tokenized_features, return_test)
            
    
#     def load_raw_dataset(self, return_test: bool = False):
#         def build_raw_features(ex):
#             prompt = self.build_prompt(ex)
#             answer_key = str(ex["answerKey"])
#             completion = self.get_answer_string(ex)
#             return self._build_raw_example(prompt, completion, answer_key)
        
#         return self._map_raw_splits(build_raw_features, return_test)

#     def evaluate(self, output, label):
#         s = output.strip().upper()
#         answer_key = str(label).strip().upper()

#         # Try to extract from "Answer: (C)" or similar
#         m_ans = re.search(r"ANSWER\s*[:\-]?\s*\(?([A-H])\)?", s)
#         if m_ans:
#             return m_ans.group(1).upper() == answer_key

#         # As a fallback, match any standalone letter A–H in the output
#         letters = re.findall(r"\b([A-H])\b", s)
#         if len(letters) == 1:
#             return letters[0] == answer_key

#         return False
    
#     def load_raw_dataset_chat(self, tokenizer, return_test=False):
#         pass
    
#     def load_tokenized_dataset_chat(self, tokenizer, max_len=256, return_test=False):
#         pass

class ASDiv(BaseDatasetLoader):
    def __init__(self, mode="eval"):
        super().__init__(mode)
        ds = load_dataset("MU-NLPC/Calc-asdiv_a")["test"].shuffle(seed=42) #dataset only has a single split
        # Option A: 80/10/10
        tmp = ds.train_test_split(test_size=0.20, seed=42)           # 80 train, 20 temp
        val_test = tmp["test"].train_test_split(test_size=0.5, seed=42)  # 10 val, 10 test
        self.raw_dict = {
            "train": tmp["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    
    def norm_trace(self, xml: str):
        # 1) <gadget id="calc">7 + 2</gadget> -> <call tool="calc">7+2</call>
        def _call_sub(m):
            tool = m.group(1).strip()
            expr = re.sub(r"\s+", "", m.group(2))  # "7 + 2" -> "7+2"
            return f'<call tool="{tool}">{expr}</call>'
        xml = re.sub(r'<gadget\s+id="([^"]+)">(.*?)</gadget>', _call_sub, xml, flags=re.S)

        # 2) Prefer <result>…</result>, else <output>…</output>
        result = None
        m_res = re.search(r"<result>(.*?)</result>", xml, flags=re.S)
        m_out = re.search(r"<output>(.*?)</output>", xml, flags=re.S)
        if m_res: result = m_res.group(1).strip()
        elif m_out: result = m_out.group(1).strip()
        else: result = ""  # fallback if missing

        # Normalize tool_output to exactly one tag
        out_val = m_out.group(1).strip() if m_out else result
        xml = re.sub(r"<output>.*?</output>", "", xml, flags=re.S)
        xml = re.sub(r"<result>.*?</result>", "", xml, flags=re.S)

        pieces = []
        # keep any <call ...> we transformed
        for call in re.findall(r'<call tool="[^"]+">.*?</call>', xml):
            pieces.append(call)
        # add single tool_output and final
        if out_val != "":
            pieces.append(f"<tool_output>{out_val}</tool_output>")
        pieces.append(f"<final>{result}</final>")
        return "".join(pieces)
    
    def build_prompt(self, problem: str):
        CANONICAL_PROMPT_PREFIX = (
            "Solve the problem using this format:\n"
            "<call tool=\"calculator\">EXPRESSION</call>\n"
            "<tool_output>RESULT</tool_output>\n"
            "<final>FINAL_ANSWER</final>\n\n"
            "Problem: "
        )
        return f"{CANONICAL_PROMPT_PREFIX}{problem.strip()}\n"
    
    def load_tokenized_dataset_chat(self, tokenizer, max_len=256, return_test=False):
        def build_tokenized_features(ex):
            problem = ex["question"]
            prompt  = self.build_prompt(problem)
            chain   = ex["chain"]
            completion = self.norm_trace(chain)

            # Common messages (no assistant content)
            base_msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ]

            return self._tokenize_chat_completion(tokenizer, base_msgs, completion, max_len)

        return self._map_tokenized_splits(build_tokenized_features, return_test)


    def load_tokenized_dataset(self, tokenizer, max_len=256, return_test=False):
        def build_tokenized_features(ex):
            problem = ex['question']
            prompt = self.build_prompt(problem)

            chain = ex['chain']
            completion = self.norm_trace(chain)

            return self._tokenize_prompt_completion(tokenizer, prompt, completion, max_len)

        return self._map_tokenized_splits(build_tokenized_features, return_test)
    
    def load_raw_dataset_chat(self, tokenizer, return_test=False):
        def build_raw_features(ex):
            problem = ex["question"]
            prompt  = self.build_prompt(problem)
            chain   = ex["chain"]
            completion = self.norm_trace(chain)

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
            return [prefix, completion, ex.get("result_float")]

        return self._map_raw_chat_splits(build_raw_features, return_test)

    
    def load_raw_dataset(self, return_test=False):
        def build_raw_features(ex):
            problem = ex['question']
            prompt = self.build_prompt(problem)

            chain = ex['chain']
            completion = self.norm_trace(chain)

            return [prompt, completion, ex['result_float']]
        
        return self._map_raw_splits(build_raw_features, return_test)
    
    def evaluate_logic(self, output, label):
        segment = output.split("<final>FINAL_ANSWER</final>")[-1].strip().rstrip()
        calls = re.findall(r'<call tool="calculator">(.*?)</call>', segment, flags=re.S)
        if not calls:
            return False

        def _clean_num_str(s: str) -> str:
            s = (s or "").strip()
            s = s.replace(",", "")
            s = re.sub(r"^\s*[\[\(]\s*(.*?)\s*[\]\)]\s*$", r"\1", s)
            return s.strip()

        try:
            label_val = float(_clean_num_str(str(label)))
        except Exception:
            return False

        def _format_value_tokens(val: float):
            if not math.isfinite(val):
                return set()
            tokens = {f"{val}", f"{val:.15g}"}
            if abs(val - round(val)) <= 1e-9:
                tokens.add(str(int(round(val))))
            trimmed = set()
            for token in tokens:
                trimmed.add(re.sub(r"\.0+$", "", token))
            return {tok for tok in trimmed if tok}

        def _prepare_expr(expr: str, prev_val: Optional[float]):
            expr = (expr or "").strip()
            expr = expr.replace(",", "")
            expr = expr.replace("^", "**")
            expr = expr.replace("×", "*").replace("·", "*")
            expr = expr.replace("÷", "/")
            expr = expr.replace("−", "-").replace("–", "-").replace("—", "-")

            if prev_val is not None:
                replacement = f"({prev_val})"
                for token in ("ans", "answer", "prev", "previous", "result", "res"):
                    expr = re.sub(rf"\b{token}\b", replacement, expr, flags=re.I)

            expr = re.sub(r"\s+", "", expr)
            if "=" in expr:
                expr = expr.split("=")[-1]
            return expr

        def _safe_eval(expr: str):
            if not expr:
                return None
            if not re.fullmatch(r"[0-9.\+\-*/()]*", expr):
                return None
            try:
                return float(eval(expr, {"__builtins__": None}, {}))
            except Exception:
                return None

        def _is_chainable(raw_expr: str, prev_val: Optional[float]):
            if prev_val is None:
                return False
            if re.search(r"\b(ans|answer|prev|previous|result|res)\b", raw_expr, flags=re.I):
                return True
            compact = re.sub(r"\s+", "", raw_expr)
            for token in _format_value_tokens(prev_val):
                if token and token in compact:
                    return True
            return False

        prev_val = None
        evaluated_values: List[float] = []
        chain_flags: List[bool] = []

        for idx, raw_expr in enumerate(calls):
            expr_to_eval = _prepare_expr(raw_expr, prev_val if idx > 0 else None)
            val = _safe_eval(expr_to_eval)
            if val is None or not math.isfinite(val):
                return False
            evaluated_values.append(val)
            if idx > 0:
                chain_flags.append(_is_chainable(raw_expr, prev_val))
            prev_val = val

        if not evaluated_values:
            return False

        if len(evaluated_values) == 1:
            final_val = evaluated_values[0]
        else:
            if chain_flags and all(chain_flags):
                final_val = evaluated_values[-1]
            else:
                last_expr_direct = _prepare_expr(calls[-1], None)
                last_val = _safe_eval(last_expr_direct)
                if last_val is None or not math.isfinite(last_val):
                    final_val = evaluated_values[-1]
                else:
                    final_val = last_val

        if not math.isfinite(final_val) or not math.isfinite(label_val):
            return False
        return math.isclose(final_val, label_val, rel_tol=1e-9, abs_tol=1e-9)


    def evaluate(self, output, label):
        output = output.split("<final>FINAL_ANSWER</final>")[-1].strip().rstrip()

        def _clean_num_str(s: str) -> str:
            s = s.strip()
            s = s.replace(",", "")
            # strip one layer of surrounding [] or ()
            s = re.sub(r"^\s*[\[\(]\s*(.*?)\s*[\]\)]\s*$", r"\1", s)
            return s.strip()
        
        s = (output or "").strip()
        candidates = []

        def _add_candidate(val: str):
            if not val:
                return
            cleaned = val.strip()
            if not cleaned:
                return
            if cleaned not in candidates:
                candidates.append(cleaned)

        def _normalize_for_compare(val: str):
            cleaned = _clean_num_str(val)
            if not cleaned:
                return ""
            try:
                return float(cleaned)
            except ValueError:
                return cleaned

        # Collect answers from canonical tags if present
        final_values = []
        tool_values = []
        for tag in ("final", "tool_output"):
            for m in re.finditer(rf"<{tag}>(.*?)</{tag}>", s, flags=re.S):
                value = m.group(1)
                if tag == "final":
                    final_values.append(value)
                else:
                    tool_values.append(value)

        for val in final_values:
            _add_candidate(val)

        add_tool_outputs = True
        if final_values and tool_values:
            final_norms = {
                norm for norm in (_normalize_for_compare(val) for val in final_values) if norm != ""
            }
            tool_norms = [
                norm for norm in (_normalize_for_compare(val) for val in tool_values) if norm != ""
            ]
            if final_norms and tool_norms:
                if any(norm not in final_norms for norm in tool_norms):
                    add_tool_outputs = False

        if add_tool_outputs:
            for val in tool_values:
                _add_candidate(val)

        # Normalize lines for heuristic extraction
        lines = [
            re.sub(r"^model\s*", "", line.strip(), flags=re.I)
            for line in re.split(r"[\r\n]+", s)
        ]
        lines = [line for line in lines if line]

        # Scan from bottom to top to prioritize later statements
        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx]

            m_final = re.search(r"(?i)final\s*answer\s*[:\-]\s*(.+)", line)
            if m_final:
                _add_candidate(m_final.group(1))
                continue

            if re.search(r"(?i)^final\s*answer\b", line) and idx + 1 < len(lines):
                _add_candidate(lines[idx + 1])
                continue

            m_answer = re.search(r"(?i)\banswer\s*[:\-]\s*(.+)", line)
            if m_answer:
                _add_candidate(m_answer.group(1))
                continue

            eq_matches = re.findall(r"=\s*([-+]?\d+(?:\.\d+)?)", line)
            if eq_matches:
                _add_candidate(eq_matches[-1])
                continue

        # Fallback: use the last numeric token in the response
        numeric_tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
        if numeric_tokens:
            _add_candidate(numeric_tokens[-1])

        if not candidates:
            return False

        try:
            label_val = float(label)
        except Exception:
            return False

        def _matches(candidate: str) -> bool:
            cleaned = _clean_num_str(candidate)
            if not cleaned:
                return False
            try:
                cand_val = float(cleaned)
            except ValueError:
                if "=" in cleaned:
                    rhs = cleaned.split("=", 1)[-1]
                    return _matches(rhs)
                return False
            if not math.isfinite(cand_val) or not math.isfinite(label_val):
                return False
            return math.isclose(cand_val, label_val, rel_tol=1e-9, abs_tol=1e-9)

        return any(_matches(cand) for cand in candidates)


class Winogrande(BaseDatasetLoader):
    def __init__(self, mode="eval"):
        super().__init__(mode)
        ds = load_dataset("allenai/winogrande", "winogrande_debiased")
        tmp = ds["validation"].shuffle(seed=42).train_test_split(test_size=0.50, seed=42) # 50 val, 20 test
        self.raw_dict = {
            "train": ds["train"],
            "validation": tmp["train"],
            "test": tmp["test"],
        }
    
    def build_prompt(self, ex):
        WINOGRANDE_TEMPL = (
            "Please choose the correct answer to fill in the blank to complete the given sentence: {sentence}\nOption1: {option1}\nOption2: {option2}\n"
        )
        return WINOGRANDE_TEMPL.format(
            sentence=ex['sentence'],
            option1=ex['option1'],
            option2=ex['option2']
        )

    def get_answer_string(self, ex):
        answer_key = int(ex["answer"])
        if answer_key == 1:
            answer = f"Option1"
        else:
            answer = f"Option2"
        return f"Answer: {answer}."

    def load_tokenized_dataset_chat(self, tokenizer, max_len=100, return_test=False):
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
            answer_key = int(ex["answer"])
            return [prefix, completion, answer_key]

        return self._map_raw_chat_splits(build_raw_features, return_test)


    def load_tokenized_dataset(self, tokenizer, max_len=256, return_test=False):
        def build_tokenized_features(ex):
            prompt = self.build_prompt(ex)
            completion = self.get_answer_string(ex)
            return self._tokenize_prompt_completion(tokenizer, prompt, completion, max_len)
        return self._map_tokenized_splits(build_tokenized_features, return_test)
    
    def load_raw_dataset(self, return_test=False):
        def build_raw_features(ex):
            prompt = self.build_prompt(ex)
            answer_key = int(ex["answer"])
            completion = self.get_answer_string(ex)
            return self._build_raw_example(prompt, completion, answer_key)
        
        return self._map_raw_splits(build_raw_features, return_test)
    
    def evaluate(self, output, label):
        s = output.strip().lower()
        m_ans = re.search(r"answer\s*[:\-]\s*(.*)", s, flags=re.S)
        if not m_ans:
            return False

        answer_section = m_ans.group(1)
        candidates = []
        for pattern, val in [(r"\boption\s*1\b|\boption1\b", 1), (r"\boption\s*2\b|\boption2\b", 2)]:
            m = re.search(pattern, answer_section)
            if m:
                candidates.append((m.start(), val))

        if not candidates:
            return False

        # Only consider the first mentioned option after "Answer:"
        chosen = sorted(candidates, key=lambda x: x[0])[0][1]
        return label == chosen

class GSM8K(BaseDatasetLoader):
    def __init__(self, mode="eval"):
        super().__init__(mode)
        ds = load_dataset("gsm8k", "main")
        tmp = ds["test"].shuffle(seed=42).train_test_split(test_size=0.20, seed=42) # 80 train, 20 test+val
        self.raw_dict = {
            "train": ds["train"],
            "validation": tmp["test"],
            "test": tmp["train"],
        }
    
    def build_prompt(self, ex):
        GSM8K_PROMPT_TMPL = (
            "{question}\n"
        )
        return GSM8K_PROMPT_TMPL.format(
            question=ex['question']
        )
    
    def get_answer_string(self, ex):
        answer = ex["answer"]
        reasoning, answer = answer.split("####")
        reasoning = reasoning.strip().rstrip()
        answer = answer.strip().rstrip()
        return f"{reasoning}\nAnswer: {answer}"

    def load_tokenized_dataset(self, tokenizer, max_len=256, return_test=False):
        def build_tokenized_features(ex):
            prompt = self.build_prompt(ex)
            completion = self.get_answer_string(ex) 
            return self._tokenize_prompt_completion(tokenizer, prompt, completion, max_len)
        return self._map_tokenized_splits(build_tokenized_features, return_test)
    
    def load_raw_dataset(self, return_test=False):
        def build_raw_features(ex):
            prompt = self.build_prompt(ex)
            completion = self.get_answer_string(ex)
            answer_value = completion.split("\nAnswer:")[-1].strip().rstrip().replace(",", "")
            return self._build_raw_example(prompt, completion, float(answer_value))
        
        return self._map_raw_splits(build_raw_features, return_test)
    
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
            answer_value = completion.split("\nAnswer:")[-1].strip().rstrip().replace(",", "")
            # Completion is the assistant response we want to learn
            return [prefix, completion, float(answer_value)]

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

    def evaluate(self, output: str, label: float) -> bool:
        _num_pat = re.compile(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?')
        if not output:
            return False

        s = output.strip()

        # Try to extract from "Answer: ..."
        m = list(re.finditer(r"(?i)answer\s*:\s*(.+)", s))
        if m:
            ans_raw = m[-1].group(1).strip()
            mnum = _num_pat.search(ans_raw)
            if mnum:
                try:
                    pred = float(mnum.group(0).replace(",", ""))
                    return abs(pred - float(label)) < 1e-4
                except:
                    pass

        # Fallback: last number in entire output
        nums = list(_num_pat.finditer(s))
        if nums:
            try:
                pred = float(nums[-1].group(0).replace(",", ""))
                return abs(pred - float(label)) < 1e-4
            except:
                return False

        return False

class MNLI(BaseDatasetLoader):
    def __init__(self, mode="eval"):
        super().__init__(mode)
        ds = load_dataset("nyu-mll/glue", "mnli_matched")
        tmp1 = ds["validation"].shuffle(seed=42).train_test_split(test_size=0.20, seed=42) # 80 train, 20 test+val
        tmp2 = tmp1["test"].shuffle(seed=42).train_test_split(test_size=0.50, seed=42) # 50 test, 50 val
        self.raw_dict = {
            "train": tmp1["train"],
            "validation": tmp2["train"],
            "test": tmp2["test"],
        }
    
    def build_prompt(self, ex):
        MNLI_PROMPT_TMPL = (
            "Read the following premise and hypothesis.\n"
            "Decide whether the hypothesis is ENTAILED by the premise, "
            "CONTRADICTED by the premise, or NEUTRAL.\n\n"
            "Premise:\n{premise}\n\n"
            "Hypothesis:\n{hypothesis}\n\n"
            "Answer:"
        )

        return MNLI_PROMPT_TMPL.format(
            premise=ex['premise'],
            hypothesis=ex['hypothesis']
        )
    
    def get_answer_string(self, ex):
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        answer_key = int(ex["label"])
        answer = label_map.get(answer_key, "unknown")
        return f"Answer: {answer.upper()}."

    def load_tokenized_dataset(self, tokenizer, max_len=100, return_test=False):
        def build_tokenized_features(ex):
            prompt = self.build_prompt(ex)
            completion = self.get_answer_string(ex)
            return self._tokenize_prompt_completion(tokenizer, prompt, completion, max_len)
        return self._map_tokenized_splits(build_tokenized_features, return_test)
    
    def load_raw_dataset(self, return_test=False):
        def build_raw_features(ex):
            prompt = self.build_prompt(ex)
            label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
            answer_key = int(ex["label"])
            answer_key = label_map.get(answer_key, "unknown")
            completion = self.get_answer_string(ex)
            return self._build_raw_example(prompt, completion, answer_key)
        
        return self._map_raw_splits(build_raw_features, return_test)

    def evaluate(self, output: str, label: str) -> bool:
        if not output:
            return False

        # Ground-truth label from dataset ('entailment', 'contradiction', 'neutral')
        gold = label.strip().lower()

        s = output.strip()

        # If there's an "Answer:" in the output, focus on what comes after it
        m = re.search(r"(?i)answer\s*:\s*(.*)", s, flags=re.S)
        if m:
            cand = m.group(1).strip()
        else:
            cand = s

        cand_lower = cand.lower()

        # Look for full-word cues
        if "entailment" in cand_lower or "entailed" in cand_lower:
            pred = "entailment"
        elif "contradiction" in cand_lower or "contradict" in cand_lower:
            pred = "contradiction"
        else:
            pred = "neutral"

        return pred == gold

    def load_tokenized_dataset_chat(self, tokenizer, max_len=100, return_test=False):
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
            label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
            answer_key = int(ex["label"])
            answer_key = label_map.get(answer_key, "unknown")
            # Completion is the assistant response we want to learn
            return [prefix, completion, answer_key]

        return self._map_raw_chat_splits(build_raw_features, return_test)

class SVAMP(BaseDatasetLoader):
    def __init__(self, mode="eval"):
        super().__init__(mode)
        ds = load_dataset("ChilleD/SVAMP")
        tmp = ds["test"].shuffle(seed=42).train_test_split(test_size=0.20, seed=42) # 80 train, 20 test+val
        self.raw_dict = {
            "train": ds["train"],
            "validation": tmp["test"],
            "test": tmp["train"],
        }
    
    def build_prompt(self, ex):
        SVAMP_PROMPT_TMPL = (
            "{body}\n{question}\n"
        )
        return SVAMP_PROMPT_TMPL.format(
            body=ex['Body'],
            question=ex['Question']
        )
    
    def get_answer_string(self, ex):
        answer = ex["Answer"]
        reasoning = ex["Equation"]
        return f"{reasoning} Answer: {answer}"

    def load_tokenized_dataset(self, tokenizer, max_len=256, return_test=False):
        def build_tokenized_features(ex):
            prompt = self.build_prompt(ex)
            completion = self.get_answer_string(ex) 
            return self._tokenize_prompt_completion(tokenizer, prompt, completion, max_len)
        return self._map_tokenized_splits(build_tokenized_features, return_test)
    
    def load_raw_dataset(self, return_test=False):
        def build_raw_features(ex):
            prompt = self.build_prompt(ex)
            completion = self.get_answer_string(ex)
            answer_value = ex["Answer"]
            return self._build_raw_example(prompt, completion, float(answer_value))
        
        return self._map_raw_splits(build_raw_features, return_test)
    
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
            answer_value = ex["Answer"]
            # Completion is the assistant response we want to learn
            return [prefix, completion, float(answer_value)]

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

    def evaluate(self, output: str, label: float) -> bool:
        _num_pat = re.compile(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?')
        if not output:
            return False

        s = output.strip()

        # Try to extract from "Answer: ..."
        m = list(re.finditer(r"(?i)answer\s*:\s*(.+)", s))
        if m:
            ans_raw = m[-1].group(1).strip()
            mnum = _num_pat.search(ans_raw)
            if mnum:
                try:
                    pred = float(mnum.group(0).replace(",", ""))
                    return abs(pred - float(label)) < 1e-4
                except:
                    pass

        # Fallback: last number in entire output
        nums = list(_num_pat.finditer(s))
        if nums:
            try:
                pred = float(nums[-1].group(0).replace(",", ""))
                return abs(pred - float(label)) < 1e-4
            except:
                return False

        return False
