import json
import os

from .dataloaders import *
from .dataloaders_LRA import *

dataset_dict = {
    "ASDiv": ASDiv,
    "BoolQ": BoolQ,
    "Winogrande": Winogrande,
    "ListOps": ListOps,
    "GSM8K": GSM8K,
    "MNLI": MNLI,
    "SVAMP": SVAMP,
}

epochs_per_dataset = {
    "BoolQ": 1,
    "Winogrande": 1,
    "ListOps": 1,
    "GSM8K": 3,
}

max_new_tokens_per_dataset = {
    "BoolQ": 10,
    "Winogrande": 5,
    "ListOps": 10,
    "GSM8K": 100,
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
