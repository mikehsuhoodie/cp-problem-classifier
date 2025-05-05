import os
from problem_types import codeforces_to_standard, leetcode_to_standard
from typing import List

def get_dataset_filepath(dataset_name: str, is_huggingface_dataset=True) -> str:
    dest_dir = "./dataset/huggingface" if is_huggingface_dataset else "./dataset"

    filename = dataset_name.replace('/', '_')
    os.makedirs(dest_dir, exist_ok=True)

    return f"{dest_dir}/{filename}.csv"

def _convert_labels(labels: List[str], labels_map) -> List[str]:
    converted_labels = []

    for label in labels:
        if label not in labels_map:
            raise RuntimeError()

        mapped_labels = labels_map[label]

        if mapped_labels is None:
            return []

        converted_labels.extend(mapped_labels)

    return converted_labels

def convert_codeforces_labels(labels: List[str]) -> List[str]:
    return _convert_labels(labels, codeforces_to_standard)

def convert_leetcode_labels(labels: List[str]) -> List[str]:
    return _convert_labels(labels, leetcode_to_standard)

