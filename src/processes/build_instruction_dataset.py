import os
import fire

import numpy as np

from src.coliee_dataset import get_segment_data, load_paragraph
from src.utils import load_json, save_json


def main(
    dataset_path,
    negative_samples_path,
    num_negatives_per_sample=5,
    output_dir="./artifacts/instruction_dataset"
):
    np.random.seed(442)

    negative_samples = load_json(negative_samples_path)
    instruction_prompt = "Determine the entailment relationship."
    input_prompt = "Query: {}\n\nDocument: {}"
    output_prompt = "Entailment:"
    os.makedirs(output_dir, exist_ok=True)
    for segment in ["train", "val", "test"]:
        inst_dataset = []
        if segment == "train":
            output_path = os.path.join(output_dir, f"legal-ent_train-n{num_negatives_per_sample}.json")
        else:
            output_path = os.path.join(output_dir, f"legal-ent_{segment}.json")
        
        case_dirs, labels = get_segment_data(dataset_path, "coliee_task2_2024", segment)
        for case_dir in case_dirs:
            case_id = os.path.split(case_dir)[1]
            
            if segment == "train":
                query_text = load_paragraph(os.path.join(case_dir, "entailed_fragment.txt"), max_length=128)
            else:
                query_text = load_paragraph(os.path.join(case_dir, "entailed_fragment.txt"), max_length=400)

            positive_docs = labels[case_id]
            for doc_file in positive_docs:
                
                _label = "True"
                if segment == "train":
                    doc_text = load_paragraph(os.path.join(case_dir, "paragraphs", doc_file))
                    doc_words = doc_text.split()
                    doc_text = " ".join(doc_words[-128:]).strip()
                    if len(doc_words) > 200:
                        _label = "False"
                else:
                    doc_text = load_paragraph(os.path.join(case_dir, "paragraphs", doc_file),
                                              max_length=400, split_last=True)

                inst_dataset.append({
                    "instruction": instruction_prompt,
                    "input": input_prompt.format(query_text, doc_text),
                    "output": output_prompt + _label
                })
            
            if segment == "train":
                negative_docs = [x + ".txt" for x in negative_samples[case_id][:num_negatives_per_sample]]
            else:
                negative_docs = os.listdir(os.path.join(case_dir, "paragraphs"))
            for doc_file in negative_docs:
                if segment == "train":
                    doc_text = load_paragraph(os.path.join(case_dir, "paragraphs", doc_file),
                                              max_length=128, split_last=True)
                else:
                    doc_text = load_paragraph(os.path.join(case_dir, "paragraphs", doc_file),
                                              max_length=400, split_last=True)
                inst_dataset.append({
                    "instruction": instruction_prompt,
                    "input": input_prompt.format(query_text, doc_text),
                    "output": output_prompt + "False" 
                })
        
        if segment == "train":
            np.random.shuffle(inst_dataset)
        save_json(output_path, inst_dataset)


if __name__ == "__main__":
    fire.Fire(main)
