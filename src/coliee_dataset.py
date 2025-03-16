import os
import re
import copy
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

from src.utils.text_utils import sentence_tokenize, word_tokenize, filter_unacsii
from src.utils.file_utils import (
    load_json,
    load_txt
)

SUPPRESSED_KEYWORDS = [
    "FRAGMENT_SUPPRESSED",
    "REFERENCE_SUPPRESSED",
    "CITATION_SUPPRESSED",
    "DATE_SUPPRESSED",
]

DATASET_INDEXES = {
    "coliee_task2_2020": {
        "train": [101, 325],
        "val": [1, 100],
        "test": [326, 425]
    },
    "coliee_task2_2021": {
        "train": [101, 425],
        "val": [1, 100],
        "test": [426, 525]
    },
    "coliee_task2_2022": {
        "train": [101, 525],
        "val": [1, 100],
        "test": [526, 625]
    },
    "coliee_task2_2023": {
        "train": [101, 625],
        "val": [1, 100],
        "test": [626, 725]
    },
    "coliee_task2_2024": {
        "train": [101, 725],
        "val": [1, 100],
        "test": [726, 825]
    }
}


def load_paragraph(
    file_path,
    uncased=False,
    min_length=None,
    max_length=None,
    split_last=False,
    filter_suppressed_keywords=True
):
    if not os.path.exists(file_path):
        return None
    text = load_txt(file_path).strip()
    if uncased:
        text = text.lower()

    text = filter_unacsii(text)
    if filter_suppressed_keywords:
        for word in SUPPRESSED_KEYWORDS:
            text = re.sub(f"\<\s?{word}+\s?\>", "", text)
            text = re.sub(f"\[\s?{word}+\s?\]", "", text)
            text = text.replace(word, "")

    text = re.sub(r"\[\s?[0-9]+\s?\]", "", text)
    words = word_tokenize(text)
    if min_length and len(words) <= min_length:
        return None

    if max_length:
        if split_last:
            words = words[-max_length:]
        else:
            words = words[:max_length]

    text = " ".join(words).strip()
    return text


def get_segment_data(dataset_path, dataset_name, eval_segment):
    data_dir = os.path.join(dataset_path, "task2_train_files_2024")
    labels_file = os.path.join(dataset_path, "task2_train_labels_2024.json")
    labels = load_json(labels_file)

    if eval_segment == "train":
        case_dirs = list(filter(
            lambda d: DATASET_INDEXES[dataset_name]["train"][1] >= int(d) >= \
                 DATASET_INDEXES[dataset_name]["train"][0],
            os.listdir(data_dir)))
        case_dirs = [os.path.join(data_dir, c) for c in sorted(case_dirs)]
        labels = {k: v for k, v in labels.items() if DATASET_INDEXES[dataset_name]["train"][1] >= int(k) >= \
                  DATASET_INDEXES[dataset_name]["train"][0]}
    elif eval_segment == "val":
        case_dirs = list(filter(
            lambda d: DATASET_INDEXES[dataset_name]["val"][1] >= int(d) >= \
                DATASET_INDEXES[dataset_name]["val"][0],
            os.listdir(data_dir)))
        case_dirs = [os.path.join(data_dir, c) for c in sorted(case_dirs)]
        labels = {k: v for k, v in labels.items() if DATASET_INDEXES[dataset_name]["val"][1] >= int(k) >= \
                  DATASET_INDEXES[dataset_name]["val"][0]}
    elif eval_segment == "test":
        if dataset_name == "coliee_task2_2024":
            data_dir = os.path.join(dataset_path, "task2_test_files_2024")
            labels_file = os.path.join(dataset_path, "task2_test_labels_2024.json")
            labels = load_json(labels_file)
        case_dirs = list(filter(
            lambda d: DATASET_INDEXES[dataset_name]["test"][1] >= int(d) >= \
                DATASET_INDEXES[dataset_name]["test"][0],
            os.listdir(data_dir)))
        case_dirs = [os.path.join(data_dir, c) for c in sorted(case_dirs)]
        labels = {k: v for k, v in labels.items() if DATASET_INDEXES[dataset_name]["test"][1] >= int(k) >= \
                  DATASET_INDEXES[dataset_name]["test"][0]}
    elif eval_segment == "all":
        if dataset_name == "coliee_task2_2024":
            test_data_dir = os.path.join(dataset_path, "task2_test_files_2024")
            test_labels_file = os.path.join(dataset_path, "task2_test_labels_2024.json")
            test_labels = load_json(test_labels_file)

            case_dirs = [os.path.join(data_dir, c) for c in sorted(os.listdir(data_dir))] + \
                [os.path.join(test_data_dir, c) for c in sorted(os.listdir(test_data_dir))]
            labels = {**labels, **test_labels}
        else:
            case_dirs = list(filter(
                lambda d: DATASET_INDEXES[dataset_name]["test"][1] >= int(d),
                os.listdir(data_dir)))
            case_dirs = [os.path.join(data_dir, c) for c in sorted(case_dirs)]
            labels = {k: v for k, v in labels.items() if DATASET_INDEXES[dataset_name]["test"][1] >= int(k)}
    else:
        raise ValueError(eval_segment)

    if isinstance(labels, list):
        labels = {k: [] for k in labels}
    return case_dirs, labels


def load_dataset(
    dataset_path,
    dataset_name,
    model_type=None,
    word_threshold=None,
    max_num_sentences=None,
    num_positives_per_example=None,
    num_negatives_per_example=None,
    sampling_strategy=None,
    query_retrieval_file=None,
    load_splits=["train", "val", "test"],
    reranking_batching=False,
):
    query_retrieval_dict = None
    if query_retrieval_file:
        print("Load query_retrieval_file: ", query_retrieval_file)
        query_retrieval_dict = load_json(query_retrieval_file)

    if "train" in load_splits:
        train_dataset = LegalCaseEntailmentDataset(
            dataset_path,
            dataset_name,
            model_type=model_type,
            segment="train",
            num_positives_per_example=num_positives_per_example,
            num_negatives_per_example=num_negatives_per_example,
            sampling_strategy=sampling_strategy,
            word_threshold=word_threshold,
            max_num_sentences=max_num_sentences,
            query_retrieval_dict=query_retrieval_dict,
            reranking_batching=reranking_batching,
        )
    else:
        train_dataset = None

    if "val" in load_splits:
        val_dataset = LegalCaseEntailmentDataset(
            dataset_path,
            dataset_name,
            model_type=model_type,
            segment="val",
            word_threshold=word_threshold,
            max_num_sentences=max_num_sentences,
            reranking_batching=reranking_batching
        )
    else:
        val_dataset = None

    if "test" in load_splits:
        test_dataset = LegalCaseEntailmentDataset(
            dataset_path,
            dataset_name,
            model_type=model_type,
            segment="test",
            word_threshold=word_threshold,
            max_num_sentences=max_num_sentences,
            reranking_batching=reranking_batching
        )
    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset


class LegalCaseEntailmentDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        dataset_name,
        model_type=None,
        segment="train",
        num_positives_per_example=None,
        num_negatives_per_example=None,
        sampling_strategy=None,
        word_threshold=None,
        max_num_sentences=None,
        query_retrieval_dict=None,
        reranking_batching=False,
    ):
        super().__init__()

        self._iter = 0
        self.model_type = model_type
        self.is_train = segment == "train"
        self.case_dirs, self.labels = get_segment_data(dataset_path, dataset_name, segment)
        self.data = self.load_cases(
            self.case_dirs,
            self.labels,
            is_train=self.is_train,
            word_threshold=word_threshold,
            query_retrieval_dict=query_retrieval_dict
        )

        self.num_positives_per_example = num_positives_per_example
        self.num_negatives_per_example = num_negatives_per_example
        self.sampling_strategy = sampling_strategy or "random"
        self.word_threshold = word_threshold
        self.max_num_sentences = max_num_sentences
        self.reranking_batching = reranking_batching

        self.positive_samples = {
            i: list(d["positive"].keys()) for i, d in self.data.items()
        }
        self.ps_iter = copy.deepcopy(self.positive_samples)
        if query_retrieval_dict:
            self.negative_samples = {i: query_retrieval_dict[i] for i in self.data}
        else:
            self.negative_samples = {
                i: list(d["negative"].keys()) for i, d in self.data.items()
            }
        self.ns_iter = copy.deepcopy(self.negative_samples)

        self.case_list = list(self.data.keys())
 
        self.input_data = []
        self.build_input_dataset()

    @staticmethod
    def load_cases(
        case_dirs,
        labels,
        is_train=False,
        word_threshold=None,
        query_retrieval_dict=None,
    ):
        id_mapping = {}
        for case_dir in case_dirs:
            case_id = os.path.split(case_dir)[1]
            query_text = load_paragraph(
                os.path.join(case_dir, "entailed_fragment.txt"),
                max_length=word_threshold
            )
            if query_text is None:
                continue
            id_mapping[case_id] = {"text": query_text, "positive": {}, "negative": {}, "docs": {}}

            doc_dir = os.path.join(case_dir, "paragraphs")
            for doc_file in os.listdir(doc_dir):
                doc_id = os.path.splitext(doc_file)[0]
                if query_retrieval_dict is not None:
                    if is_train:
                        if doc_id not in query_retrieval_dict[case_id] and doc_file not in labels[case_id]:
                            continue
                    elif doc_id not in query_retrieval_dict[case_id]:
                        continue

                doc_text = load_paragraph(
                    os.path.join(doc_dir, doc_file),
                    min_length=10,
                    max_length=word_threshold,
                )
                if doc_text is None:
                    continue
                if doc_file in labels[case_id]:
                    id_mapping[case_id]["positive"][doc_id] = doc_text
                else:
                    id_mapping[case_id]["negative"][doc_id] = doc_text
                id_mapping[case_id]["docs"][doc_id] = doc_text
        return id_mapping

    def build_input_dataset(self):
        if self.model_type in ["CAUSAL_LM", "FEATURE_EXTRACTION", "SEQ_CLS"]:
            if self.reranking_batching:
                self.build_reranking_dataset()
            else:
                self.build_pair_dataset()
        elif self.model_type in ["RET", "T5"]:
            self.build_reranking_dataset()
        else:
            raise ValueError(self.model_type)
        self._iter += 1

    def build_pair_dataset(self):
        print("\nBuild pair dataset: ")
        self.input_data = []
        for case, case_data in tqdm(self.data.items()):
            if self.num_positives_per_example:
                n_positives = min(self.num_positives_per_example, len(case_data["positive"]))
                positive_ids = np.random.choice(
                    list(case_data["positive"].keys()), size=n_positives, replace=False)
            else:
                positive_ids = list(case_data["positive"].keys())
            if self.num_negatives_per_example:
                n_negatives = min(self.num_negatives_per_example, len(case_data["negative"]))
                negative_ids = np.random.choice(
                    list(case_data["negative"]), size=n_negatives, replace=False)
            else:
                negative_ids = list(case_data["negative"].keys())

            for nid in negative_ids:
                neg_text = case_data["negative"][nid]
                if self.max_num_sentences:
                    neg_text_sents = sentence_tokenize(neg_text)
                    neg_text = " ".join(neg_text_sents[:self.max_num_sentences])
                self.input_data.append((
                    case_data["text"],
                    neg_text,
                    0,
                    {"query_id": case, "doc_id": nid}
                ))
            
            for pid in positive_ids:
                pos_text = case_data["positive"][pid]
                if self.max_num_sentences:
                    pos_text_sents = sentence_tokenize(pos_text)
                    pos_text = " ".join(pos_text_sents[:self.max_num_sentences])
                self.input_data.append((
                    case_data["text"],
                    pos_text,
                    1,
                    {"query_id": case, "doc_id": pid}
                ))

    def build_reranking_dataset(self):
        print("\nBuild reranking dataset: ")

        self.input_data = []
        for case_id, case_data in tqdm(self.data.items()):
            if self.num_positives_per_example:
                n_positives = min(self.num_positives_per_example, len(case_data["positive"]))
                if self.sampling_strategy == "random":
                    positive_ids = np.random.choice(
                        list(case_data["positive"].keys()), size=n_positives, replace=False)
                else:
                    if len(self.ps_iter[case_id]) >= n_positives:
                        positive_ids = self.ps_iter[case_id][:n_positives]
                        self.ps_iter[case_id] = self.ps_iter[case_id][n_positives:]
                    else:
                        diff = n_positives - len(self.ps_iter[case_id])
                        positive_ids = self.ps_iter[case_id]
                        positive_ids += self.positive_samples[case_id][:diff]
                        self.ps_iter[case_id] = self.positive_samples[case_id][diff:]
            else:
                positive_ids = list(case_data["positive"].keys())
            if self.is_train and len(positive_ids) == 0:
                    continue

            if self.num_negatives_per_example:
                n_negatives = min(self.num_negatives_per_example, len(case_data["negative"]))
                if self.sampling_strategy == "random":
                    negative_ids = np.random.choice(
                        list(case_data["negative"]), size=n_negatives, replace=False)
                else:
                    if len(self.ns_iter[case_id]) >= n_negatives:
                        negative_ids = self.ns_iter[case_id][:n_negatives]
                        self.ns_iter[case_id] = self.ns_iter[case_id][n_negatives:]
                    else:
                        diff = n_negatives - len(self.ns_iter[case_id])
                        negative_ids = self.ns_iter[case_id] + self.negative_samples[case_id][:diff]
                        self.ns_iter[case_id] = self.negative_samples[case_id][diff:]
            else:
                negative_ids = list(case_data["negative"].keys())

            case_batch = []
            for nid in negative_ids:
                neg_text = case_data["negative"][nid]
                if self.max_num_sentences:
                    neg_text_sents = sentence_tokenize(neg_text)
                    neg_text = " ".join(neg_text_sents[:self.max_num_sentences])
                case_batch.append((
                    case_data["text"],
                    neg_text,
                    0,
                    {"query_id": case_id, "doc_id": nid}
                ))

            for pid in positive_ids:
                pos_text = case_data["positive"][pid]
                if self.max_num_sentences:
                    pos_text_sents = sentence_tokenize(pos_text)
                    pos_text = " ".join(pos_text_sents[:self.max_num_sentences])
                case_batch.append((
                    case_data["text"],
                    pos_text,
                    1,
                    {"query_id": case_id, "doc_id": pid}
                ))
            self.input_data.append(case_batch)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx]
