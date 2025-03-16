import fire

from src.utils import load_jsonl


def main(predict_file):
    predict_data = load_jsonl(predict_file)

    tp, fp, fn = 0, 0, 0
    for data in predict_data:
        if data["label"].endswith("True"):
            if data["predict"].endswith("True"):
                tp += 1
            else:
                fn += 1
        elif data["predict"].endswith("True"):
            fp += 1
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    print(f1, precision, recall)


if __name__ == "__main__":
    fire.Fire(main)
