import os
import json
import pickle


def save_txt(file_path, text, makes_dir=False):
    if makes_dir:
        save_dir = os.path.split(file_path)[0]
        os.makedirs(save_dir, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(text)


def load_txt(file_path, skip=0):
    with open(file_path) as f:
        while skip > 0:
            f.readline()
            skip -= 1
        data = f.read()
    return data


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def save_json(file_path, d, makes_dir=False):
    if makes_dir:
        save_dir = os.path.split(file_path)[0]
        os.makedirs(save_dir, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(d, f, indent=4)


def load_jsonl(file_path):
    with open(file_path) as f:
        result = [json.loads(jline) for jline in f.read().splitlines()]
    return result


def pickle_load_appended_dict(file_path):
    d = {}
    with open(file_path, "rb") as f:
        while 1:
            try:
                data = pickle.load(f)
                d = {**d, **data}
            except EOFError:
                break
    return d
