import os
import fire
import shutil
import subprocess
import jsonlines

from src.coliee_dataset import get_segment_data, load_paragraph


def create_bm25_indexes(dataset_path, dataset_name, output_dir):
    tmp_dir = ".bm25_tmp"
    for segment in ["train", "val", "test"]:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)

        indexes_dir = os.path.join(output_dir, segment)
        shutil.rmtree(indexes_dir, ignore_errors=True)
        os.makedirs(indexes_dir)

        case_dirs, _ = get_segment_data(dataset_path, dataset_name, eval_segment=segment)
        for case_dir in case_dirs:
            case_id = os.path.split(case_dir)[1]
            doc_dir = os.path.join(case_dir, "paragraphs")
            doc_cases = sorted(os.listdir(doc_dir))
            for doc_case in doc_cases:
                doc_case_file = os.path.join(doc_dir, doc_case)
                doc_case_data = load_paragraph(doc_case_file)
                doc_num = doc_case.split(".txt")[0]
                dict_ = {"id": f"{case_id}_doc{doc_num}_task2", "contents": doc_case_data}

                with jsonlines.open(f"{tmp_dir}/doc.jsonl", mode="a") as writer:
                    writer.write(dict_)

        subprocess.run(["python", "-m", "pyserini.index", "-collection", "JsonCollection",
                        "-generator", "DefaultLuceneDocumentGenerator", "-threads", "1", "-input",
                        f"{tmp_dir}", "-index", f"{indexes_dir}", "-storePositions", "-storeDocvectors",
                        "-storeRaw"])


if __name__ == "__main__":
    fire.Fire(create_bm25_indexes)
