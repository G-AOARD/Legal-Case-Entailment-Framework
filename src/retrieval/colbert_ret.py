import sys
from collections import defaultdict

import torch
import numpy as np

from tqdm import tqdm

sys.path.append("modules/ColBERT")
from colbert.modeling.colbert import colbert_score
from colbert import Checkpoint
from colbert.infra import ColBERTConfig

from src.utils.optimal_transport import compute_embeddings_alignment
from src.utils.tokenizer_utils import tokens_to_words
from src.utils.text_utils import process_stopwords
from src.utils import aggregate_op, sort_dict, flatten


class ColBERT:
    def __init__(
        self, model_path, doc_padding, alignment, max_length=512, subword_weights=1.,
        filter_stopwords=False, ot_configs=None,
    ):
        assert alignment in ["maxsim", "otp"]
        self.alignment = alignment
        self.max_length = max_length
        self.subword_weights = subword_weights
        self.filter_stopwords = filter_stopwords
        self.ot_configs = ot_configs

        self.doc_padding = doc_padding
        self.n_prefix_tokens = 2
        self.n_suffix_tokens = 1

        config = ColBERTConfig(
            doc_maxlen=self.max_length, query_maxlen=self.max_length,
            mask_punctuation=True, attend_to_mask_tokens=False,
            nbits=32, checkpoint=model_path
        )
        self.model = Checkpoint(model_path, colbert_config=config, verbose=0)
        self.query_tokenizer = self.model.query_tokenizer
        self.doc_tokenizer = self.model.doc_tokenizer

    def get_query_tokens(self, text):
        return self.query_tokenizer.tok.tokenize(text, max_length=self.max_length, add_special_tokens=False)

    def get_query_token_ids(self, text):
        return self.query_tokenizer.tok(
            text, max_length=self.max_length - self.n_prefix_tokens - self.n_suffix_tokens,
            add_special_tokens=False
        )

    def get_query_embeddings(self, texts):
        return self.model.queryFromText(texts, padding="longest")

    def get_query_token_embeddings(self, embeddings, input_ids):
        assert embeddings.size()[1] == input_ids.size()[1], f"{embeddings.size()} {input_ids.size()}"
        input_ids_list = input_ids[0].cpu().numpy().tolist()
        sep_token_pos = input_ids_list.index(self.query_tokenizer.sep_token_id)
        return embeddings[:, self.n_prefix_tokens: sep_token_pos, :], input_ids[:, self.n_prefix_tokens: sep_token_pos]

    def get_doc_tokens(self, text):
        return self.doc_tokenizer.tok.tokenize(text, max_length=self.max_length, add_special_tokens=False)

    def get_doc_token_ids(self, text):
        return self.doc_tokenizer.tok(
            text, max_length=self.max_length - self.n_prefix_tokens - self.n_suffix_tokens,
            padding=self.doc_padding, add_special_tokens=False
        )

    def get_doc_embeddings(self, texts, bsize=32):
        return self.model.docFromText(texts, bsize=bsize, padding=self.doc_padding)[0]

    def get_doc_token_embeddings(self, embeddings, input_ids):
        assert embeddings.size()[1] == input_ids.size()[1], f"{embeddings.size()} {input_ids.size()}"
        input_ids_list = input_ids[0].cpu().numpy().tolist()
        sep_token_pos = input_ids_list.index(self.doc_tokenizer.sep_token_id)
        return embeddings[:, self.n_prefix_tokens: sep_token_pos, :], input_ids[:, self.n_prefix_tokens: sep_token_pos]

    def align_embeddings(self, query_embeddings, doc_embeddings, query_text=None, doc_text=None,
                         corpus=None):
        query_words = tokens_to_words(self.get_query_tokens(query_text))
        query_word_ids = [-1] * self.n_prefix_tokens + self.get_query_token_ids(query_text).word_ids() + \
            [-1] * self.n_suffix_tokens
        if self.filter_stopwords:
            query_non_stopwords_indices, query_words, query_word_ids = process_stopwords(
                query_words, query_word_ids)
            query_embeddings = query_embeddings[:, query_non_stopwords_indices, :]

        doc_words = tokens_to_words(self.get_doc_tokens(doc_text))
        doc_word_ids = [-1] * self.n_prefix_tokens + self.get_doc_token_ids(
            doc_text).word_ids()[:self.max_length - self.n_prefix_tokens - self.n_suffix_tokens] + \
                [-1] * self.n_suffix_tokens
        if self.filter_stopwords:
            doc_non_stopwords_indices, doc_words, doc_word_ids = process_stopwords(
                doc_words, doc_word_ids)
            doc_embeddings = doc_embeddings[:, doc_non_stopwords_indices, :]

        ot_dist, ot_mat = compute_embeddings_alignment(
            query_embeddings[0].cpu(), doc_embeddings[0].cpu(), self.ot_configs,
            query_text=query_text, doc_text=doc_text, query_words=query_words, doc_words=doc_words,
            query_word_ids=query_word_ids, doc_word_ids=doc_word_ids, corpus=corpus,
            subword_weights=self.subword_weights
        )
        return ot_dist, ot_mat

    def score(self, query_embeddings, doc_embeddings, query_text=None, doc_text=None, corpus=None):
        if self.alignment == "maxsim":
            if self.filter_stopwords:
                query_words = tokens_to_words(self.get_query_tokens(query_text))
                query_word_ids = [-1] * self.n_prefix_tokens + self.get_query_token_ids(query_text).word_ids() + \
                    [-1] * self.n_suffix_tokens
                query_non_stopwords_indices, query_words, query_word_ids = process_stopwords(
                    query_words, query_word_ids)
                query_embeddings = query_embeddings[:, query_non_stopwords_indices, :]
            
                doc_words = tokens_to_words(self.get_doc_tokens(doc_text))
                doc_word_ids = [-1] * self.n_prefix_tokens + self.get_doc_token_ids(
                    doc_text).word_ids()[:self.max_length - self.n_prefix_tokens - self.n_suffix_tokens] + \
                        [-1] * self.n_suffix_tokens
                doc_non_stopwords_indices, doc_words, doc_word_ids = process_stopwords(
                    doc_words, doc_word_ids)
                doc_embeddings = doc_embeddings[:, doc_non_stopwords_indices, :]

            _score = colbert_score(
                query_embeddings, doc_embeddings, torch.ones(doc_embeddings.shape[:2])
            ).item()
        elif self.alignment == "otp":
            ot_dist, _ = self.align_embeddings(
                query_embeddings, doc_embeddings, query_text=query_text, doc_text=doc_text,
                corpus=corpus
            )
            _score = -ot_dist
        return _score

    def get_ot_alignment(self, query_text, doc_text):
        query_embeddings = self.get_query_embeddings([query_text])
        doc_embeddings = self.get_doc_embeddings([doc_text])
        ot_dist, ot_mat = self.align_embeddings(query_embeddings, doc_embeddings, query_text, doc_text)
        return ot_dist, ot_mat

    def get_word_alignments(self, query_text, doc_text):
        query_embeddings = self.get_query_embeddings([query_text])
        query_words = tokens_to_words(self.get_query_tokens(query_text))
        query_word_ids = [-1] * self.n_prefix_tokens + self.get_query_token_ids(query_text).word_ids() + \
            [-1] * self.n_suffix_tokens
        if self.filter_stopwords:
            _, _, query_word_ids = process_stopwords(
                query_words, query_word_ids)

        doc_embeddings = self.get_doc_embeddings([doc_text])
        doc_words = tokens_to_words(self.get_doc_tokens(doc_text))
        doc_word_ids = [-1] * self.n_prefix_tokens + self.get_doc_token_ids(
            doc_text).word_ids()[:self.max_length - self.n_prefix_tokens - self.n_suffix_tokens] + \
                [-1] * self.n_suffix_tokens
        if self.filter_stopwords:
            _, _, doc_word_ids = process_stopwords(
                doc_words, doc_word_ids)

        _, ot_mat = self.align_embeddings(query_embeddings, doc_embeddings, query_text, doc_text)
        alignment_indices = np.argwhere(ot_mat > 0)
        rows, cols = alignment_indices[:, 0].tolist(), alignment_indices[:, 1].tolist()

        word_alignments = defaultdict(lambda: [])
        alignment_weights = defaultdict(lambda: defaultdict(lambda: 0))
        for r, c in zip(rows, cols):
            if c >= len(doc_word_ids):
                continue
            query_word_pos = query_word_ids[r]
            doc_word_pos = doc_word_ids[c]
            if query_word_pos == -1 or doc_word_pos == -1:
                continue
            word_alignments[query_words[query_word_pos]].append(doc_words[doc_word_pos])
            alignment_weights[query_words[query_word_pos]][doc_words[doc_word_pos]] += ot_mat[r, c]

        for k, v in word_alignments.items():
            word_alignments[k] = list(set(v))
        return word_alignments, alignment_weights

    def visualize_alignment(self, query_text, doc_text):
        word_alignments = self.get_word_alignments(query_text, doc_text)
        query_attn_words = set(word_alignments.keys())
        doc_attn_words = set(flatten(list(word_alignments.values())))

        query_tokens = self.get_query_tokens(query_text)
        query_words = tokens_to_words(query_tokens)
        doc_tokens = self.get_doc_tokens(doc_text)
        doc_words = tokens_to_words(doc_tokens)
        query_attn_str, doc_attn_str = "", ""
        for i, w in enumerate(query_words):
            if w in query_attn_words:
                query_attn_str += w.upper() + " "
            else:
                query_attn_str += w + " "

        for i, w in enumerate(doc_words):
            if w in doc_attn_words:
                doc_attn_str += w.upper() + " "
            else:
                doc_attn_str += w + " "

        return query_attn_str, doc_attn_str, word_alignments


def predict_colbert(test_dataset, model):
    query_scores = defaultdict(lambda: {})
    for query_batch in tqdm(test_dataset, desc="ColBERT prediction"):
        query_id = query_batch[0][3]["query_id"]
        query_text = query_batch[0][0]

        doc_ids = [query_batch[i][3]["doc_id"] for i in range(len(query_batch))]
        doc_texts = [query_batch[i][1] for i in range(len(query_batch))]
        doc_scores = defaultdict(lambda: 0)

        doc_segment_ids = doc_ids
        doc_segments = doc_texts
        corpus = [query_text] + doc_segments

        query_embeddings = model.get_query_embeddings([query_text])
        for i, doc_id in enumerate(doc_segment_ids):
            if not doc_segments[i]:
                continue
            doc_embeddings = model.get_doc_embeddings([doc_segments[i]])

            score = model.score(
                query_embeddings, doc_embeddings, query_text=query_text, doc_text=doc_segments[i],
                corpus=corpus
            )
            doc_scores[doc_id] = aggregate_op(score, doc_scores[doc_id], "max")
        query_scores[query_id] = sort_dict(doc_scores, reverse=True)
    return query_scores
