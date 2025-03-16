
import os
import re
import string

import nltk
import langdetect

from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords

from src.utils import flatten


ENGLISH_STOP_WORDS = set(nltk_stopwords.words('english')).union(set(sklearn_stopwords))
SPECIAL_CHARACTERS = "/-'#$%\'()*+-/<=>@[\\]^_`{|}~" + '""“”’' + \
    '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
PUNCTUATION = ".,!?:;"


def remove_stopwords(text):
    words = word_tokenize(text)
    new_text = []
    for w in words:
        if w.lower() not in ENGLISH_STOP_WORDS:
            new_text.append(w)
    return " ".join(new_text)


def extract_stopwords(text):
    words = word_tokenize(text)
    stopword_list = set()
    for w in words:
        if w in stopword_list:
            continue
        w = w.lower()
        if any(w == sw for sw in ENGLISH_STOP_WORDS):
            stopword_list.add(w)
    return list(stopword_list)


def process_stopwords(words, word_ids):
    non_stopwords = []
    stopword_indicies = []
    last_word_id = -1
    flag = False
    for i, word_id in enumerate(word_ids):
        if word_id < 0 or word_id is None:
            continue
        if word_id == last_word_id:
            if flag:
                stopword_indicies.append(i)
        else:
            last_word_id = word_id
            word = words[word_id].lower()
            if any(word == sw for sw in ENGLISH_STOP_WORDS):
                flag = True
                stopword_indicies.append(i)
            else:
                flag = False
                if len(word) == 1:
                    stopword_indicies.append(i)
                else:
                    non_stopwords.append(word)

    non_stopword_indicies = [i for i in range(len(word_ids)) if i not in set(stopword_indicies)]
    non_stopword_ids = [word_ids[i] for i in non_stopword_indicies]
    
    return non_stopword_indicies, non_stopwords, non_stopword_ids


def is_english_text(text):
    try:
        return langdetect.detect(text) == "en"
    except Exception as e:
        print("Exception in is_english_text: ", text)
        return False


def is_upper_and_snake_case(word):
    return bool(re.match(r'[A-Z_]+$', word)) and is_all_uppercase(word)


def is_all_uppercase(word):
    return bool(re.match(r'[A-Z]+', word))


def filter_unacsii(text):
    return text.encode("ascii", errors="ignore").decode()


def extract_years(text):
    years = re.findall(r'\b(18\d{2}|19\d{2}|200\d|201\d|202[0-3])\b', text)
    return [int(y) for y in years]


def word_tokenize(doc):
    return nltk.word_tokenize(doc)


def sentence_tokenize(doc):
    return nltk.sent_tokenize(doc)


def segment_document(doc, max_sent_per_segment, stride, max_segment_len=None):
    sentences = sentence_tokenize(doc)
    segments = []
    for i in range(0, len(sentences), stride):
        segment = " ".join(sentences[i:i + max_sent_per_segment])

        if max_segment_len:
            segment = " ".join(segment.split()[:max_segment_len])
        segments.append(segment)
    return segments


def chunking_document(text, tokenizer=None, size=-1, stride=None, max_length=-1):
    assert size > 0 or max_length > 0

    sentences = sentence_tokenize(text)
    segments, chunk = [], []
    i = 0
    while i < len(sentences):
        if size > 0:
            if len(chunk) == size:
                segments.append(" ".join(chunk))
                if stride is None:
                    chunk = [sentences[i]]
                else:
                    chunk = chunk[stride:] + [sentences[i]]
            else:
                chunk.append(sentences[i])
        elif max_length > 0:
            chunk_text = " ".join(chunk) + sentences[i]
            if tokenizer is not None:
                chunk_token_len = len(tokenizer(chunk_text, add_special_tokens=False)["input_ids"])
            else:
                chunk_token_len = len(chunk_text.split())
            if chunk_token_len > max_length:
                segments.append(" ".join(chunk))
                if stride is None:
                    chunk = [sentences[i]]
                else:
                    chunk = chunk[stride:] + [sentences[i]]
            else:
                chunk.append(sentences[i])
        i += 1

    segments.append(" ".join(chunk))
    return segments