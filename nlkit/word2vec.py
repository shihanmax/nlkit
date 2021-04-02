from collections import Counter
import logging

import torch
from gensim.models import Word2Vec

logger = logging.getLogger(__file__)


def build_vocab_from_text_file(
    file_path, keep_tokens=("<pad>", "<unk>", "<sos>", "<eos>"), 
    freq_threshold=3, stop_words=(" ",),
):
    """Build vocab from a text file.

    Args:
        file_path (str): text file path
        keep_tokens (tuple, optional): preset tokens. 
            Defaults to ("<pad>", "<unk>", "<sos>", "<eos>").
        freq_threshold (int, optional): word frequency limition. Defaults to 3.
        stop_words (iterable[str], optional): stop words that we don't want
            keep in vocab. Defaults to (" ",).

    Returns:
        tuple: (str_to_idx: dict, idx_to_str: dict)
    """
    word_counter = Counter()
    
    with open(file_path, 'r') as frd:
        for line in frd.readlines():
            word_counter.update(Counter(line.strip()))

    # remove stop words
    if stop_words:
        for word in stop_words:
            word_counter.pop(word)
    
    str_to_idx = {}
    idx_to_str = {}
    cnt = 0
    
    for token in keep_tokens:
        str_to_idx[token] = cnt
        idx_to_str[cnt] = token
        cnt += 1
        
    for word, freq in word_counter.items():
        # filter words with frequency less than threshold
        if freq < freq_threshold:
            continue
        str_to_idx[word] = cnt
        idx_to_str[cnt] = word
        cnt += 1
         
    return str_to_idx, idx_to_str


def load_wv_model(
    w2v_model_path, binary, keep_tokens=("<pad>", "<unk>", "<sos>", "<eos>"),
):
    """Load w2v model and build vocab with pretrained embeddings.

    Args:
        w2v_model_path (str): w2v model path
        binary (bool): indicates if the w2v a binary model
        keep_tokens (tuple, optional): preset tokens. 
            Defaults to ("<pad>", "<unk>", "<sos>", "<eos>").

    Raises:
        Exception: raises when w2v model fails to load

    Returns:
        tuple: (str_to_idx: dict, idx_to_str: dict, embedding: torch.tensor)
    """
    try:
        if binary:
            w2v = Word2Vec.load(w2v_model_path)
        else:
            from gensim.models import KeyedVectors
            w2v = KeyedVectors.load_word2vec_format(
                w2v_model_path, binary=False,
            )

        logger.info("Init: done loading existing w2v model")

    except Exception as e:
        raise Exception(
            f"model load failed: {w2v_model_path} !"
            f"message: {e}",
        )

    str_to_idx = {}
    cnt = 0
    
    for token in keep_tokens:
        str_to_idx[token] = cnt
        cnt += 1

    embedding = torch.tensor(self.w2v.wv.vectors, dtype=torch.float64)

    for_pad = torch.zeros(1, embedding.shape[1], dtype=embedding.dtype)
    for_unk = torch.randn(1, embedding.shape[1], dtype=embedding.dtype)
    for_start = torch.zeros(1, embedding.shape[1], dtype=embedding.dtype)
    for_end = torch.randn(1, embedding.shape[1], dtype=embedding.dtype)

    embedding = torch.cat(
        [for_pad, for_unk, for_start, for_end, embedding], dim=0,
    )

    for token in w2v.wv.vocab.keys():
        # skip unk and pad, start, end
        str_to_idx[token] = w2v.wv.vocab[token].index + len(keep_tokens)

    idx_to_str = {v: k for k, v in str_to_idx.items()}

    return str_to_idx, idx_to_str, embedding
