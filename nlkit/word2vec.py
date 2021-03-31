import logging

import torch
from gensim.models import Word2Vec

logger = logging.getLogger(__file__)


class Word2VecWrapper(object):

    def __init__(self, w2v_model_path, binary):
        self.w2v = None
        self.size = None

        self.pad = "<pad>"
        self.unk = "<unk>"
        self.start_decode_token = "<sos>"
        self.end_decode_token = "<eos>"

        self.pad_idx = 0
        self.unk_idx = 1
        self.start_decode_idx = 2
        self.end_decode_idx = 3

        self.str_to_idx, self.idx_to_str, self.embedding = self._init_model(
            w2v_model_path, binary,
        )

    def _init_model(self, w2v_model_path, binary):
        try:
            if binary:
                self.w2v = Word2Vec.load(w2v_model_path)
            else:
                from gensim.models import KeyedVectors
                self.w2v = KeyedVectors.load_word2vec_format(
                    w2v_model_path, binary=False,
                )

            logger.info("Init: done loading existing w2v model")

        except Exception as e:
            raise Exception(
                f"model load failed: {w2v_model_path} !"
                f"message: {e}",
            )

        str_to_idx = {
            self.unk: self.unk_idx,
            self.pad: self.pad_idx,
            self.start_decode_token: self.start_decode_idx,
            self.end_decode_token: self.end_decode_idx,
        }

        embedding = torch.tensor(self.w2v.wv.vectors, dtype=torch.float64)

        for_pad = torch.zeros(1, embedding.shape[1], dtype=embedding.dtype)
        for_unk = torch.randn(1, embedding.shape[1], dtype=embedding.dtype)
        for_start = torch.zeros(1, embedding.shape[1], dtype=embedding.dtype)
        for_end = torch.randn(1, embedding.shape[1], dtype=embedding.dtype)

        embedding = torch.cat(
            [for_pad, for_unk, for_start, for_end, embedding], dim=0,
        )

        for token in self.w2v.wv.vocab.keys():
            # skip unk and pad, start, end
            str_to_idx[token] = self.w2v.wv.vocab[token].index + 4

        idx_to_str = {v: k for k, v in str_to_idx.items()}

        self.size = len(idx_to_str)

        return str_to_idx, idx_to_str, embedding
