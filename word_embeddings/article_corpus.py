import random
import numpy as np
import pandas as pd


class ArticleCorpus:
    def __init__(self):
        self.read_dataset()
        self._tokens = None
        self.create_corpus()
        self.create_vocab()
        self.get_rejection_prob()

    def read_dataset(self):
        self.data = pd.read_pickle("./data/processed_dataset.pkl")

    def create_corpus(self):
        titles = self.data.title.values.tolist()
        abstracts = self.data.abstract.values.tolist()
        all_texts = titles + abstracts
        corpus = [this_text.strip().split() for this_text in all_texts]
        self.corpus = corpus

    def create_vocab(self):

        vocab = []
        word_to_idx = dict()
        idx_to_word = dict()
        word_to_freq = dict()
        corpus_size = 0
        idx = 0

        for article in self.corpus:
            for word in article:
                corpus_size += 1
                if word not in vocab:
                    vocab.append(word)
                    word_to_idx[word] = idx
                    idx_to_word[idx] = word
                    word_to_freq[word] = 1
                    idx += 1
                else:
                    word_to_freq[word] += 1

        vocab.append("UNK")
        word_to_idx["UNK"] = idx
        idx_to_word[idx] = "UNK"
        word_to_freq["UNK"] = 1
        corpus_size += 1

        self._vocab = vocab
        self._word_to_idx = word_to_idx
        self._idx_to_word = idx_to_word
        self._word_to_freq = word_to_freq
        self.corpus_size = corpus_size
        self.vocab_size = len(vocab)

    def get_rejection_prob(self):
        threshold = 100.0
        word_to_rejection_prob = dict()
        for i in range(self.vocab_size):
            word = self._vocab[i]
            word_freq = self._word_to_freq[word] * 1.0
            rp = max(0, 1.0 - np.sqrt(threshold / word_freq))
            word_to_rejection_prob[word] = rp
        self._word_to_rejection_prob = word_to_rejection_prob

    def get_random_context(self, window=5):
        n_doc = len(self.corpus)
        doc_id = random.randint(0, n_doc - 1)
        doc_text = self.corpus[doc_id]
        doc_text = [
            word
            for word in doc_text
            if random.random() >= self._word_to_rejection_prob[word]
        ]
        c_idx = random.randint(0, len(doc_text) - 1)
        center_word = doc_text[c_idx]
        context = doc_text[max(0, c_idx - window) : c_idx]
        if c_idx + 1 < len(doc_text):
            context += doc_text[c_idx + 1 : min(c_idx + window, len(doc_text))]
        context = [word for word in context if word != center_word]
        if len(context) > 0:
            return center_word, context
        else:
            return self.get_random_context(window)


if __name__ == "__main__":
    corpus = ArticleCorpus()
    center_word, context = corpus.get_random_context()
    print(center_word)
    print(context)
