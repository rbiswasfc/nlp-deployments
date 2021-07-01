import random
import numpy as np
import pandas as pd


class ArticleCorpus:
    """
    create article corpus object from a dataframe containing text fields. 
    It prodivies the following public attributes and methods:
    
    @ attributes:
    corpus -> text data for building the object
    corpus_size -> total number of words in the corpus
    vocab -> a list of all distinct words in the corpus
    vocab_size -> total number of distinct words in the corpus
    word_to_idx -> a dict to map word to corresponding index
    idx_to_word -> a dict to map indices to corresponding words
    word_to_freq -> word frequency dict 

    @ methods:
    get_random_context -> get a random (center, context) pair
    """

    def __init__(self, corpus):
        """
        initialize the corpus object

        :param corpus: underlying text data from building the corpus object
        :type corpus: List[List]
        """
        assert type(corpus) == list, "Error: corpus should be list of lists"
        assert type(corpus[0]) == list, "Error: corpus should be list of lists"

        self._corpus = None
        self._vocab = []
        self._corpus_size = 0
        self._vocab_size = 0

        self._word_to_idx = dict()
        self._idx_to_word = dict()
        self._word_to_freq = dict()
        self._word_to_rejection_prob = dict()

        self.add_data(corpus)
        self._get_rejection_prob()

    @property
    def vocab(self):
        return self._vocab

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def corpus(self):
        return self._corpus

    @property
    def corpus_size(self):
        return self._corpus_size

    @property
    def word_to_idx(self):
        return self._word_to_idx

    @property
    def idx_to_word(self):
        return self._idx_to_word

    @property
    def word_to_freq(self):
        return self._word_to_freq

    @classmethod
    def from_dataframe(cls, df, text_col):
        """
        Alternative constructor for the corpus object. 
        A dataframe with text data field can used to build the object.

        :param df: dataframe with underlying text data
        :type df: pd.DataFrame
        :param text_col: column name for the text data
        :type text_col: str
        :return: corpus object
        :rtype: ArticleCorpus
        """
        text_data = df[text_col].values.tolist()
        corpus = [this_text.strip().split() for this_text in text_data]
        return cls(corpus)

    def add_data(self, data, text_col=None):
        """
        add data to corpus

        :param data: data to add to corpus
        :type data: List[List] or pd.DataFrame
        :param text_col: provide this parameter if dataframe is passed as data
        :type text_col: str, optional
        """
        if isinstance(data, pd.DataFrame):
            assert text_col, "provide column data for text data"
            assert text_col in data.columns, "{} column not found".format(text_col)
            text_data = data[text_col].values.tolist()
            data = [this_text.strip().split() for this_text in text_data]

        assert type(data) == list, "Error: corpus should be list of lists"
        assert type(data[0]) == list, "Error: corpus should be list of lists"

        if not self.corpus:
            self._corpus = data
        else:
            self._corpus.extend(data)

        for article in data:
            for word in article:
                self._corpus_size += 1
                if word not in self._vocab:
                    self._vocab.append(word)
                    self._word_to_idx[word] = self._vocab_size
                    self._idx_to_word[self._vocab_size] = word
                    self._word_to_freq[word] = 1
                    self._vocab_size += 1
                else:
                    self._word_to_freq[word] += 1

        if "UNK" not in self._vocab:
            self._vocab.append("UNK")
            self._word_to_idx["UNK"] = self._vocab_size
            self._idx_to_word[self._vocab_size] = "UNK"
            self._word_to_freq["UNK"] = 1
            self._vocab_size += 1

    def _get_rejection_prob(self):
        """
        implements subsampling of frequent words
        if randomly generated probability more than rejection probability for the word,
        then the word will be considered in training.
        """
        threshold = 100.0  # TODO: update therehold based on word frequency distribution
        for i in range(self._vocab_size):
            word = self._vocab[i]
            word_freq = self._word_to_freq[word] * 1.0
            rp = max(0, 1.0 - np.sqrt(threshold / word_freq))
            self._word_to_rejection_prob[word] = rp

    def get_random_context(self, window=5):
        """
        generate a random (center, context) pair

        :param window: window size for the context, defaults to 5
        :type window: int, optional
        :return: center word, context words
        :rtype: tuple (str, List)
        """
        n_doc = len(self._corpus)
        doc_id = random.randint(0, n_doc - 1)
        doc_text = self._corpus[doc_id]
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
    df = pd.read_pickle("./data/processed_dataset.pkl")
    corpus = ArticleCorpus.from_dataframe(df, "abstract")
    center_word, context = corpus.get_random_context()
    print(center_word)
    print(context)
