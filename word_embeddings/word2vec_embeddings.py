import random
import logging
import numpy as np
import pandas as pd
from time import time
from utils import setup_logger, check_create_dir
from article_corpus import ArticleCorpus
from embedding_interface import IWordEmbedding
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger = setup_logger(logger)


class Word2VecBase(IWordEmbedding):
    def __init__(self, embedding_dim=16, loss_type="nll"):
        assert loss_type in [
            "nce",
            "nll",
        ], "loss type not supported, available options are 'nce' and 'nll'"

        self.loss_type = loss_type
        self._vocab_size = None
        self._embedding_dim = embedding_dim
        self._mat_center = None
        self._mat_outside = None

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def model_weights(self):
        if isinstance(self._mat_center, np.ndarray):
            w = np.concatenate((self._mat_center, self._mat_outside), axis=0)
            return w
        else:
            return None

    @staticmethod
    def sigmoid(x):
        """
        compute sigmoid function of the input

        :param x: input value
        :type x: float or np.ndarray
        :return: element wise sigmoid function applied on input
        :rtype: float or np.ndarray
        """
        s = 1.0 / (1 + np.exp(-x))
        return s

    @staticmethod
    def softmax(x):
        """
        compute softmax of the input 

        :param x: input logits
        :type x: np.ndarray
        :return: computed probability from logits using softmax
        :rtype: np.ndarray
        """
        x -= np.expand_dims(
            np.max(x, axis=-1), axis=-1
        )  # numerical trick for stability
        x = np.exp(x)
        factor = np.expand_dims(np.sum(x, axis=-1), axis=-1)
        x /= factor
        return x

    def _init_embeddings(self):
        self._mat_center = (
            np.random.rand(self._vocab_size, self._embedding_dim) - 0.5
        )  # / self._embedding_dim
        # self._mat_outside = np.zeros((self._vocab_size, self._embedding_dim))
        self._mat_outside = np.random.rand(self._vocab_size, self._embedding_dim) - 0.5

    def _get_nll_loss_and_grad(self, center_idx, outside_idx):
        # get outside word vector -> (embedding_dim, )
        u_o = self._mat_outside[outside_idx]
        # get center word vector -> (embedding_dim, )
        v_c = self._mat_center[center_idx]
        # get softmax probability -> (1, vocab_size)
        p_i = self.softmax(np.dot(self._mat_outside, v_c.reshape(-1, 1)).T)
        # get negative log likelihood loss (NLL Loss)
        loss = -np.log(p_i[0, outside_idx])  # scalar
        # get gradient wrt center embedding matrix
        grad_center = np.zeros((self._vocab_size, self._embedding_dim))
        grad_center[center_idx, :] = -u_o + np.dot(p_i, self._mat_outside).squeeze()
        # get gradient wrt outside embedding matrix
        p_i[0, outside_idx] -= 1
        grad_outside = np.outer(p_i.squeeze(), v_c)
        # get gradient wrt model weights -> (2*vocab_size, embedding_dim)
        grad_weights = np.concatenate((grad_center, grad_outside), axis=0)
        return loss, grad_weights

    def _get_nce_loss_and_grad(self, center_idx, outside_idx):
        num_negative = 10
        negative_indices = []

        # negative sampling
        while len(negative_indices) < num_negative:
            new_idx = random.randint(0, self._vocab_size - 1)
            if new_idx != outside_idx:
                negative_indices.append(new_idx)
        indices = [outside_idx] + negative_indices

        u_o = self._mat_outside[outside_idx]  # (d, )
        U_k = self._mat_outside[negative_indices]  # (k, d)
        v_c = self._mat_center[center_idx]
        prob_indices = np.hstack(
            [self.sigmoid(np.dot(u_o, v_c)), self.sigmoid(np.dot(U_k, v_c))]
        )

        p = np.zeros(shape=(1, self._vocab_size))
        for i, idx in enumerate(indices):
            p[0, idx] += prob_indices[i]
        p[0, outside_idx] -= 1

        # LOSS
        loss = (
            -np.log(self.sigmoid(np.dot(u_o, v_c)))
            - np.log(self.sigmoid(-np.dot(U_k, v_c))).sum()
        )
        grad_center = np.zeros((self._vocab_size, self._embedding_dim))
        grad_center[center_idx, :] = np.dot(p, self._mat_outside).squeeze()
        grad_outside = np.outer(p.squeeze(), v_c)
        grad_weights = np.concatenate((grad_center, grad_outside), axis=0)
        return loss, grad_weights

    def _train(self, corpus, bs=128, alpha=0.25, max_iter=5000):
        # check folder existence for storing model weights
        check_create_dir("./data")
        max_window_size = 8

        def train_one_batch(cur_alpha):
            loss = 0
            grad = np.zeros(self.model_weights.shape)

            for _ in range(bs):
                window = random.randint(1, max_window_size)
                center_word, context = corpus.get_random_context(window)

                for outside_word in context:
                    center_idx = corpus.word_to_idx[center_word]
                    outside_idx = corpus.word_to_idx[outside_word]
                    if self.loss_type == "nll":
                        this_loss, this_grad = self._get_nll_loss_and_grad(
                            center_idx, outside_idx
                        )
                    else:
                        this_loss, this_grad = self._get_nce_loss_and_grad(
                            center_idx, outside_idx
                        )
                    loss += this_loss / bs
                    grad += this_grad / bs

            # update weights (sgd)
            self._mat_center -= cur_alpha * grad[: self._vocab_size, :]
            self._mat_outside -= cur_alpha * grad[self._vocab_size :, :]
            return loss

        # training
        print_interval = 10
        save_interval = 500
        anneal_interval = 1000

        loss_history = []
        start_time = time()
        cur_alpha = alpha

        for i in range(1, max_iter):
            cur_loss = train_one_batch(cur_alpha)
            loss_history.append(cur_loss)

            if (i % print_interval) == 0:
                duration = (time() - start_time) / 60.0
                logger.info(
                    "Iteration: {} | Loss: {:.2f} | Elapsed Time: {:.1f} mins".format(
                        i, cur_loss, duration
                    )
                )
                # plot losses
                plt.plot(loss_history)
                plt.xlabel("Batch Number")
                plt.ylabel("Loss")
                plt.title("Loss History")
                plt.savefig(
                    "./data/train_history_word2vec_{}.png".format(self.loss_type),
                    dpi=200,
                )
                plt.clf()

            if (i % save_interval) == 0:
                logger.info("saving model weights...")
                file_path = "./data/word2vec_{}.npy".format(self.loss_type)
                np.save(file_path, self.model_weights)

            if (i % anneal_interval) == 0:
                cur_alpha *= 0.5
                logger.info("decreasing the learning rate to {:.4f}".format(cur_alpha))
        return loss_history

    def create_embeddings(self, corpus):
        self._vocab_size = corpus.vocab_size
        self._word_to_idx = corpus.word_to_idx
        self._init_embeddings()
        losses = self._train(corpus)

        plt.plot(losses)
        plt.xlabel("Batch Number")
        plt.ylabel("Loss")
        plt.title("Loss History")
        plt.savefig(
            "./data/train_history_word2vec_{}.png".format(self.loss_type), dpi=200
        )
        plt.clf()

    def get_embeddings(self, words):
        indices = [self._word_to_idx[word] for word in words]
        word_vectors = self._mat_center[indices, :]
        return word_vectors


class Word2VecTorch(IWordEmbedding):
    pass


if __name__ == "__main__":
    df = pd.read_pickle("./data/processed_dataset.pkl")
    corpus = ArticleCorpus.from_dataframe(df, "abstract")
    word2vec = Word2VecBase(loss_type="nce")
    word2vec.create_embeddings(corpus)
    print(
        np.dot(
            word2vec.get_embeddings(["micromorphic"])[0],
            word2vec.get_embeddings(["continuum"])[0],
        )
    )

