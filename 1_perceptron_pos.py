import random

import numpy as np

from nltk import ConditionalFreqDist
from tqdm import tqdm

from athnlp.readers.brown_pos_corpus import BrownPosTag

EPOCHS = 3


class PerceptronPOSTagger:

    def __init__(self, corpus):
        """
        Constructor

        :param corpus: the corpus to train and evaluate on
        :type corpus: BrownPosTag
        """
        self.corpus = corpus
        self.pos_tags = corpus.dictionary.y_dict.names
        self.vocab_size = len(corpus.dictionary.x_dict)
        self.baseline_cfd = None
        self.weights_per_label = {}

    def train(self):
        """
        Train POS Tagger
        """

        # Init weight list/vector (no weights for anything at first)
        for tag in self.corpus.dictionary.y_dict.names:
            self.weights_per_label[tag] = {}

        print("Training perceptron")
        mistakes = 0
        for epoch in range(EPOCHS):
            random.shuffle(self.corpus.train)
            for sent in tqdm(self.corpus.train, desc=f"Epoch {epoch}"):
                for (word, tag) in sent.get_tag_word_tuples():
                    feature = self._get_word_feature(word)
                    # Get prediction
                    pred_tag = self._get_prediction(feature, self.weights_per_label)
                    # If there was a misprediction
                    if pred_tag != tag:
                        # Update both affected weights
                        for f in feature:
                            self.weights_per_label[tag][f] = self.weights_per_label[tag].get(f, 0) + 1
                            self.weights_per_label[pred_tag][f] = self.weights_per_label[tag].get(f, 0) - 1
                        mistakes += 1

    def _get_word_feature(self, word):
        return [word]

    def _get_prediction(self, feature, model_weights):
        best_label = ""
        best_score = -1

        for (label, weights) in model_weights.items():
            score = sum(weights.get(f, 0) for f in feature)
            if score > best_score:
                best_label = label
                best_score = score

        return best_label

    def evaluate(self):
        total = 0
        correct = 0
        for sent in tqdm(self.corpus.dev, desc="Evaluating"):
            for (word, gold_tag) in sent.get_tag_word_tuples():
                total += 1
                predicted_tag = self._get_prediction(self._get_word_feature(word), self.weights_per_label)
                if predicted_tag == gold_tag:
                    correct += 1

        baseline_accuracy = correct / total * 100
        print(f"Simple Perceptron Accuracy: {baseline_accuracy:3.2f}%")

    def train_baseline(self):
        """
        This trains a simple baseline which just uses majority class voting for every word in vocabulary
        disregarding of its context

        :return:
        :rtype:
        """
        self.baseline_cfd = ConditionalFreqDist(tp for seq_list in self.corpus.train for tp in seq_list.get_tag_word_tuples())

    def _get_tag_for_word_baseline(self, word):
        return

    def evaluate_baseline(self):
        """
        Evaluate accuracy of the baseline
        """
        total = 0
        correct = 0
        for sent in tqdm(self.corpus.dev, desc="Evaluating"):
            for (word, gold_tag) in sent.get_tag_word_tuples():
                total += 1
                # Known word?
                if word in self.baseline_cfd:
                    # Get majority class for this tag
                    predicted_tag = self.baseline_cfd[word].most_common(1)[0][0]
                    if predicted_tag == gold_tag:
                        correct += 1

        baseline_accuracy = correct / total * 100
        print(f"Baseline (Majority Class Voting) Accuracy: {baseline_accuracy:3.2f}%")


if __name__ == "__main__":
    corpus = BrownPosTag()
    pos_tagger = PerceptronPOSTagger(corpus)
    pos_tagger.train_baseline()
    pos_tagger.evaluate_baseline()
    pos_tagger.train()
    pos_tagger.evaluate()
