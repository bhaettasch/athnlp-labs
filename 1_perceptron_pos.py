import random
import abc
from copy import deepcopy

import math
from nltk import ConditionalFreqDist
from tqdm import tqdm

from athnlp.readers.brown_pos_corpus import BrownPosTag

MAX_EPOCHS_WITHOUT_IMPROVEMENTS = 3


class POSTagger:
    def __init__(self):
        self.name = "POS Tagger"
        self.corpus = corpus
        self.pos_tags = corpus.dictionary.y_dict.names
        self.vocab_size = len(corpus.dictionary.x_dict)

    @abc.abstractmethod
    def _get_word_feature(self, word, sent):
        """
        Encode the given word (using the sent information if applicable)

        :param word: word to encode
        :type word: str
        :param sent: context (sent the word is from)
        :type sent: Sequence
        :return: encoded word
        :rtype: Any
        """
        pass

    @abc.abstractmethod
    def _get_prediction(self, word):
        """
        Get a prediction for the given word

        :param word: word to get tag for
        :type word: str
        :return: tag prediction
        :rtype: str
        """
        pass

    @abc.abstractmethod
    def train(self):
        """
        Train tagger model on the given corpus
        """
        pass

    def evaluate(self):
        """
        Evaluate the tagger
        :return: list of dataset-acurracy-tuples
        :rtype: list[(str, float)]
        """
        accuracies = []

        # Evaluate all data splits
        for (split_name, split_data) in [("Train", self.corpus.train), ("Dev", self.corpus.dev), ("Test", self.corpus.test)]:
            # Loop over all sentences and all words in it and count the share of correct predictions
            total = 0
            correct = 0
            for sent in tqdm(split_data, desc=f"Evaluating {split_name}"):
                for (word, gold_tag) in sent.get_tag_word_tuples():
                    total += 1
                    correct += self._get_prediction(self._get_word_feature(word, sent)) == gold_tag
            accuracies.append((split_name, correct / total * 100))

        accuracies_string = ' '.join(f"{name}: {acc:3.2f}%" for name, acc in accuracies)
        print(f"Accuracies:\n{accuracies_string}")
        return accuracies


class MajorityClassPOSTagger(POSTagger):

    def __init__(self, corpus):
        super().__init__()
        self.name = "Majority Class POS Tagger"
        self.word_pos_cfd = None

    def train(self):
        """
        This trains a simple baseline which just uses majority class voting for every word in vocabulary
        disregarding of its context
        """
        self.word_pos_cfd = ConditionalFreqDist(tp for seq_list in self.corpus.train for tp in seq_list.get_tag_word_tuples())

    def _get_word_feature(self, word, sent):
        return word

    def _get_prediction(self, word):
        if word in self.word_pos_cfd:
            return self.word_pos_cfd[word].most_common(1)[0][0]
        return ""


class UnigramPerceptronPOSTagger(POSTagger):

    def __init__(self, corpus):
        """
        Constructor

        :param corpus: the corpus to train and evaluate on
        :type corpus: BrownPosTag
        """
        super().__init__()
        self.name = "Unigram Perceptron POS Tagger"
        self.weights_per_label = {}
        self.weights_per_label_best = {}

    def train(self):
        """
        Train POS Tagger
        """

        # Init weight list/vector (no weights for anything at first)
        for tag in self.corpus.dictionary.y_dict.names:
            self.weights_per_label[tag] = {}

        print("Training perceptron")
        mistakes_best = math.inf
        epochs_without_improvements = 0
        epoch = 0
        # Train until no improvements happen anymore
        while epochs_without_improvements < MAX_EPOCHS_WITHOUT_IMPROVEMENTS:
            # Multiple epochs are only useful when you change something (shuffle the learning order)
            random.shuffle(self.corpus.train)
            epoch += 1
            mistakes = 0
            count_examples = 0

            # Loop over all examples and learn from them
            for sent in tqdm(self.corpus.train, desc=f"Epoch {epoch}"):
                for (word, tag) in sent.get_tag_word_tuples():
                    count_examples += 1
                    feature = self._get_word_feature(word, sent)
                    # Get prediction
                    pred_tag = self._get_prediction(feature)
                    # If there was a misprediction
                    if pred_tag != tag:
                        # Update both affected weights
                        for f in feature:
                            self.weights_per_label[tag][f] = self.weights_per_label[tag].get(f, 0) + 1
                            self.weights_per_label[pred_tag][f] = self.weights_per_label[tag].get(f, 0) - 1
                        mistakes += 1

            # Break condition based upon last changes to the score
            print(f"Error: {(mistakes/count_examples*100):3.2f} %")
            if mistakes < mistakes_best:
                self.weights_per_label_best = deepcopy(self.weights_per_label)
                epochs_without_improvements = 0
                mistakes_best = mistakes
            else:
                epochs_without_improvements += 1

            self.weights_per_label = self.weights_per_label_best

    def _get_word_feature(self, word, sent):
        return [word]

    def _get_prediction(self, feature):
        best_label = ""
        best_score = -1

        for (label, weights) in self.weights_per_label.items():
            score = sum(weights.get(f, 0) for f in feature)
            if score > best_score:
                best_label = label
                best_score = score

        return best_label


if __name__ == "__main__":
    print("Loading corpus")
    corpus = BrownPosTag()

    print("Testing models")
    for tagger_type in [MajorityClassPOSTagger, UnigramPerceptronPOSTagger]:
        tagger = tagger_type(corpus)
        print(f"\n\n==={tagger.name}===\n")
        tagger.train()
        tagger.evaluate()
