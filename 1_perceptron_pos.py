"""
input: labeled data D
initialize w(0) = 0
initialize k = 0 (number of mistakes)
repeat
get new training example (xi ; yi ) 2 D
predict byi = argmaxy2Y w(k)  (xi ; y)
if byi 6= yi then
update w(k+1) = w(k) + (xi ; yi ) 􀀀 (xi ; byi )
increment k
end if
until maximum number of epochs
output: model weights w
"""
from nltk import ConditionalFreqDist

from athnlp.readers.brown_pos_corpus import BrownPosTag


class PerceptronPOSTagger:

    def __init__(self, corpus):
        """
        Constructor

        :param corpus: the corpus to train and evaluate on
        :type corpus: BrownPosTag
        """
        self.corpus = corpus
        self.pos_tags = corpus.dictionary.y_dict.names
        self.baseline_cfd = None

    def _get_word_representation(self, word):
        """
        Get binary number/vector representation for word

        :param word: word to represent
        :type word: str
        :return: word representation suitable for further processing
        :rtype: ?
        """
        # TODO Tranfer into encoded format instead of int
        return self.corpus.dictionary.y_dict.get_label_id(word)

    def train(self):
        pass

    def evaluate(self):
        pass

    def train_baseline(self):
        """
        This trains a simple baseline which just uses majority class voting for every word in vocabulary
        disregarding of its context

        :return:
        :rtype:
        """
        # sample_sent = [('Merger', 'noun'), ('proposed', 'verb')]
        self.baseline_cfd = ConditionalFreqDist(tp for seq_list in self.corpus.train for tp in seq_list.get_tag_word_tuples())

    def _get_tag_for_word_baseline(self, word):
        return

    def evaluate_baseline(self):
        """
        Evaluate accuracy of the baseline
        """
        total = 0
        correct = 0
        for sent in self.corpus.dev:
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
