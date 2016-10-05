from __future__ import division

import math
import os
import re
import nltk

from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords


# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

# Path to dataset
PATH_TO_DATA = './large_movie_review_dataset'
# e.g. "/users/brendano/inlp/hw1/large_movie_review_dataset"
# or r"c:\path\to\large_movie_review_dataset", etc.
TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
TEST_DIR = os.path.join(PATH_TO_DATA, "test")

senti_words = {}
#token_regex = re.compile(ur'[0-9]+\/[0-9]+|\w+|#[a-zA-Z0-9_]+|:[OoDdSsPp3\)\(\|/\$\_]+|-\.-|-_-|\(8[\)]?|;\)|\u2665|[\uD800-\uDBFF][\uDC00-\uDFFF]|<3|\w+\'\w+|\w+', re.U|re.I)
token_regex = re.compile(ur'[0-9]+\/[0-9]+|\w+|#[a-zA-Z0-9_]+|:[OoDdSsPp3\)\(\|/\$\_]+|-\.-|-_-|\(8[\)]?|;\)|=[\(\)]|:-[\(\)]|<3', re.U|re.I)
tokenizer = nltk.RegexpTokenizer(token_regex)

contractions_dict = {
"amn't":"am not",
"aren't":"are not",
"can't":"cannot",
"could've":"could have",
"couldn't":"could not",
"couldn't've":"could not have",
"didn't":"did not",
"doesn't":"does not",
"don't":"do not",
"gonna":"going to",
"gotta":"got to",
"hadn't":"had not",
"hadn't've":"he had not have",
"hasn't":"has not",
"hasn't've":"has not have",
"haven't":"have not",
"he'd've":"he would have",
"he'll":"he will",
"how'll":"how will",
"i'd've":"i would have",
"i'll":"i will",
"i'm":"i am",
"i've":"i have",
"i'ven't":"i have not",
"isn't":"is not",
"it'd've":"it would have",
"it'll":"it will",
"it's":"it is",
"It's":"it is",
"let's":"let us",
"ma'am":"madam",
"mightn't":"might not",
"mightn't've":"might not have",
"might've":"might have",
"mustn't":"must not",
"mustn't've":"must not have",
"must've":"must have",
"needn't":"need not",
"not've":"not have",
"o'clock":"of the clock",
"ol'":"old",
"oughtn't":"ought not",
"oughtn't've":"ought not to have",
"shan't":"shall not",
"she'd've":"she would have",
"she'll":"she will",
"should've":"should have",
"shouldn't":"should not",
"shouldn't've":"should not have",
"somebody'd've":"somebody would have",
"somebody'll":"somebody will",
"someone'd've":"someone would have",
"someone'll":"someone will",
"something'd've":"something would have",
"something'll":"something will",
"'sup":"what's up",
"that'll":"that will",
"there'd've":"there would have",
"there're":"there are",
"they'd've":"they would have",
"they'd'ven't":"they would have not",
"they'll": "they will",
"they'lln't've":"they will not have",
"they'll'ven't":"they will have not",
"they're":"they are",
"they've":"they have",
"they'ven't":"they have not",
"'tis":"it is",
"'twas":"it was",
"wanna":"want to",
"wasn't":"was not",
"we'd've":"we would have",
"we'll":"we will",
"we'lln't've":"we will not have",
"we're":"we are",
"we've":"we have",
"weren't":"were not",
"what'd":"what did",
"what'll":"what will",
"what're":"what are",
"what've":"what have",
"where'd":"where did",
"where've":"where have",
"who'd've":"who would have",
"who'll":"who will",
"who're":"who are",
"who've":"who have",
"why'd":"why did",
"why'll":"why will",
"why're":"why are",
"won't":"will not",
"won't've":"will not have",
"would've":"would have",
"wouldn't":"would not",
"wouldn't've":"would not have",
"y'all":"you all, literally",
"y'all'd've":"you all would have",
"y'all'dn't've":"you all would not have",
"y'all'll":"you all will",
"y'all'on't":"you all will not",
"y'all'll've":"you all will have",
"y'all're":"you all are",
"y'all'll'ven't":"you all will have not",
"you'd've":"you would have",
"you'll":"you will",
"you're":"you are",
"you'ren't":"you are not",
"you've":"you have",
"you'ven't":"you have not"
}

pos_emo = [':)', ':))', ':)))', ':))))', ':)))))', ';)', ':d', ':p', '=)', '<3', ':-)']
neg_emo = [':(', '=(', ':-(']

pos_rating = ['6/10', '7/10', '8/10', '9/10', '10/10', '5/5', '4/5', '3/5']
neg_rating = ['0/10', '1/10', '2/10', '3/10', '4/10', '1/5', '2/5', '0/5']


contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def normalize(text):
    text = expand_contractions(text)
    return text.lower().replace("<br />", "") ## Remove html break lines


def get_senti_words():
    if not senti_words:
        with open('subjclueslen1-HLTEMNLP05.tff', 'r') as doc:
            content = doc.readlines()
        for s in content:
            word = None
            sentiment = None
            for v in s.split():
                c = v.split('=')
                if c[0] == 'word1':
                    word = c[1]
                if c[0] == 'priorpolarity':
                    sentiment = c[1]
            if word and sentiment:
                senti_words[word] = sentiment
    return senti_words


def tokenize_doc(doc):
    
    english_stopwords = stopwords.words('english')
    stemmer = PorterStemmer()
    senti_words = get_senti_words()
    bow = defaultdict(float)
    lowered_tokens = tokenizer.tokenize(doc.lower().replace('<br />', ''))
    for token in lowered_tokens:
        if token not in english_stopwords: 
            if token in senti_words:
                sentiment = senti_words[token]
                if sentiment == 'negative':
                    bow['negative_sentiment'] += 1.0
                if sentiment == 'positive':
                    bow['positive_sentiment'] += 1.0
            if token in pos_emo:
                bow['pos_emo'] += 1.0
            if token in neg_emo:
                bow['neg_emo'] += 1.0
            if token in pos_rating:
                bow['pos_rating'] += 1.0
            if token in neg_rating:
                bow['neg_rating'] += 1.0
            bow[token] += 1.0
    return bow

class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self):
        
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }


    def train_model(self, training_data):

        for label in [ POS_LABEL, NEG_LABEL ]:
            for bow in training_data[label]:
                self.update_model(bow, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print "REPORTING CORPUS STATISTICS"
        print "NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL]
        print "NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL]
        print "NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL]
        print "NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL]
        print "VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab)

    def update_model(self, bow, label):
        """
        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each class (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each class (self.class_total_doc_counts)
        """
        for w in bow:
            self.class_word_counts[label][w] += 1
            self.class_total_word_counts[label] +=1
            self.vocab.add(w)
        self.class_total_doc_counts[label] += 1


    def tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """

        bow = tokenize_doc(doc)
        self.update_model(bow, label)

    def top_n(self, label, n):
        """

        Returns the most frequent n tokens for documents with class 'label'.
        """
        return sorted(self.class_word_counts[label].items(), key=lambda (w,c): -c)[:n]

    def p_word_given_label(self, word, label):
        """
        Returns the probability of word given label (i.e., P(word|label))
        according to this NB model.
        """
        return self.class_word_counts[label][word] / self.class_total_word_counts[label]
    
    def p_word_given_label_and_psuedocount(self, word, label, alpha):
        """
        Returns the probability of word given label wrt psuedo counts.
        alpha - psuedocount parameter
        """
        return (self.class_word_counts[label][word] + alpha) /\
         (self.class_total_word_counts[label] + (alpha * len(self.vocab)))

    def log_likelihood(self, bow, label, alpha):
        """
        Computes the log likelihood of a set of words give a label and psuedocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; psuedocount parameter
        """
        return sum(math.log(self.p_word_given_label_and_psuedocount(w, label, alpha)) for w in bow)

    def log_prior(self, label):
        """
        Implement me!

        Returns a float representing the fraction of training documents
        that are of class 'label'.
        """
        return math.log(self.class_total_doc_counts[label] / sum(self.class_total_doc_counts.values()))

    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        alpha - psuedocount parameter
        bow - a bag of words (i.e., a tokenized document)
        Computes the unnormalized log posterior (of doc being of class 'label').
        """
        return self.log_prior(label) + self.log_likelihood(bow, label, alpha)

    def classify(self, bow, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        bow - a bag of words (i.e., a tokenized document)

        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior).
        """
        pos_pos = self.unnormalized_log_posterior(bow, POS_LABEL, alpha)
        neg_pos = self.unnormalized_log_posterior(bow, NEG_LABEL, alpha)

        if pos_pos > neg_pos:
            return POS_LABEL
        else:
            return NEG_LABEL


    def likelihood_ratio(self, word, alpha):
        """
        alpha - psuedocount parameter.
        Returns the ratio of P(word|pos) to P(word|neg).
        """
        return self.p_word_given_label_and_psuedocount(word, POS_LABEL, alpha) /\
         self.p_word_given_label_and_psuedocount(word, NEG_LABEL, alpha)

    def evaluate_classifier_accuracy(self, alpha, test_data):
        """
        alpha - psuedocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        confusion_matrix = { POS_LABEL: defaultdict(int),
                            NEG_LABEL: defaultdict(int) }

        with open('wrong_classifications.csv', 'w') as wrong_doc:
            for label in [ POS_LABEL, NEG_LABEL ]:
                for bow in test_data[label]:
                    predicted_label = self.classify(bow, alpha)
                    confusion_matrix[label][predicted_label] += 1
                    if(predicted_label != label):
                        wrong_doc.write(str(bow))
                        wrong_doc.write('\n')
                        wrong_doc.write(predicted_label)
                        wrong_doc.write('\n')
                        
        true_positive = confusion_matrix[POS_LABEL][POS_LABEL]
        true_negative = confusion_matrix[NEG_LABEL][NEG_LABEL]
        false_positive = confusion_matrix[NEG_LABEL][POS_LABEL]
        false_negative = confusion_matrix[POS_LABEL][NEG_LABEL]

        return ((true_positive + true_negative) \
            / (true_positive + false_positive + true_negative + false_negative))

def produce_hw1_results():
    # PRELIMINARIES

    # QUESTION 1.1
    # uncomment the next two lines when ready to answer question 1.2
    print "VOCABULARY SIZE: " + str(len(nb.vocab))
    print ''

    # QUESTION 1.2
    # uncomment the next set of lines when ready to answer qeuestion 1.2
    print "TOP 10 WORDS FOR CLASS " + POS_LABEL + " :"
    for tok, count in nb.top_n(POS_LABEL, 10):
        print '', tok, count
    print ''

    # print "TOP 10 WORDS FOR CLASS " + NEG_LABEL + " :"
    for tok, count in nb.top_n(NEG_LABEL, 10):
       print '', tok, count
    print ''
    print '[done.]'

def plot_psuedocount_vs_accuracy(psuedocounts, accuracies):
    """
    A function to plot psuedocounts vs. accuries. You may want to modify this function
    to enhance your plot.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12,8))
    plt.plot(psuedocounts, accuracies)
    plt.xlabel('Psuedocount Parameter')
    plt.ylabel('Accuracy (%)')
    plt.title('Psuedocount Parameter vs. Accuracy Experiment')
    plt.xticks(psuedocounts)
    plt.savefig('psuedocount_vs_accuracy.pdf')

def evaluate_pseudocounts():
    psuedocounts = [x * 0.5 for x in range(1, 21)]
    accuracies = []
    for c in psuedocounts:
        accuracies.append(nb.evaluate_classifier_accuracy(c))
    plot_psuedocount_vs_accuracy(psuedocounts, accuracies)

def read_data(path):
    data = {
        POS_LABEL: [],
        NEG_LABEL: []
    }
    pos_path = os.path.join(path, POS_LABEL)
    neg_path = os.path.join(path, NEG_LABEL)
    print "Starting training with paths %s and %s" % (pos_path, neg_path)
    for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
        filenames = os.listdir(p)
        for f in filenames:
            with open(os.path.join(p,f),'r') as doc:
                content = doc.read()
                bow = tokenize_doc(content)
                data[label].append(bow)
    return data 

mean = lambda values: sum(values) / float(len(values))

def cross_validation(training, num_folds):

    subset_size = int(len(training[POS_LABEL])/num_folds)
    test_data = {
        POS_LABEL: [],
        NEG_LABEL: []
    }
    training_data = {
        POS_LABEL: [],
        NEG_LABEL: []
    }
    accuracies = []
    for i in range(num_folds):
        for label in [POS_LABEL, NEG_LABEL]:
            test_data[label] = training[label][i*subset_size:][:subset_size]
            test_data[label] = training[label][i*subset_size:][:subset_size]
            
            training_data[label] = training[label][:i*subset_size] + training[label][(i+1)*subset_size:]
            training_data[label] = training[label][:i*subset_size] + training[label][(i+1)*subset_size:]

        nb = NaiveBayes()
        nb.train_model(training_data)
        accuracy = nb.evaluate_classifier_accuracy(1, test_data)
        accuracies.append(accuracy)
    return mean(accuracies)  

if __name__ == '__main__':
    training_data = read_data(TRAIN_DIR)
    print "Training data", len(training_data[POS_LABEL])
    test_data = read_data(TEST_DIR)
    print "Testing data", len(test_data[POS_LABEL])
    
    nb = NaiveBayes()
    nb.train_model(training_data)
    train_accurary = cross_validation(training_data, 10)
    print "Train Accurary", train_accurary
    accuracy = nb.evaluate_classifier_accuracy(1, test_data)
    print "Test Accuracy", accuracy

