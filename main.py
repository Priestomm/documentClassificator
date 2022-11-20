import random
import math
import collections
import nltk
import string
from nltk.metrics import ConfusionMatrix
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import EuroparlCorpusReader

languages = ['english', 'italian', 'spanish', 'dutch', 'german', 'portuguese', 'finnish', 'swedish', 'greek']
numbers = []
for number in range(10000):
    numbers.append(str(number))
stop_words = list(nltk.corpus.stopwords.words(lang) for lang in languages)
punctuation = string.punctuation

root = '/Users/tommasodelprete/nltk_data/corpora/europarl_raw/'

english = EuroparlCorpusReader(root, '.*\.en')
documents = list()
for fileid in english.fileids():
    for sentence in english.raw(fileid).split('\n'):
        documents.append((sentence, 'eng'))

print(len(documents))
prefToLang = {'da':'danish', 'nl':'dutch', 'fi':'finnish', 'de':'german', 'fr':'french', 'it':'italian', 'pt':'portuguese', 'el':'greek', 'es':'spanish', 'sv':'swedish'}

for lang in prefToLang.keys():
    non_english = EuroparlCorpusReader(root, ".*\.{}".format(lang))
    for sentence in non_english.raw(f'{prefToLang[lang]}/ep-00-01-17.{lang}').split('\n'):
        documents.append((sentence, 'noteng'))

print(len(documents))
random.shuffle(documents)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = list()
for document in documents:
    for word in word_tokenize(document[0]):
        words.append(word)
words_without_sw = []

for word in words:
    if word not in stop_words and word not in punctuation and word not in numbers:
        # print(f"{word} : {lemmatizer.lemmatize(word)} : {stemmer.stem(lemmatizer.lemmatize(word))}")
        words_without_sw.append(lemmatizer.lemmatize(stemmer.stem(word)))

freq = nltk.probability.FreqDist(words_without_sw)# BOW
word_feature = list(freq)[:5000]

def document_features(document):
    document_words = word_tokenize(document)
    features = {}
    for word in word_feature:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in documents]

print(len(featuresets))
train_set, test_set = featuresets[math.floor(len(featuresets)/2):], featuresets[:math.floor(len(featuresets)/2)]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(classifier.classify(document_features("mauro")))

refMat = list()
testMat = list()
for i, (feats, label) in enumerate(test_set):
    refMat.append('eng') if label == 'eng' else refMat.append('noteng')
    observed = classifier.classify(feats)
    testMat.append('eng') if observed == 'eng' else testMat.append('noteng')

cm = ConfusionMatrix(refMat, testMat)
print(cm.evaluate())