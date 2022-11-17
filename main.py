import nltk
import string
from nltk.tokenize import RegexpTokenizer
import matplotlib as mpl
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

stop_words = nltk.corpus.stopwords.words('english')
words = nltk.corpus.gutenberg.words()
words_without_sw = []
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
punctuation = string.punctuation

for word in words:
    if word not in stop_words and word not in punctuation:
        # print(f"{word} : {lemmatizer.lemmatize(word)} : {stemmer.stem(lemmatizer.lemmatize(word))}")
        words_without_sw.append(lemmatizer.lemmatize(stemmer.stem(word)))

freq = nltk.probability.FreqDist(words_without_sw)

def document_features(document):
    document_words = set(document)
    features = {}
    for word in words_without_sw:
        features['contains({})'.format(word)] = (word in document_words)
    return features

words_test = nltk.corpus.brown.words()
print(document_features(words_test))
featuresets = [(document_features(d), c) for (d,c) in words_test]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
#print(freq.most_common(10))
#freq.plot(30)
