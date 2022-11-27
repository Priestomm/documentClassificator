# Assignment 1
## _The Naive Bayes Classifier_

The project implements a simple classifier, written in python, using the NLTK library.
The purpose is to recognize and classify english documents labeled as **eng** and non english texts, which are labeled as **noteng**.

## Pipeline
The classification process follows the pipeline listed below:
- **Import** of English and NotEnglish corpora
- **Segmentation** of the documents into paragraphs
- **Tokenization** of the paragraphs
- Removing of **stopwords**, **punctuation** and **numbers**
- **Stemming** of the tokenized words
- **Lemmatizatio**n of the lemmatized words
- Construction of the **Bag Of Words**
- **Train** and **test**

## Corpora used
I've chosen to use as training and testing set the *EuroParl* corpora, which contain extracts of speeches from the european parliament in different languages.

I used a total of 10 english corpora and 10 non english corpora, divided into paragraphs (the split character is '\n') and labeled as *eng* or *noteng*, hence for the training and testing, the set of sentences that the segmentation process produced was split into two subsets of equal dimensions.

## Statistics
- **Precision**: The number of true positives / true positives + false positives
- **Recall**: The number of trie positives / true positives + false negatives
- **F-Measure**: 2 * Precision * Recall / Precision + Recall
