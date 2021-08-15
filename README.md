# Linguipedia - Codefest'19
## Music Album Review Prediction

To build an intelligent system and determine the score of reviews given by all the music album listeners.

Assign a value in the range of 0 to 4 according to audience reviews.

To train an RNN based LSTM architecture that with the help of NLP achieves the desired task of classifying reviews.

## Dataset Description

Train Dataset
	
Train dataset contains 35,833 reviews each having following 3 attributes (id,review,score).

Test dataset has 17,650 reviews

## Data Preprocessing 

Before applying the actual logic on the given review, it is necessary to preprocess the given string so that we can avoid unnecessary errors and to achieve that, we will apply the basic steps given below-

Converting uppercase to lowercase
Tokenization
Lemmatization
Stopword Removal

## First step is to Convert Uppercase to Lowercase

## Tokenization
The next step in pre-processing is to generate the tokens from the given review and it is performed by applying tokenization on the given review string. 

By this, the review is splitted into words, and punctuation characters like space and comma are removed.

Tokenization is of two types- Word Tokenize and Sentence Tokenize. 

A RegexpTokenizer uses a regular expression to break a string into substrings.

## Lemmatization
Lemmatization is the process of grouping together the different forms of similar word so they can be analysed as a single item.
For ex- the word  better  is derived from the word  good , and hence, the lemma is  good.
The root word is called a  lemma  in the lemmatization process.
To do so, it is necessary to have detailed dictionaries which the algorithm can look through to link the form back to its lemma.

## Stopword Removal
In this step all the frequently occurring words like is, a, an, the, for etc. are removed from the given review.

By removing these words we remove the low-level information present in the text, so that we can focus on the important ones.

## Word Embedding

Word embedding is used for the representation of words for text analysis, usually in the form of a real-valued vector that encodes the meaning of the word and predicts that words that are near in the vector space will have similar meanings.

The model contains 300-dimensional vectors for 3-million words.

Dataset used for training is Google News Dataset.

## Word2Vec
Word2vec is a two-layer neural network that “vectorizes” words to process text. It takes a text corpus as input and produces a series of vectors in the form of feature vectors that describe terms in the corpus.

Although Word2vec is not a deep neural network, it does convert text to a numerical format that deep neural networks can comprehend.
Word2vec's aim and utility is to group vectors with identical terms together in vector space. That is, it uses mathematics to detect correlations.

## Modelling 

We have gone for Lstm model
Problem in bag of words [0,1,0,1,....] , tfidf [0.2,0.6,0.1...] 
Sequence information is discarded . So ,semantic of the sentence was not maintained and long term dependencies are not maintained.
As they are stateless models .

## Sequential Model

RNN  sequence model - by remembering sequence
We see sentence from left to right (each word by word)
We maintain state -> state vector(memory/context).

## Problems In Simple Recurrent Neural Network

Vanishing gradient problem - Changes in weights is negligible so we do not converge to Global minima (in view point of loss).
Exploding grading problem - Changes in weights are very so we do not converge to Global minima (in view point of loss).
Multiplicative gradient  can be exponentially decreasing/increasing with respect to the number of layers.

## LSTM are used for catching long term dependencies.
Input gate - add new information that is useful to predict next few words
Memory cell -  is used to pass on the information
Forget gate -  helps to forget un-useful information
Output gate - which extract the useful information from the memory cell

## Categorical Cross Entropy   
It is a Softmax activation plus a Cross-Entropy loss
Cross entropy -  distance between those two probability (softmax, one hot encoded) vectors is called the Cross Entropy. 
In categorical cross entropy graph  if the prediction is bad so the loss is high. Correspondingly weights change in the matrix will be high.
Residual (SSR) : sum of squares due to regression

Then we are going to compare those probabilities to the one-hot encoded labels using the cross entropy function. 
In categorical cross entropy, graph is lograthmic and if the loss is high the weights change in the matrix will be high.

## Optimizer 
An optimizer is a method or algorithm to update the various parameters that can reduce the loss in much less effort.
Adam optimizer - best for multiclass classification.
Here we want to use categorical cross-entropy as we have got a multiclass classification problem and the Adam optimizer, which is the most commonly used optimizer.


## One hot encoded  
We see output corresponding to each class . So that loss can be easily calculated.
Eg. Suppose there are 3 classifiers dog,cat & snake. Then one hot encoding for snake will be [1,0,0] , for dog it will be [0,1,0] and for snake it will be [0,0,1].

## Pandas 
Pandas is software library written for python. It is mainly used for data manipulation and data analysis.

## NumPy  
NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

## NLTK 
NLTK is a Python library that work with human language data and can be used for statistical natural language processing (NLP). It includes tokenizatio stopword word removal,stemming, lemmatisation and semantic reasoning text processing libraries.

## Gensim 
Gensim is an open-source library for unsupervised topic modeling and natural language processing, using modern statistical machine learning.

## Keras 
Keras is an open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.


