import nltk
import numpy as np
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, PorterStemmer
#lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    #return lemmatizer.lemmatize(word.lower())
    return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentences, all_words):
    tokenized_sentences = [stem(w) for w in tokenized_sentences]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentences:
            bag[idx] = 1.0
            print(idx,w)
    return bag
