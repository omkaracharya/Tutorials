import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
total_lines = 10000000


def create_lexicon(pos_file, neg_file):
    lexicon = []
    for file in [pos_file, neg_file]:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines[:total_lines]:
                words = word_tokenize(line.lower())
                lexicon += list(words)

    lexicon = [lemmatizer.lemmatize(lex) for lex in lexicon]
    word_counts = Counter(lexicon)
    # word_counts = {"word1":1000, "word2":500, ...}

    lexicon2 = []
    for word in word_counts:
        # No common words and no least appearing words
        if 1000 > word_counts[word] > 50:
            lexicon2.append(word)

    return lexicon2


def sample_handling(sample, lexicon, classification):
    data = []
    # [
    #     [[1,0,1,1,0],[0,1]],[[0,1,0,1,0],[1,0]],..
    # ]
    # [0, 1] - negative, [1, 0] - positive

    with open(sample, 'r') as f:
        lines = f.readlines()
        for line in lines[:total_lines]:
            words = word_tokenize(line.lower())
            words = [lemmatizer.lemmatize(word) for word in words]
            features = np.zeros(len(lexicon))
            for word in words:
                if word.lower() in lexicon:
                    index = lexicon.index(word.lower())
                    features[index] += 1
            features = list(features)
            data.append([features, classification])

    return data


def create_data(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    data = []
    data += sample_handling(pos, lexicon, [1, 0])
    data += sample_handling(neg, lexicon, [0, 1])
    random.shuffle(data)

    data = np.array(data)
    number_of_validation_instances = int(test_size * len(data))

    train_X = list(data[:, 0][:-number_of_validation_instances])
    train_Y = list(data[:, 1][:-number_of_validation_instances])
    validation_X = list(data[:, 0][-number_of_validation_instances:])
    validation_Y = list(data[:, 1][-number_of_validation_instances:])

    return train_X, validation_X, train_Y, validation_Y


if __name__ == '__main__':
    train_X, validation_X, train_Y, validation_Y = create_data('../data/pos.txt', '../data/neg.txt')
    with open('sentiment_data.pkl', 'wb') as f:
        pickle.dump([train_X, validation_X, train_Y, validation_Y], f)
