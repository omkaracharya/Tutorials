from nltk.corpus import stopwords, state_union
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.stem import PorterStemmer
from nltk import pos_tag

# example_sentence = "This is an example showing off stop words filteration."
# example_words = ["Python", "pythoner", "pytHoning", "pythoning", "Pythoned", "PYTHONLY"]
# stop_words = set(stopwords.words("English"))
# # print set(word_tokenize(example_sentence)) - stop_words
#
# ps = PorterStemmer()
# stems = [ps.stem(word) for word in word_tokenize(example_words)]
# print stems

train_text = state_union.raw("2005-GWBush.txt")
test_text = state_union.raw("2006-GWBush.txt")
pst = PunktSentenceTokenizer(train_text)
tokenized = pst.tokenize(test_text)


def process_content():
    try:
        for i in tokenized:
            words = word_tokenize(i)
            tagged = pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))

process_content()