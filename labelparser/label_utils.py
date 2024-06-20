import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

NON_ALPHANUM = re.compile('[^a-zA-Z]')
CAMEL_PATTERN_1 = re.compile('(.)([A-Z][a-z]+)')
CAMEL_PATTERN_2 = re.compile('([a-z0-9])([A-Z])')


def sanitize_label(label):
    # handle some special cases
    label = label.replace('\n', ' ').replace('\r', '')
    label = label.replace('(s)', 's')
    label = re.sub(' +', ' ', label)
    # turn any non alphanumeric characters into whitespace
    label = NON_ALPHANUM.sub(' ', label)
    label = label.strip()
    # remove single character parts
    label = " ".join([part for part in label.split() if len(part) > 1])
    # handle camel case
    label = _camel_to_white(label)
    # make all lower case
    label = label.lower()
    return label


def split_label(label):
    label = label.lower()
    result = re.split('[^a-zA-Z]', label)
    return result


def _camel_to_white(label):
    label = CAMEL_PATTERN_1.sub(r'\1 \2', label)
    return CAMEL_PATTERN_2.sub(r'\1 \2', label)


def split_and_lemmatize_label(label):
    words = split_label(label)
    lemmas = [lemmatize_word(w) for w in words]
    return lemmas


def lemmatize_word(word):
    lemma = lemmatizer.lemmatize(word, pos='v')
    lemma = re.sub('ise$', 'ize', lemma)
    return lemma
