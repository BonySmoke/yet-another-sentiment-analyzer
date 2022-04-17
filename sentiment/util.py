from nltk.corpus import stopwords

def update_stopwords(new_words=[]):
    '''
    Add new words to the stopwords list
    '''
    stop_words = stopwords.words('english')
    stop_words.extend(new_words)
    return stop_words
