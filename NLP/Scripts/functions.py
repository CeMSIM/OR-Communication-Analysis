# -*- coding: utf-8 -*-
'''
Last updated: 11/10/2020
by Xinwen
Please also update the instruction txt.

'''
#text prepare code
def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    import nltk
    import re
    nltk.download('stopwords',quiet=True)
    from nltk.corpus import stopwords
    
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('',text)
    list_of_strings = [str(s) for s in STOPWORDS]
    text_split = text.split()
    for l1 in list_of_strings:
      for l2 in text_split:
        if l1==l2:
          text_split.remove(l2)
    text = " ".join(text_split)
    return text

#Classifier training code
def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    ovr = OneVsRestClassifier(LogisticRegression(max_iter=2000))
    ovr.fit(X_train, y_train)
    return ovr

#bag of words model code generate dictionary
def bag_of_words_model(text, dict_size):
    import collections
    words_counts = {}
    text = ' '.join(text)
    words_list = text.split()
    words_counts = collections.Counter(words_list)
    
    word_order_dict = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)#form tuple with order
    word_dict_list = [tup[0] for tup in word_order_dict]
    word_dict_list = word_dict_list[:dict_size] #list of words from high freq to low freq, with dict size
    WORDS_TO_INDEX = {val : idx for idx, val in enumerate(word_dict_list)}
    INDEX_TO_WORDS = {value:key for key, value in WORDS_TO_INDEX.items()}# you can choose to output this
    ALL_WORDS = WORDS_TO_INDEX.keys()# you can choose to output this

    return WORDS_TO_INDEX

# bag of words for the text
def bag_of_words_vector(text, words_to_index, dict_size):
    """
        text: a string without split
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    import numpy as np
    result_vector = np.zeros(dict_size,dtype=int)
    text = text.split()
    for word in text:
      idx = [val for key, val in words_to_index.items() if word in key]
      result_vector[idx] = int(1)
    return result_vector

# Output evaluation scores. Default return accuracy, f1 macro and precision_macro. 
# Other options avaliable: f1_micro, f1_weighted, precision_micro, precision_weighted
def evaluation_scores(y_val, predicted):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    accuracy = accuracy_score(y_val , predicted)
    
    F1_macro=f1_score(y_val, predicted, average='macro',zero_division=1)
    #F1_micro=f1_score(y_val, predicted, average='micro')
    #F1_weighted=f1_score(y_val, predicted, average='weighted')
    
    precision_macro=precision_score(y_val, predicted, average='macro',zero_division=1)
    #precision_micro=precision_score(y_val, predicted, average='micro')
    #precision_weighted=precision_score(y_val, predicted, average='weighted')
    
    return accuracy, F1_macro, precision_macro