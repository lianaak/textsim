import os 
import numpy as np
import spacy 
import string


#following steps for preprocessing the data:
#1 lowercasing to help consistency in outputs
#2 tokenizing to break sentences into tokens
#3 lemmatizing to transform words to root form 

if __name__ == '__main__':
    print("preprocessing...")
else:
    #Microsoft Research Corpus - https://www.microsoft.com/en-us/download/details.aspx?id=52398
    train_data = open('./data/msr_paraphrase_train.txt', 'r', encoding='latin-1')
    unprocessed_train_data = np.array([example.split("\t") for example in train_data.readlines()])

    nlp = spacy.load('en_core_web_sm') # no need for large model

    sentence1_tokens = []
    sentence2_tokens = []

    punct = list(string.punctuation)

    count = 0
    for (sentence1,sentence2) in zip(unprocessed_train_data[:,3],unprocessed_train_data[:,4]):
        #error expected str, got numpy.str -> str()

        #remove punctuation
        for punctuation in punct:
            sentence1 = sentence1.replace(punctuation, '')
            sentence2 = sentence2.replace(punctuation, '')

        #convert to lowercase and tokenize 
        sentence1 = nlp.make_doc(str(sentence1.lower()))
        sentence2 = nlp.make_doc(str(sentence2.lower()))
        token_list1 = []
        token_list2 = []

        #lemmatize
        for token1,token2 in zip(sentence1,sentence2):
            token_list1.append(token1.lemma_)
            token_list2.append(token2.lemma_)
        sentence1_tokens.append(token_list1)
        sentence2_tokens.append(token_list2)

    processed_data = np.array([unprocessed_train_data[:,0],unprocessed_train_data[:,1],unprocessed_train_data[:,2],sentence1_tokens,sentence2_tokens])
    processed_data = processed_data.transpose()
    #sz = 0
    #for tokens in processed_data[4,:]:
    #    for token in tokens:
    #        print(token)
    #    sz += len(tokens)