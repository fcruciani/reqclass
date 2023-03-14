from PROMISE_NFR_dataset_loader import PROMISE_NFR_dataset as Promise
#from nlp_datasets import CNN_1D_SimpleGLoveEncoder as CNNEncoder
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) #prevents trailing elipses
pd.set_option('display.max_rows', None)
#from keras.utils import to_categorical
#import RML_Classifiers as RML
#from RML_Classifiers import basepath
#MAX_SENTENCE_LENGTH = 1
#WORD_EMBEDDING_SIZE = 50

print("PREPROCESSING DATA")
promise = Promise()

promise_df, req_classes_df = promise.load_text()
print(promise_df.head(12))
print(req_classes_df.head(100))

#req_classes_df = req_classes_df.head(100)
print(len(promise_df))
all_classes = np.unique(req_classes_df['Label'].values)
print(all_classes)

words_per_class = {}
sentences_per_class = {}

#minimum number of words (used to define most frequent classes)
MIN_NUM_WORDS = 500
most_frequent_classes = []
all_reqs_count = 0
all_sentences_count = 0
all_words_count = 0
for label in all_classes:
    df = promise_df.loc[ promise_df['Label'] == label ]
    print(label,len(df), 'sentences')
    sentences_per_class[label] = len(df)
    all_sentences_count += len(df)
    words_per_class[label] = 0
    for i,row in df.iterrows():
        sentence = row['Sentence']
        num_words = len(sentence.split(' '))
        words_per_class[label] += num_words
        all_words_count += num_words
    if words_per_class[label] >= MIN_NUM_WORDS:
        most_frequent_classes.append(label)

print("all sentences: ",all_sentences_count)
print("all words: ",all_words_count)
promise_df=promise_df[promise_df['Label'].isin(most_frequent_classes)]
req_classes_df=req_classes_df[req_classes_df['Label'].isin(most_frequent_classes)]

print(sentences_per_class)
print(words_per_class)
print(most_frequent_classes)

all_classes = np.unique(req_classes_df['Label'].values)
print(all_classes)
input("pause")
req_indeces = req_classes_df['Req Number'].values 
req_classes = req_classes_df['Label'].values



from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit( np.unique(req_classes) )
y_indeces = le.transform(req_classes)
#print(y[:50])
for i in np.unique(req_classes):
    print(i, le.transform([i]))


for fold in ["1","2","3","4","5"]:


    from sklearn.model_selection import train_test_split
    #split 80 / 20 train - vld/test
    X_train_indeces, X_test_indeces,  y_train_cl, y_test_cl = train_test_split(req_indeces, y_indeces, 
                                                        stratify=y_indeces,
                                                        random_state=17*int(fold), 
                                                        test_size=0.3)

    y = le.transform(promise_df['Label'].values)
    promise_df = promise_df.assign(label_encoded=y)

    train_df = promise_df.iloc[X_train_indeces]
    test_df = promise_df.iloc[X_test_indeces]

    train_df = train_df.sort_index()
    test_df = test_df.sort_index()

    #test_df = test_df.assign(label_encoded=y_test_cl)
    print(train_df.head())
    print(test_df.head())

    print("Train #: ",len(train_df))
    print("Test #: ",len(test_df))

    #whole_df = pd.DataFrame(zipped, columns=[ 'Label', 'Req Number'])
    train_df.to_csv("./data/train_f"+fold+"_MajCla.csv",header=False,index=True)
    test_df.to_csv("./data/test_f"+fold+"_MajCla.csv",header=False,index=True)