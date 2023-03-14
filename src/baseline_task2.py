import numpy as np
import pandas as pd
from encoders import GloveEncoder, BERTEncoder, DistilBERTEncoder, SBERTEncoder, UniversalSentenceEncoder

import utils
from utils import get_encodings
from keras.utils import to_categorical

y_glove_all_pred = []
y_glove_all_true = [] 
y_bert_all_pred = []
y_bert_all_true = [] 
y_distibert_all_pred = []
y_distilbert_all_true = [] 
y_sbert_all_pred = []
y_sbert_all_true = [] 
y_use_all_pred = []
y_use_all_true = [] 

glove = GloveEncoder(WORD_EMBEDDING_SIZE=300)
bert = BERTEncoder()
distilbert = DistilBERTEncoder()
sbert = SBERTEncoder()
use = UniversalSentenceEncoder()

def train_test_model(train_df,test_df,encoder,name,WORD_EMBEDDING_SIZE):
    X_train, y_train_labels, train_req_num = get_encodings(train_df,encoder,WORD_EMBEDDING_SIZE)
    y_train_cl = y_train_labels
    #train
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier()
    neigh.fit(X_train, y_train_cl)
    #test
    names = [ 'F','LF','O','PE','SE','US']
    print(np.unique(test_df['Label'].values))
    print(names)
    y_req_cl  = []
    y_pred_cl_mean = []
    
    for index,row in test_df.iterrows():
        #print('Req',index)
        sentence = row['Sentence']
        x_vectors = encoder.encode_sentence(sentence)
        y_preds = neigh.predict_proba(x_vectors)
        y_pred_mean = np.mean( y_preds, axis=0 )
        print('Req',index,row['Label Encoded'],"predicted:",np.argmax(y_pred_mean))       
        y_req_cl.append(row['Label Encoded'])
        y_pred_cl_mean.append(np.argmax(y_pred_mean))
    return y_pred_cl_mean, y_req_cl
        

for fold in ["1","2","3","4","5"]:
    train_df = pd.read_csv("../data/train_f"+fold+"_MajCla.csv",names=['Sentence','Label','Req Number','Label Encoded'])
    test_df = pd.read_csv("../data/test_f"+fold+"_MajCla.csv",names=['Sentence','Label','Req Number','Label Encoded'])
    
    ###Training using Glove
    #defining glove model
    GLOVE_WORD_EMBEDDING_SIZE = 300
    inputshape = (GLOVE_WORD_EMBEDDING_SIZE,)
    y_pred, y_true = train_test_model(train_df,test_df,glove,"glove",GLOVE_WORD_EMBEDDING_SIZE)
    if fold == "1":
        y_glove_all_pred = y_pred
        y_glove_all_true = y_true
    else:
        y_glove_all_pred.extend(y_pred)
        y_glove_all_true.extend(y_true)
    
    #Training using BERT word embeddings
    BERT_WORD_EMBEDDING_SIZE = 768
    inputshape = (BERT_WORD_EMBEDDING_SIZE,)
    y_pred, y_true = train_test_model(train_df,test_df,bert,"bert",BERT_WORD_EMBEDDING_SIZE)
    if fold == "1":
        y_bert_all_pred = y_pred
        y_bert_all_true = y_true

    else:
        y_bert_all_pred.extend(y_pred)
        y_bert_all_true.extend(y_true)

    #Training using DistilBERTBERT 
    DISTILBERT_WORD_EMBEDDING_SIZE = 768
    inputshape = (DISTILBERT_WORD_EMBEDDING_SIZE,)
    y_pred, y_true = train_test_model(train_df,test_df,distilbert,"distilbert",DISTILBERT_WORD_EMBEDDING_SIZE)
    if fold == "1":
        y_distilbert_all_pred = y_pred
        y_distilbert_all_true = y_true
    else:
        y_distilbert_all_pred.extend(y_pred)
        y_distilbert_all_true.extend(y_true)
    
    #Training using SBERT DistilBERT
    SENTENCE_DISTILBERT_WORD_EMBEDDING_SIZE = 384
    inputshape = (SENTENCE_DISTILBERT_WORD_EMBEDDING_SIZE,)
    y_pred, y_true = train_test_model(train_df,test_df,sbert,"sbert",SENTENCE_DISTILBERT_WORD_EMBEDDING_SIZE)
    if fold == "1":
        y_sbert_all_pred = y_pred
        y_sbert_all_true = y_true
    else:
        y_sbert_all_pred.extend(y_pred)
        y_sbert_all_true.extend(y_true)

    #Training using USE
    USE_WORD_EMBEDDING_SIZE = 512
    inputshape = (USE_WORD_EMBEDDING_SIZE,)
    y_pred, y_true = train_test_model(train_df,test_df,use,"use",USE_WORD_EMBEDDING_SIZE)
    if fold == "1":
        y_use_all_pred = y_pred
        y_use_all_true = y_true
    else:
        y_use_all_pred.extend(y_pred)
        y_use_all_true.extend(y_true)

names = [ 'F','LF','O','PE','SE','US']
#glove results
utils.printClassificationReport(y_glove_all_pred, y_glove_all_true, classes=names,filename="MF_CR_GLOVE_BASE_knn.txt")
utils.plotConfusionMatrix(y_glove_all_pred,y_glove_all_true,classes=names,saveFig=True, showGraph=False,filename="MF_CM_GLOVE_BASE_knn.png")
#bert results
utils.printClassificationReport(y_bert_all_pred, y_bert_all_true, classes=names,filename="MF_CR_BERT_U_BASE_knn.txt")
utils.plotConfusionMatrix(y_bert_all_pred,y_bert_all_true,classes=names,saveFig=True, showGraph=False,filename="MF_CM_BERT_U_BASE_knn.png")
#distilbert results
utils.printClassificationReport(y_distilbert_all_pred, y_distilbert_all_true, classes=names,filename="MF_CR_DISTILBERT_BASE_knn.txt")
utils.plotConfusionMatrix(y_distilbert_all_pred,y_distilbert_all_true,classes=names,saveFig=True, showGraph=False,filename="MF_CM_DISTILBERT_BASE_knn.png")
#sbert results
utils.printClassificationReport(y_sbert_all_pred, y_sbert_all_true, classes=names,filename="MF_CR_SBERT_BASE_knn.txt")
utils.plotConfusionMatrix(y_sbert_all_pred,y_sbert_all_true,classes=names,saveFig=True, showGraph=False,filename="MF_CM_SBERT_BASE_knn.png")
#glove results
utils.printClassificationReport(y_use_all_pred, y_use_all_true, classes=names,filename="MF_CR_USE_BASE_knn.txt")
utils.plotConfusionMatrix(y_use_all_pred,y_use_all_true,classes=names,saveFig=True, showGraph=False,filename="MF_CM_USE_BASE_knn.png")