import numpy as np
import pandas as pd
from encoders import GloveEncoder, BERTEncoder, DistilBERTEncoder, SBERTEncoder, UniversalSentenceEncoder
import mlp_models as mlp
import utils
from utils import get_encodings
from keras.utils import to_categorical
#WORD_EMBEDDING_SIZE = 300

#
labels = [ 'A','F','FT','L','LF','MN','O','PE','SC','SE','US']
labels_bin = ['NF','F']
#note: Functional (F) class 1

glove = GloveEncoder(WORD_EMBEDDING_SIZE=300)
bert = BERTEncoder(WORD_EMBEDDING_SIZE=768)
distilbert = DistilBERTEncoder()
sbert = SBERTEncoder()
use = UniversalSentenceEncoder()

def train_model(train_df,encoder,model,WORD_EMBEDDING_SIZE,bs=32):
    X_train, y_train_labels, train_req_num = get_encodings(train_df,encoder,WORD_EMBEDDING_SIZE)
    
    #transform to binary
    #all labels > 1 are set to 0 (NF)
    y_train_labels = np.asarray(y_train_labels)
    y_train_labels[ y_train_labels > 1] = 0 
    
    y_train_cl = y_train_labels
    from sklearn.model_selection import train_test_split
    X_train, X_vld,  y_train_cl, y_vld_cl = train_test_split(X_train, y_train_cl, 
                                                        stratify=y_train_cl,
                                                        random_state=42, 
                                                        test_size=0.1)

    print("Train: ",X_train.shape,len(y_train_cl))
    print("Vld: ",X_vld.shape,len(y_vld_cl))

    ##class weight 
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train_cl),y=y_train_cl)
    class_weights = dict(zip(np.unique(y_train_cl), class_weights))
    print(class_weights)
    y_train = to_categorical(y_train_cl)
    y_vld = to_categorical(y_vld_cl)
    #bs = 32 #batch size
    model.fit(X_train,y_train,validation_data=(X_vld,y_vld),class_weights=class_weights, epochs=5000,verbose=1,batch_size=bs)
    model.plotAccuracy(showGraph=False,saveFig=True)

for fold in ["1","2","3","4","5"]:
    train_df = pd.read_csv("../data/train_f"+fold+"_NOPO.csv",names=['Sentence','Label','Req Number','Label Encoded'])

    ###Training using Glove
    #defining glove model
    GLOVE_WORD_EMBEDDING_SIZE = 300
    inputshape = (GLOVE_WORD_EMBEDDING_SIZE,)
    glove_mlp = mlp.MLP_LARGE(name="BIN_GloVe_f"+fold,inputshape=inputshape,n_outputs=2,lr=0.1,patience=75)
    train_model(train_df,glove,glove_mlp,GLOVE_WORD_EMBEDDING_SIZE)
    
    #Training using BERT word embeddings
    BERT_WORD_EMBEDDING_SIZE = 768
    inputshape = (BERT_WORD_EMBEDDING_SIZE,)
    bert_mlp = mlp.MLP_LARGE(name="BIN_BERT_f"+fold,inputshape=inputshape,n_outputs=2,lr=0.1,patience=75)
    train_model(train_df,bert,bert_mlp,BERT_WORD_EMBEDDING_SIZE)
    
    #Training using DistilBERTBERT 
    DISTILBERT_WORD_EMBEDDING_SIZE = 768
    inputshape = (DISTILBERT_WORD_EMBEDDING_SIZE,)
    distilbert_mlp = mlp.MLP_LARGE(name="BIN_DistilBERT_f"+fold,inputshape=inputshape,n_outputs=2,lr=0.1,patience=75)
    train_model(train_df,distilbert,distilbert_mlp,DISTILBERT_WORD_EMBEDDING_SIZE)

    
    #Training using SBERT DistilBERT
    SENTENCE_DISTILBERT_WORD_EMBEDDING_SIZE = 384
    inputshape = (SENTENCE_DISTILBERT_WORD_EMBEDDING_SIZE,)
    sbert_mlp = mlp.MLP_LARGE(name="BIN_SBERT_f"+fold,inputshape=inputshape,n_outputs=2,lr=0.1,patience=75)
    train_model(train_df,sbert,sbert_mlp,SENTENCE_DISTILBERT_WORD_EMBEDDING_SIZE,bs=3)


    #Training using USE
    USE_WORD_EMBEDDING_SIZE = 512
    inputshape = (USE_WORD_EMBEDDING_SIZE,)
    use_mlp = mlp.MLP_LARGE(name="BIN_USE_f"+fold,inputshape=inputshape,n_outputs=2,lr=0.1,patience=75)
    train_model(train_df,use,use_mlp,USE_WORD_EMBEDDING_SIZE,bs=3)
    