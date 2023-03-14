import numpy as np
import pandas as pd
from encoders import GloveEncoder, BERTEncoder, DistilBERTEncoder, SBERTEncoder, UniversalSentenceEncoder
import mlp_models as mlp
import utils
from utils import get_encodings
from keras.utils import to_categorical
import statistics


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
bert = BERTEncoder(WORD_EMBEDDING_SIZE=768)
distilbert = DistilBERTEncoder()
sbert = SBERTEncoder()
use = UniversalSentenceEncoder()

def test_model_sentence(test_df,encoder,model,WORD_EMBEDDING_SIZE,fold):
	#X_test, y_test_labels, test_req_num = get_glove_encodings(test_df)
	#y_test = to_categorical(y_test_labels)
	names = [ 'NF','F']#,'FT','L','LF','MN','O','PE','SC','SE','US']
	print(np.unique(test_df['Label'].values))
	print(names)
	y_req_cl  = []
	y_pred_cl_mean = []
	for index,row in test_df.iterrows():
		print('Req',index)
		sentence = row['Sentence']

		x_vector = encoder.encode_sentence(sentence)
		x = np.zeros((1,WORD_EMBEDDING_SIZE))
		x[0,:] = np.asarray(x_vector)
		y_pred = model.model.predict(x)
		#print(y_preds)
		y_preds_cl = np.argmax(y_pred[0])
		expected_y = row['Label Encoded']
		if expected_y > 1:
			expected_y = 0
		y_req_cl.append(expected_y)
		#print(y_preds_cl, " true ",y_req_cl) 
		y_pred_cl_mean.append(y_preds_cl)

	utils.printClassificationReport(y_pred_cl_mean, y_req_cl, classes=names,filename="CR_TestF"+fold+"_"+model.name+"_mean.txt")
	utils.plotConfusionMatrix(y_pred_cl_mean,y_req_cl,classes=names,saveFig=True, showGraph=False,filename="CM_TestF"+fold+"_"+model.name+"_mean.png")
	return y_pred_cl_mean,y_req_cl

def test_model(test_df,encoder,model,fold):
	#X_test, y_test_labels, test_req_num = get_glove_encodings(test_df)
	#y_test = to_categorical(y_test_labels)
	names = [ 'N','F']#,'FT','L','LF','MN','O','PE','SC','SE','US']
	print(np.unique(test_df['Label'].values))
	print(names)
	y_req_cl  = []
	y_pred_cl_mean = []
	y_pred_cl_mv = []
	for index,row in test_df.iterrows():
		print('Req',index)
		sentence = row['Sentence']
		x_vectors = encoder.encode_sentence(sentence)
		y_preds = model.model.predict(x_vectors)
		y_pred_mean = np.mean( y_preds, axis=0 )
		#print(y_preds)
		y_preds_cl = [np.argmax(y) for y in y_preds]
		mode = statistics.mode(y_preds_cl)
		print(y_preds_cl,row['Label Encoded'])
		print(np.argmax(y_pred_mean),mode)
		expected_y = row['Label Encoded']
		if expected_y > 1:
			expected_y = 0
		y_req_cl.append(expected_y)
		y_pred_cl_mean.append(np.argmax(y_pred_mean))
		y_pred_cl_mv.append(mode)

	utils.printClassificationReport(y_pred_cl_mean, y_req_cl, classes=names,filename="CR_TestF"+fold+"_"+model.name+"_mean.txt")
	utils.plotConfusionMatrix(y_pred_cl_mean,y_req_cl,classes=names,saveFig=True, showGraph=False,filename="CM_TestF"+fold+"_"+model.name+"_mean.png")

	utils.printClassificationReport(y_pred_cl_mv, y_req_cl, classes=names,filename="CR_TestF"+fold+"_"+model.name+"_majvoting.txt")
	utils.plotConfusionMatrix(y_pred_cl_mv,y_req_cl,classes=names,saveFig=True, showGraph=False,filename="CM_TestF"+fold+"_"+model.name+"_majvoting.png")
	return y_pred_cl_mean,y_req_cl

for fold in ["1","2","3","4","5"]:
	test_df = pd.read_csv("../data/test_f"+fold+"_NOPO.csv",names=['Sentence','Label','Req Number','Label Encoded'])
	
	###Training using Glove
	#defining glove model
	GLOVE_WORD_EMBEDDING_SIZE = 300
	inputshape = (GLOVE_WORD_EMBEDDING_SIZE,)
	glove_mlp = mlp.MLP_LARGE(name="BIN_GloVe_f"+fold,inputshape=inputshape,n_outputs=2,lr=0.1,patience=250)
	glove_mlp.loadBestWeights()
	y_pred, y_true = test_model(test_df,glove,glove_mlp,fold)
	if fold == "1":
		y_glove_all_pred = y_pred
		y_glove_all_true = y_true
	else:
		y_glove_all_pred.extend(y_pred)
		y_glove_all_true.extend(y_true)

	#Training using BERT word embeddings
	BERT_WORD_EMBEDDING_SIZE = 768
	inputshape = (BERT_WORD_EMBEDDING_SIZE,)
	bert_mlp = mlp.MLP_LARGE(name="BIN_BERT_f"+fold,inputshape=inputshape,n_outputs=2,lr=0.1,patience=150)
	bert_mlp.loadBestWeights()
	y_pred, y_true = test_model(test_df,bert,bert_mlp,fold)
	if fold == "1":
		y_bert_all_pred = y_pred
		y_bert_all_true = y_true
	else:
		y_bert_all_pred.extend(y_pred)
		y_bert_all_true.extend(y_true)
	
	#Training using DistilBERTBERT 
	DISTILBERT_WORD_EMBEDDING_SIZE = 768
	inputshape = (DISTILBERT_WORD_EMBEDDING_SIZE,)
	distilbert_mlp = mlp.MLP_LARGE(name="BIN_DistilBERT_f"+fold,inputshape=inputshape,n_outputs=2,lr=0.1,patience=150)
	distilbert_mlp.loadBestWeights()
	y_pred, y_true = test_model(test_df,distilbert,distilbert_mlp,fold)
	if fold == "1":
		y_distilbert_all_pred = y_pred
		y_distilbert_all_true = y_true
	else:
		y_distilbert_all_pred.extend(y_pred)
		y_distilbert_all_true.extend(y_true)
	
	#Training using SBERT DistilBERT
	SENTENCE_DISTILBERT_WORD_EMBEDDING_SIZE = 384
	inputshape = (SENTENCE_DISTILBERT_WORD_EMBEDDING_SIZE,)
	sbert_mlp = mlp.MLP_LARGE(name="BIN_SBERT_f"+fold,inputshape=inputshape,n_outputs=2,lr=0.1,patience=150)
	sbert_mlp.loadBestWeights()
	y_pred, y_true = test_model_sentence(test_df,sbert,sbert_mlp,SENTENCE_DISTILBERT_WORD_EMBEDDING_SIZE,fold)
	if fold == "1":
		y_sbert_all_pred = y_pred
		y_sbert_all_true = y_true
	else:
		y_sbert_all_pred.extend(y_pred)
		y_sbert_all_true.extend(y_true)
	
	#Training using USE
	USE_WORD_EMBEDDING_SIZE = 512
	inputshape = (USE_WORD_EMBEDDING_SIZE,)
	use_mlp = mlp.MLP_LARGE(name="BIN_USE_f"+fold,inputshape=inputshape,n_outputs=2,lr=0.1,patience=150)
	use_mlp.loadBestWeights()
	y_pred, y_true = test_model_sentence(test_df,use,use_mlp,USE_WORD_EMBEDDING_SIZE,fold)
	if fold == "1":
		y_use_all_pred = y_pred
		y_use_all_true = y_true
	else:
		y_use_all_pred.extend(y_pred)
		y_use_all_true.extend(y_true)
	
names = ["NF","F"]
#glove results
utils.printClassificationReport(y_glove_all_pred, y_glove_all_true, classes=names,filename="ALL_BIN_CR_GLOVE_LARGE.txt")
utils.plotConfusionMatrix(y_glove_all_pred,y_glove_all_true,classes=names,saveFig=True, showGraph=False,filename="ALL_CM_GLOVE_LARGE.png")
#bert results
utils.printClassificationReport(y_bert_all_pred, y_bert_all_true, classes=names,filename="ALL_BIN_CR_BERT_LARGE.txt")
utils.plotConfusionMatrix(y_bert_all_pred,y_bert_all_true,classes=names,saveFig=True, showGraph=False,filename="ALL_CM_BERT_LARGE.png")
#distilbert results
utils.printClassificationReport(y_distilbert_all_pred, y_distilbert_all_true, classes=names,filename="ALL_BIN_CR_DISTILBERT_LARGE.txt")
utils.plotConfusionMatrix(y_distilbert_all_pred,y_distilbert_all_true,classes=names,saveFig=True, showGraph=False,filename="ALL_BIN_CM_DISTILBERT_LARGE.png")
#sbert results
utils.printClassificationReport(y_sbert_all_pred, y_sbert_all_true, classes=names,filename="ALL_BIN_CR_SBERT_LARGE.txt")
utils.plotConfusionMatrix(y_sbert_all_pred,y_sbert_all_true,classes=names,saveFig=True, showGraph=False,filename="ALL_BIN_CM_SBERT_LARGE.png")
#glove results
utils.printClassificationReport(y_use_all_pred, y_use_all_true, classes=names,filename="ALL_BIN_CR_USE_LARGE.txt")
utils.plotConfusionMatrix(y_use_all_pred,y_use_all_true,classes=names,saveFig=True, showGraph=False,filename="ALL_BIN_CM_USE_LARGE.png")