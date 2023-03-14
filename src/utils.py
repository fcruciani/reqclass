import string
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn

def remove_punctuation(text):
	text_no_punct = ""
	for char in text:
		if char not in string.punctuation:
			text_no_punct += char
		else:
			text_no_punct += " "
	#text_no_punct = "".join([char for char in text if char not in string.punctuation])
	return text_no_punct

def printClassificationReport(pred,true,classes,filename=""):
	cr = classification_report(np.array(true), np.array(pred),target_names=classes,digits=4)
	print(cr)
	if not filename == "":
		with open(filename,"w") as out_file:
			out_file.write(cr)

def printAccuracyScore(pred, true,filename=""):
	acc = accuracy_score(np.array(true), np.array(pred))
	print("Accuracy: ",acc)
	if not filename == "":
		with open(filename,"w") as out_file:
			out_file.write(str(acc))

def printBalancedAccuracyScore( pred, true,filename=""):
	bal_acc = accuracy_score(np.array(true), np.array(pred))
	print("Balanced Accuracy: ",bal_acc)
	if not filename == "":
		with open(filename,"w") as out_file:
			out_file.write(str(bal_acc))

def plotConfusionMatrix(pred,true,classes,saveFig=True,showGraph=False,filename="undefined",normalize=True):
	cm = confusion_matrix(np.array(true), np.array(pred) )
	fig = plt.figure(figsize=(8, 7))
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	df_cm = pd.DataFrame(cm,index=classes,columns=classes)
	if normalize:
		ax = sn.heatmap(df_cm,cmap='Blues',annot=True)
	else:
		ax = sn.heatmap(df_cm,cmap='Blues',annot=True,fmt='d')
	plt.yticks(rotation=0)
	if showGraph:
		plt.show()
	if saveFig:
		fig.tight_layout()
		if filename == "undefined":
			fig.savefig("undefined_CM.png",dpi=300)
		else:
			fig.savefig(filename,dpi=300)

def get_encodings(df,encoder,WORD_EMBEDDING_SIZE):
    labels = []
    req_numbers = []
    encodings = []
    for index,row in df.iterrows():
        #print(index,row['Label'],row['Req Number'])
        x_vectors = encoder.encode_sentence(row['Sentence'])
        y_labels = []
        y_req_numbers = []
        for i in range(len(x_vectors)):
            y_labels.append(row['Label Encoded'])
            y_req_numbers.append( row['Req Number'] )
            encodings.append(x_vectors[i])
        labels.extend(y_labels)
        req_numbers.extend(y_req_numbers)
        #print(len(encodings),len(labels),len(req_numbers))
    X_encodings = np.zeros((len(encodings),WORD_EMBEDDING_SIZE) )
    X_encodings[:] = [x for x in encodings]
    return  X_encodings, labels, req_numbers