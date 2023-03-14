import numpy as np
import pandas as pd
import utils
from gensim.parsing.preprocessing import STOPWORDS



##### GloVE encoder ##################
class GloveEncoder():
	def __init__(self,WORD_EMBEDDING_SIZE=50,MAX_SENTENCE_LENGTH=10) -> None:
		self.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
		self.WORD_EMBEDDING_SIZE = WORD_EMBEDDING_SIZE
		print("loading glove")
		df = pd.read_csv('../pretrained/glove.6B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
		print("building dictionary")
		self.glove = {key: val.values for key, val in df.T.items()}
		print('done')

	def encode_word(self,word):
		try:
			vec = self.glove[word.lower()]
			return vec, True
		except:
			return np.zeros((self.WORD_EMBEDDING_SIZE)), False


	def encode_sentence(self,sentence):
		clean_sentence = utils.remove_punctuation(sentence)
		words = clean_sentence.split(" ")
		vectors = []
		for word in words:
			if word in STOPWORDS:
				continue
			try:
				vec = self.glove[word.lower()]
				x_vec = np.zeros((1,self.WORD_EMBEDDING_SIZE))
				x_vec[0,:] = vec
				vectors.append(x_vec)
			except:
					pass # OOD word - skipping
		word_vectors = np.zeros((len(vectors),self.WORD_EMBEDDING_SIZE))
		for i,vector in enumerate(vectors):
			word_vectors[i,:] = vector
		return word_vectors

	def encode_sentence_as_tensor(self,sentence,MAX_SENTENCE_LENGTH):
		clean_sentence = utils.remove_punctuation(sentence)
		words = clean_sentence.split(" ")
		count = 0
		vectors = []
		for word in words:
			if word in STOPWORDS:
				if word != "system":
					#print("removing stopword: ",word)
					continue
				if word == "":	
					continue
					#count += 1
			try:
				vec = self.glove[word.lower()]
				x_vec = np.zeros((1,self.WORD_EMBEDDING_SIZE))
				x_vec[0,:] = vec
				vectors.append(x_vec)
				count += 1
				
			except:
					#list_of_words = []
					#list_of_words.append(word)
					pass
		X_es = []
		while len(vectors) < MAX_SENTENCE_LENGTH:
			vectors.append(np.zeros((1,self.WORD_EMBEDDING_SIZE)))
			print("padding short sentence")
		#y_es_cl = []

		for i in range(max(1,len(vectors)-MAX_SENTENCE_LENGTH)):
			x = np.zeros((MAX_SENTENCE_LENGTH,self.WORD_EMBEDDING_SIZE))
			for s in range(MAX_SENTENCE_LENGTH):
				try:
					x[s] = vectors[i+s]
				except:
						pass
				#print(x)
				X_es.append(x)
				#y_es_cl.append(g['Label'].values[0])
			#predictions = g['Prediction'].values
			#real_value = g['Label'].values[0]

		#print(len(X_es))
		X_step = np.zeros((len(X_es),MAX_SENTENCE_LENGTH,self.WORD_EMBEDDING_SIZE))
		for i,x in enumerate(X_es):
			X_step[i] = x 
		#print(X_step.shape)
		
		#print(y_es.shape)
		return X_step


##### BERT encoder ##################
import torch
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel

class BERTEncoder():
	def __init__(self,uncased=False,WORD_EMBEDDING_SIZE=768,MAX_SENTENCE_LENGTH=10) -> None:
		# Load pre-trained model tokenizer (vocabulary)
		modelname = 'bert-base-cased'
		if uncased:
			modelname = 'bert-base-uncased'
		self.uncased = uncased
		self.tokenizer = BertTokenizer.from_pretrained(modelname)
		# Load pre-trained model (weights)
		self.model = BertModel.from_pretrained(modelname, output_hidden_states = True )
		# Put the model in "evaluation" mode, meaning feed-forward operation.
		self.model.eval()

	def encode_sentence(self, sentence):
		if self.uncased:
			sentence = sentence.lower()
		marked_text = "[CLS] " + sentence + " [SEP]"
		# Tokenize our sentence with the BERT tokenizer.
		tokenized_text = self.tokenizer.tokenize(marked_text)
		# Map the token strings to their vocabulary indeces.
		indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
		# Display the words with their indeces.
		#for tup in zip(tokenized_text, indexed_tokens):
		#	print('{:<12} {:>6,}'.format(tup[0], tup[1]))
		# Mark each of the 22 tokens as belonging to sentence "1".
		segments_ids = [1] * len(tokenized_text)
		# Convert inputs to PyTorch tensors
		tokens_tensor = torch.tensor([indexed_tokens])
		segments_tensors = torch.tensor([segments_ids])
		# Run the text through BERT, and collect all of the hidden states produced
		# from all 12 layers. 
		with torch.no_grad():
			outputs = self.model(tokens_tensor, segments_tensors)
			# Evaluating the model will return a different number of objects based on 
			# how it's  configured in the `from_pretrained` call earlier. In this case, 
			# becase we set `output_hidden_states = True`, the third item will be the 
			# hidden states from all layers. See the documentation for more details:
			# https://huggingface.co/transformers/model_doc/bert.html#bertmodel
			hidden_states = outputs[2]
			#hidden_states shape is [# layers, # batches, # tokens, # features]
			#permuting to [# tokens, # layers, # features]
			# First Concatenate the tensors for all layers. We use `stack` here to
			# create a new dimension in the tensor.
			token_embeddings = torch.stack(hidden_states, dim=0)
			#token_embeddings.size()
			# then Remove dimension 1, the "batches".
			token_embeddings = torch.squeeze(token_embeddings, dim=1)
			# finally Swap dimensions 0 and 1.
			token_embeddings = token_embeddings.permute(1,0,2)
			# Stores the token vectors, with shape [? x 768]
			token_vecs_sum = []
			token_vecs_avg = []
			# `token_embeddings` is a [? x 12 x 768] tensor.
			# For each token in the sentence...
			for token in token_embeddings:
				# `token` is a [12 x 768] tensor
				# Sum the vectors from the last four layers.
				sum_vec = torch.sum(token[-4:], dim=0)
				#avg_vec = torch.mean(token[-4:], dim=0)
				# Use `sum_vec` to represent `token`.
				token_vecs_sum.append(sum_vec)
				#token_vecs_avg.append(avg_vec)
		x_encodings = np.zeros((len(token_vecs_sum),768))
		x_encodings[:] = [x.numpy() for x in token_vecs_sum]
		return x_encodings#, token_vecs_avg


class DistilBERTEncoder():
	def __init__(self,uncased=False,WORD_EMBEDDING_SIZE=384,MAX_SENTENCE_LENGTH=10) -> None:
		modelname = 'distilbert-base-cased'
		if uncased:
			modelname = 'distilbert-base-uncased'
		self.uncased = uncased
		# Load pre-trained model tokenizer (vocabulary)
		self.tokenizer = DistilBertTokenizer.from_pretrained(modelname)
		# Load pre-trained model (weights)
		self.model = DistilBertModel.from_pretrained(modelname, output_hidden_states = True )
		# Put the model in "evaluation" mode, meaning feed-forward operation.
		self.model.eval()

	def encode_sentence(self, sentence):
		if self.uncased:
			sentence = sentence.lower()
		marked_text = "[CLS] " + sentence + " [SEP]"
		# Tokenize our sentence with the BERT tokenizer.
		tokenized_text = self.tokenizer.tokenize(marked_text)
		# Map the token strings to their vocabulary indeces.
		indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
		# Display the words with their indeces.
		#for tup in zip(tokenized_text, indexed_tokens):
		#	print('{:<12} {:>6,}'.format(tup[0], tup[1]))
		# Mark each of the 22 tokens as belonging to sentence "1".
		#segments_ids = [1] * len(tokenized_text)
		# Convert inputs to PyTorch tensors
		tokens_tensor = torch.tensor([indexed_tokens])
		#segments_tensors = torch.tensor([segments_ids])
		# Run the text through BERT, and collect all of the hidden states produced
		# from all 12 layers. 
		with torch.no_grad():
			outputs = self.model(tokens_tensor)#, segments_tensors)
			# Evaluating the model will return a different number of objects based on 
			# how it's  configured in the `from_pretrained` call earlier. In this case, 
			# becase we set `output_hidden_states = True`, the third item will be the 
			# hidden states from all layers. See the documentation for more details:
			# https://huggingface.co/transformers/model_doc/bert.html#bertmodel
			hidden_states = outputs[1]
			#hidden_states shape is [# layers, # batches, # tokens, # features]
			#permuting to [# tokens, # layers, # features]
			# First Concatenate the tensors for all layers. We use `stack` here to
			# create a new dimension in the tensor.
			token_embeddings = torch.stack(hidden_states, dim=0)
			#token_embeddings.size()
			# then Remove dimension 1, the "batches".
			token_embeddings = torch.squeeze(token_embeddings, dim=1)
			# finally Swap dimensions 0 and 1.
			token_embeddings = token_embeddings.permute(1,0,2)
			# Stores the token vectors, with shape [? x 768]
			token_vecs_sum = []
			token_vecs_avg = []
			# `token_embeddings` is a [? x 12 x 768] tensor.
			# For each token in the sentence...
			for token in token_embeddings:
				# `token` is a [12 x 768] tensor
				# Sum the vectors from the last four layers.
				sum_vec = torch.sum(token[-4:], dim=0)
				#avg_vec = torch.mean(token[-4:], dim=0)
				# Use `sum_vec` to represent `token`.
				token_vecs_sum.append(sum_vec)
				#token_vecs_avg.append(avg_vec)
		x_encodings = np.zeros((len(token_vecs_sum),768))
		x_encodings[:] = [x.numpy() for x in token_vecs_sum]
		return x_encodings#, token_vecs_avg

class DistilBERTEncoderOLD():
	def __init__(self,WORD_EMBEDDING_SIZE=384,MAX_SENTENCE_LENGTH=10) -> None:
		# Load pre-trained model tokenizer (vocabulary)
		self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
		# Load pre-trained model (weights)
		self.model = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states = True )
		# Put the model in "evaluation" mode, meaning feed-forward operation.
		self.model.eval()

	def encode_sentence(self, sentence):
		sentence = sentence.lower()
		marked_text = "[CLS] " + sentence + " [SEP]"
		# Tokenize our sentence with the BERT tokenizer.
		tokenized_text = self.tokenizer.tokenize(marked_text)
		# Map the token strings to their vocabulary indeces.
		indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
		# Display the words with their indeces.
		#for tup in zip(tokenized_text, indexed_tokens):
		#	print('{:<12} {:>6,}'.format(tup[0], tup[1]))
		# Mark each of the 22 tokens as belonging to sentence "1".
		#segments_ids = [1] * len(tokenized_text)
		# Convert inputs to PyTorch tensors
		tokens_tensor = torch.tensor([indexed_tokens])
		#segments_tensors = torch.tensor([segments_ids])
		# Run the text through BERT, and collect all of the hidden states produced
		# from all 12 layers. 
		with torch.no_grad():
			outputs = self.model(tokens_tensor)#, segments_tensors)
			# Evaluating the model will return a different number of objects based on 
			# how it's  configured in the `from_pretrained` call earlier. In this case, 
			# becase we set `output_hidden_states = True`, the third item will be the 
			# hidden states from all layers. See the documentation for more details:
			# https://huggingface.co/transformers/model_doc/bert.html#bertmodel
			hidden_states = outputs[1]
			#hidden_states shape is [# layers, # batches, # tokens, # features]
			#permuting to [# tokens, # layers, # features]
			# First Concatenate the tensors for all layers. We use `stack` here to
			# create a new dimension in the tensor.
			token_embeddings = torch.stack(hidden_states, dim=0)
			#token_embeddings.size()
			# then Remove dimension 1, the "batches".
			token_embeddings = torch.squeeze(token_embeddings, dim=1)
			# finally Swap dimensions 0 and 1.
			token_embeddings = token_embeddings.permute(1,0,2)
			# Stores the token vectors, with shape [? x 768]
			token_vecs_sum = []
			token_vecs_avg = []
			# `token_embeddings` is a [? x 12 x 768] tensor.
			# For each token in the sentence...
			for token in token_embeddings:
				# `token` is a [12 x 768] tensor
				# Sum the vectors from the last four layers.
				sum_vec = torch.sum(token[-4:], dim=0)
				#avg_vec = torch.mean(token[-4:], dim=0)
				# Use `sum_vec` to represent `token`.
				token_vecs_sum.append(sum_vec)
				#token_vecs_avg.append(avg_vec)
		x_encodings = np.zeros((len(token_vecs_sum),768))
		x_encodings[:] = [x.numpy() for x in token_vecs_sum]
		return x_encodings#, token_vecs_avg

##### USE encoder ##################
class UniversalSentenceEncoder:
    def __init__(self):
            import tensorflow_hub as hub
            self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def encode_sentence(self,sentence):
            return [self.model([sentence])[0]] #shape (1,512)


##### SBERT Sentence BERT encoder ##################
### model_name can be one from https://www.sbert.net/docs/pretrained_models.html
#### DistiBERT is 'all-MiniLM-L6-v2' 384 dimension -  BERT 'all-mpnet-base-v2' 768 dimension 
class SBERTEncoder:
    def __init__(self,model_name='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        
    def encode_sentence(self,sentence):
        return [self.model.encode(sentence)] #shape (384,)