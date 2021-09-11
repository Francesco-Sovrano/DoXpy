import os
import random
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import multiprocessing
import types
import spacy # for natural language processing
# import neuralcoref # for Coreference Resolution
# python3 -m spacy download en_core_web_md
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() # use 1.X API
tf.get_logger().setLevel('ERROR') # Reduce logging output.
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for dev in gpu_devices:
	tf.config.experimental.set_memory_growth(dev, True)
import tensorflow_hub as hub
import tensorflow_text
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, pipeline
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from knowpy.misc.cache_lib import load_or_create_cache, create_cache, load_cache

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("spacy").setLevel(logging.ERROR)

is_listable = lambda x: type(x) in (list,tuple)

class ModelManager():
	# static members
	__nlp_models = {}
	__tf_embedders = {}
	__hf_embedders = {}
	__sbert_embedders = {}

	def __init__(self, model_options=None):
		if not model_options:
			model_options = {}
		self.model_options = model_options
		self.disable_spacy_component = []
		self.__batch_size = model_options.get('batch_size', 100)
		self.__with_cache = model_options.get('with_cache', True)

		self.__spacy_cache = {}
		self.__tf_cache = {}
		self.__hf_cache = {}
		self.__sbert_cache = {}

		self.__spacy_model = model_options.get('spacy_model', 'en_core_web_md')
		self.__n_threads = model_options.get('n_threads', -1)
		if self.__n_threads < 0:
			self.__n_threads = multiprocessing.cpu_count()
			
		self.__use_cuda = self.model_options.get('use_cuda', False)
		self.__tf_model = model_options.get('tf_model', {})
		self.__hf_model = model_options.get('hf_model', {})
		self.__sbert_model = model_options.get('sbert_model', {})

	def store_cache(self, cache_name):
		cache_dict = {
			'tf_cache': self.__tf_cache,
			'spacy_cache': self.__spacy_cache,
			'hf_cache': self.__hf_cache,
			'sbert_model': self.__sbert_cache,
		}
		create_cache(cache_name, lambda: cache_dict)

	def load_cache(self, cache_name):
		loaded_cache = load_cache(cache_name)
		if loaded_cache:

			tf_cache = loaded_cache.get('tf_cache',None)
			if tf_cache:
				self.__tf_cache = tf_cache

			hf_cache = loaded_cache.get('hf_cache',None)
			if hf_cache:
				self.__hf_cache = hf_cache

			sbert_cache = loaded_cache.get('sbert_cache',None)
			if sbert_cache:
				self.__sbert_cache = sbert_cache

			spacy_cache = loaded_cache.get('spacy_cache',None)
			if spacy_cache:
				self.__spacy_cache = spacy_cache

	@staticmethod
	def get_cached_values(value_list, cache, fetch_fn, key_fn=lambda x:x):
		missing_values = [q for q in value_list if key_fn(q) not in cache]
		if len(missing_values) > 0:
			new_values = fetch_fn(missing_values)
			cache.update({key_fn(q):v for q,v in zip(missing_values, new_values)})
		return [cache[key_fn(q)] for q in value_list]

	@staticmethod
	def load_nlp_model(spacy_model, use_cuda):
		print('## Loading Spacy model <{}>...'.format(spacy_model))
		# go here <https://spacy.io/usage/processing-pipelines> for more information about Language Processing Pipeline (tokenizer, tagger, parser, etc..)
		try:
			if use_cuda:
				activated = spacy.prefer_gpu()
				if activated:
					print('Running spacy on GPU')
			else:
				spacy.require_cpu()
				print('Running spacy on CPU')
			nlp = spacy.load(spacy_model)
		except OSError:
			print('Downloading language model for the spaCy POS tagger\n'
				"(don't worry, this will only happen once)")
			spacy.cli.download(spacy_model)
			if use_cuda:
				activated = spacy.prefer_gpu()
				if activated:
					print('Running spacy on GPU')
			else:
				spacy.require_cpu()
				print('Running spacy on CPU')
			nlp = spacy.load(spacy_model)
		# nlp.add_pipe(nlp.create_pipe("merge_noun_chunks"))
		# nlp.add_pipe(nlp.create_pipe("merge_entities"))
		# nlp.add_pipe(nlp.create_pipe("merge_subtokens"))
		#################################
		# nlp.add_pipe(neuralcoref.NeuralCoref(nlp.vocab), name='neuralcoref', last=True) # load NeuralCoref and add it to the pipe of SpaCy's model
		# def remove_unserializable_results(doc): # Workaround for serialising NeuralCoref's clusters
		# 	def cluster_as_doc(c):
		# 		c.main = c.main.as_doc()
		# 		c.mentions = [
		# 			m.as_doc()
		# 			for m in c.mentions
		# 		]
		# 	# doc.user_data = {}
		# 	if not getattr(doc,'_',None):
		# 		return doc
		# 	if not getattr(doc._,'coref_clusters',None):
		# 		return doc
		# 	for cluster in doc._.coref_clusters:
		# 		cluster_as_doc(cluster)
		# 	for token in doc:
		# 		for cluster in token._.coref_clusters:
		# 			cluster_as_doc(cluster)
		# 	return doc
		# nlp.add_pipe(remove_unserializable_results, last=True)
		print('## Spacy model loaded')
		return nlp
	
	@staticmethod
	def load_tf_model(tf_model):
		cache_dir = tf_model.get('cache_dir',None)
		if cache_dir:
			try:
				Path(cache_dir).mkdir(parents=True, exist_ok=True)
				os.environ["TFHUB_CACHE_DIR"] = cache_dir
			except:
				print(f"Couldn't create cache_dir {cache_dir}")

		use_cpu = not tf_model.get('use_cuda',None)

		model_url = tf_model['url']
		is_qa_model = ModelManager.tf_is_qa_model(tf_model)
		if is_qa_model:
			print(f'## Loading TF model <{model_url}> for QA, on {"CPU" if use_cpu else "GPU"}...')
		else:
			print(f'## Loading TF model <{model_url}>, on {"CPU" if use_cpu else "GPU"}...')
		if use_cpu:
			# with random.choice(tf.config.experimental.list_physical_devices('CPU')):
			with tf.device('/cpu:0'):
				module = hub.load(model_url)
		else:
			module = hub.load(model_url)
		get_input = lambda y: tf.constant(tuple(map(lambda x: x[0] if is_listable(x) else x, y)))
		if is_qa_model:
			get_context = lambda y: tf.constant(tuple(map(lambda x: x[1] if is_listable(x) else '', y)))
			q_label = "query_encoder" if 'query_encoder' in module.signatures else 'question_encoder'
			q_module = lambda doc: module.signatures[q_label](input=get_input(doc))['outputs'].numpy() # The default signature is identical with the question_encoder signature.
			a_module = lambda doc: module.signatures['response_encoder'](input=get_input(doc), context=get_context(doc))['outputs'].numpy()
		else:
			q_module = a_module = lambda doc: module(get_input(doc)).numpy()
		print('## TF model loaded')
		return {
			'question': q_module,
			'answer': a_module
		}

	@staticmethod
	def load_hf_model(hf_model):
		model_name = hf_model['url']
		model_type = hf_model['type']
		model_framework = hf_model.get('framework', 'pt')
		cache_dir = hf_model.get('cache_dir',None)
		if cache_dir:
			model_path = os.path.join(cache_dir, model_name.replace('/','-'))
			if not os.path.isdir(model_path):
				os.mkdir(model_path)
		else:
			model_path = None
		use_cpu = (not hf_model.get('use_cuda',None)) or torch.cuda.device_count()==0
		print(f'###### Loading {model_type} model <{model_name}> for {model_framework}, on {"CPU" if use_cpu else "GPU"} ######')
		config = AutoConfig.from_pretrained(model_name, cache_dir=model_path) # Download configuration from S3 and cache.
		tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
		print(f'###### <{model_name}> loaded ######')
		return {
			'pipeline': pipeline(model_type, 
				model=model_name, 
				tokenizer=model_name, 
				framework=model_framework, 
				device=-1 if use_cpu else 0
			),
            'tokenizer': tokenizer,
            'config': config,
		}

	@staticmethod
	def load_sbert_model(sbert_model):
		model_url = sbert_model['url']
		use_cuda = sbert_model['use_cuda']
		cache_dir = sbert_model.get('cache_dir',None)
		is_qa_model = ModelManager.sbert_is_qa_model(sbert_model)
		if is_qa_model:
			print(f"## Loading sentence_transformers model <{model_url}> for QA, cuda support {use_cuda}...")
		else:
			print(f"## Loading sentence_transformers model <{model_url}>, cuda support {use_cuda}...")
		sbert_model = SentenceTransformer(model_url, device='cpu' if not use_cuda or not gpu_devices else 'cuda', cache_folder=cache_dir)

		get_input = lambda y: tuple(map(lambda x: x[0] if is_listable(x) else x, y))
		if is_qa_model:
			if model_url == 'nq-distilbert-base-v1':
				a_template = lambda x: (x[0],x[1])
				ctx_sbert_model = sbert_model
			else:
				a_template = lambda x: f"{x[0]} [SEP] {x[1]}"
				ctx_sbert_model = SentenceTransformer(model_url.replace('question_encoder','ctx_encoder'), device='cpu' if not use_cuda or not gpu_devices else 'cuda', cache_folder=cache_dir)
			get_context = lambda y: tuple(map(lambda x: x[1] if is_listable(x) else '', y))
			q_module = lambda doc,**args: sbert_model.encode(get_input(doc),**args)
			a_module = lambda doc,**args: ctx_sbert_model.encode(list(map(a_template, zip(get_input(doc),get_context(doc)))),**args)
		else:
			q_module = a_module = lambda doc: sbert_model.encode(get_input(doc),**args)
		print('## SBERT model loaded')
		return {
			'question': q_module,
			'answer': a_module
		}
	
	def get_nlp_model(self):
		if ModelManager.__nlp_models.get(self.__spacy_model, None) is None:
			ModelManager.__nlp_models[self.__spacy_model] = ModelManager.load_nlp_model(self.__spacy_model, self.__use_cuda)
		return ModelManager.__nlp_models[self.__spacy_model]

	def get_tf_model(self):
		model_key = self.__tf_model['url']
		if ModelManager.__tf_embedders.get(model_key, None) is None:
			ModelManager.__tf_embedders[model_key] = ModelManager.load_tf_model(self.__tf_model)
		return ModelManager.__tf_embedders[model_key]

	@staticmethod
	def tf_is_qa_model(model_dict):
		return 'qa' in model_dict['url'].lower()

	@staticmethod
	def sbert_is_qa_model(model_dict):
		return 'question_encoder' in model_dict['url'].lower()

	def get_hf_model(self):
		model_key = (self.__hf_model['url'],self.__hf_model['type'])
		if ModelManager.__hf_embedders.get(model_key, None) is None:
			ModelManager.__hf_embedders[model_key] = ModelManager.load_hf_model(self.__hf_model)
		return ModelManager.__hf_embedders[model_key]

	def get_sbert_model(self):
		model_key = self.__sbert_model['url']
		if ModelManager.__sbert_embedders.get(model_key, None) is None:
			ModelManager.__sbert_embedders[model_key] = ModelManager.load_sbert_model(self.__sbert_model)
		return ModelManager.__sbert_embedders[model_key]

	def nlp(self, text_list, disable=None, n_threads=None, batch_size=None):
		if not disable:
			disable = self.disable_spacy_component
		if not n_threads: # real multi-processing: https://git.readerbench.com/eit/prepdoc/blob/f8e93b6d0a346e9a53dac2e70e5f1712d40d6e1e/examples/parallel_parse.py
			n_threads = self.__n_threads
		if not batch_size:
			batch_size = self.__batch_size
		def fetch_fn(missing_text):
			return self.get_nlp_model().pipe(
				missing_text, 
				disable=disable, 
				batch_size=min(batch_size, int(np.ceil(len(missing_text)/n_threads))),
				n_process=min(n_threads, len(missing_text)), # The keyword argument n_threads on the .pipe methods is now deprecated, as the v2.x models cannot release the global interpreter lock. (Future versions may introduce a n_process argument for parallel inference via multiprocessing.) - https://spacy.io/usage/v2-1#incompat
			)
		if not self.__with_cache:
			return list(fetch_fn(text_list))
		return self.get_cached_values(text_list, self.__spacy_cache, fetch_fn)

	def run_tf_embedding(self, inputs, norm=None, without_context=False):
		def fetch_fn(missing_queries):
			# print(missing_queries)
			tf_model = self.get_tf_model()
			# Feed missing_queries into current tf graph
			batch_list = [
				missing_queries[i*self.__batch_size:(i+1)*self.__batch_size] 
				for i in range(np.int(np.ceil(len(missing_queries)/self.__batch_size)))
			]# if len(missing_queries) > self.__batch_size else [missing_queries]
			encoder = tf_model['question' if without_context else 'answer']
			if len(batch_list) > 1:
				print(f'TF: Modelling {len(missing_queries)} sentences in {len(batch_list)} batches..')
				batched_embeddings = tuple(map(encoder, tqdm(batch_list)))
				# batched_embeddings = tuple(map(encoder, batch_list))
				embeddings = np.concatenate(batched_embeddings, 0)
			else:
				embeddings = encoder(batch_list[0])
			# Normalize the embeddings, if required
			if norm is not None:
				embeddings = normalize(embeddings, norm=norm)
			return embeddings
		if not self.__with_cache:
			return list(fetch_fn(inputs))
		return self.get_cached_values(inputs, self.__tf_cache, fetch_fn, key_fn=lambda x:(x,without_context))

	def run_sbert_embedding(self, inputs, convert_to_tensor=False, without_context=False):
		def fetch_fn(missing_inputs):
			sbert_model = self.get_sbert_model()
			batch_list = [
				missing_inputs[i*self.__batch_size:(i+1)*self.__batch_size] 
				for i in range(np.int(np.ceil(len(missing_inputs)/self.__batch_size)))
			]# if len(missing_inputs) > self.__batch_size else [missing_inputs]
			encoder = sbert_model['question' if without_context else 'answer']
			if len(batch_list) > 1:
				print(f'SBert: Modelling {len(missing_inputs)} sentences in {len(batch_list)} batches..')
				batched_embeddings = [
					encoder(batch, convert_to_tensor=convert_to_tensor)
					for batch in tqdm(batch_list)
					# for batch in batch_list
				]
				embeddings = np.concatenate(batched_embeddings, 0)
			else:
				embeddings = encoder(batch_list[0], convert_to_tensor=convert_to_tensor)
			return embeddings
		if not self.__with_cache:
			return list(fetch_fn(inputs))
		return self.get_cached_values(inputs, self.__sbert_cache, fetch_fn)

	def run_hf_task(self, inputs, **kwargs):
		def fetch_fn(missing_inputs):
			hf_model = self.get_hf_model()
			# print(f'HF: Embedding {len(missing_inputs)} documents.')
			# return [hf_model['pipeline'](i, **kwargs) for i in tqdm(missing_inputs)]
			return [hf_model['pipeline'](i, **kwargs) for i in missing_inputs]
		cache_key = '.'.join(map(lambda x: '='.join(map(str,x)), sorted(kwargs.items(), key=lambda x:x[0])))
		if not self.__with_cache:
			return fetch_fn(inputs)
		return self.get_cached_values(inputs, self.__hf_cache, fetch_fn, key_fn=lambda x: '.'.join((cache_key,x)))

	def get_default_embedder(self):
		if self.__tf_model:
			return self.run_tf_embedding
		elif self.__sbert_model:
			return self.run_sbert_embedding
		return None

	def get_default_similarity_fn(self):
		if self.__tf_model:
			# np.inner == lambda x,y: np.matmul(x,np.transpose(y))
			return np.inner
		elif self.__sbert_model:
			return cosine_similarity
		return None

	def get_default_embedding(self, text_list, without_context=True):
		embedding_fn = self.get_default_embedder()
		assert embedding_fn is not None, 'cannot find a proper embedding_fn'
		return embedding_fn(text_list, without_context=without_context)

	def get_default_similarity(self, a, b):
		similarity_fn = self.get_default_similarity_fn()
		assert similarity_fn is not None, 'cannot find a proper similarity_fn'
		return similarity_fn(a,b)

	def get_similarity_ranking(self, source_text_list, target_text_list, without_context=False, get_embedding_fn=None, get_similarity_fn=None):
		if get_embedding_fn is None:
			get_embedding_fn = self.get_default_embedding
		if get_similarity_fn is None:
			get_similarity_fn = self.get_default_similarity
		source_embeddings = get_embedding_fn(source_text_list, without_context=without_context)
		target_embeddings = get_embedding_fn(target_text_list, without_context=False)
		similarity_vec = get_similarity_fn(source_embeddings,target_embeddings)
		return np.argsort(similarity_vec, kind='stable', axis=-1), similarity_vec

	def get_most_similar_idx_n_label(self, source_text_list, target_text_list, without_context=False, get_embedding_fn=None, get_similarity_fn=None):
		if get_embedding_fn is None:
			get_embedding_fn = self.get_default_embedding
		if get_similarity_fn is None:
			get_similarity_fn = self.get_default_similarity
		similarity_vec = get_similarity_fn(
			get_embedding_fn(source_text_list, without_context=without_context),
			get_embedding_fn(target_text_list, without_context=False)
		)
		argmax_list = np.argmax(similarity_vec, axis=-1)
		return list(zip(argmax_list, np.take(similarity_vec,argmax_list)))

	def remove_similar_labels(self, tuple_list, threshold=0.97, key=None, without_context=True, get_embedding_fn=None, get_similarity_fn=None, sort_by_uniqueness=False):
		if key is None:
			key = lambda x: x[0] if isinstance(x, (list,tuple)) else x
		if get_embedding_fn is None:
			get_embedding_fn = self.get_default_embedding
		if get_similarity_fn is None:
			get_similarity_fn = self.get_default_similarity
		value_list = list(map(key,tuple_list))
		similarity_vec = get_similarity_fn(
			get_embedding_fn(value_list, without_context=without_context),
			get_embedding_fn(value_list, without_context=without_context)
		)
		if sort_by_uniqueness:
			sorted_idx_vec = np.argsort(np.sum(similarity_vec, axis=-1), kind='stable', axis=-1)
			sorted_similarity_vec = np.take(similarity_vec, sorted_idx_vec, axis=0)
			return [
				tuple_list[j]
				for i,j in enumerate(sorted_idx_vec.tolist())
				if not np.any(sorted_similarity_vec[i][:i] >= threshold)
			]
		# for i,_ in enumerate(value_list):
		# 	for j,v in enumerate(similarity_vec[i][:i]):
		# 		if v >= threshold:
		# 			print(0, v, key(tuple_list[i]))
		# 			print(1, v, key(tuple_list[j]))
		# 			break
		return [
			v
			for i,v in enumerate(tuple_list)
			if not np.any(similarity_vec[i][:i] >= threshold)
		]

	def filter_by_similarity_to_target(self, source_tuple_list, target_tuple_list, threshold=0.97, source_key=None, target_key=None, source_without_context=True, target_without_context=True, get_embedding_fn=None, get_similarity_fn=None):
		if source_key is None:
			source_key = lambda x: x[0] if isinstance(x, (list,tuple)) else x
		if target_key is None:
			target_key = lambda x: x[0] if isinstance(x, (list,tuple)) else x
		if get_embedding_fn is None:
			get_embedding_fn = self.get_default_embedding
		if get_similarity_fn is None:
			get_similarity_fn = self.get_default_similarity
		similarity_vec = get_similarity_fn(
			get_embedding_fn(list(map(source_key,source_tuple_list)), without_context=source_without_context),
			get_embedding_fn(list(map(target_key,target_tuple_list)), without_context=target_without_context)
		)
		# for i,_ in enumerate(source_tuple_list):
		# 	for j,v in enumerate(similarity_vec[i]):
		# 		if v >= threshold:
		# 			print(0, v, source_key(source_tuple_list[i]))
		# 			print(1, v, target_key(target_tuple_list[j]))
		# 			break
		return [
			v
			for i,v in enumerate(source_tuple_list)
			if not np.any(similarity_vec[i] >= threshold)
		]
