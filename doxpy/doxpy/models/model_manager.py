import os
import random
import re
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import multiprocessing
import types
import spacy # for natural language processing
# import neuralcoref # for Coreference Resolution
# python3 -m spacy download en_core_web_md
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() # use 1.X API

import tensorflow_hub as hub
import tensorflow_text
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, pipeline, set_seed
from more_itertools import unique_everseen

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from doxpy.misc.cache_lib import load_or_create_cache, create_cache, load_cache
from doxpy.misc.utils import *

import warnings
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.get_logger().setLevel(logging.ERROR) # Reduce logging output.
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for dev in gpu_devices:
	tf.config.experimental.set_memory_growth(dev, True)

# warnings.filterwarnings('ignore')
logging.getLogger("spacy").setLevel(logging.ERROR)

# set_seed(42)
is_listable = lambda x: type(x) in (list,tuple)

class ModelManager():
	# static members
	__nlp_models = {}
	__tf_embedders = {}
	__hf_embedders = {}
	__sbert_embedders = {}

	logger = logging.getLogger('doxpy')

	def __init__(self, model_options=None):
		if not model_options:
			model_options = {}
		self.model_options = model_options
		self.disable_spacy_component = []
		self.with_tqdm = model_options.get('with_tqdm', False)

		self.__default_cache_dir = model_options.get('default_cache_dir', None)
		self.__default_batch_size = model_options.get('default_batch_size', 1000)
		self.__with_cache = model_options.get('with_cache', True)

		self.reset_cache()

		self.__spacy_model = model_options.get('spacy_model', 'en_core_web_md')
		self.__n_threads = model_options.get('n_threads', -1)
		if self.__n_threads < 0:
			self.__n_threads = multiprocessing.cpu_count()
			
		self.__use_cuda = self.model_options.get('use_cuda', False)
		self.__tf_model_options = model_options.get('tf_model', {})
		self.__hf_model_options = model_options.get('hf_model', {})
		self.__sbert_model_options = model_options.get('sbert_model', {})

	def reset_cache(self):
		self.__spacy_cache = {}
		self.__tf_cache = {}
		self.__hf_cache = {}
		self.__sbert_cache = {}

	@property
	def n_threads(self):
		return self.__n_threads

	@property
	def default_cache_dir(self):
		return self.__default_cache_dir

	@property
	def default_batch_size(self):
		return self.__default_batch_size

	def tqdm(self, it, **args):
		if isinstance(it, (list,tuple)) and len(it) <= 1:
			return it
		if args.get('total', float('inf')) <= 1:
			return it
		if self.with_tqdm:
			return tqdm(it, **args)
		return it

	def store_cache(self, cache_name):
		cache_dict = {
			'tf_cache': self.__tf_cache,
			'spacy_cache': self.__spacy_cache,
			'hf_cache': self.__hf_cache,
			'sbert_cache': self.__sbert_cache,
		}
		create_cache(cache_name, lambda: cache_dict)

	def load_cache(self, cache_name):
		loaded_cache = load_cache(cache_name)
		if not loaded_cache:
			return False

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

		return True

	@staticmethod
	def get_cached_values(value_list, cache, fetch_fn, key_fn=lambda x:x):
		if isinstance(value_list, types.GeneratorType):
			value_list = tuple(value_list)
		missing_values = tuple(
			q 
			for q in unique_everseen(filter(lambda x:x, value_list), key=key_fn) 
			if key_fn(q) not in cache
		)
		if len(missing_values) > 0:
			new_values = fetch_fn(missing_values)
			cache.update({
				key_fn(q):v 
				for q,v in zip(missing_values, new_values)
			})
			# del missing_values
			# del new_values
		return [cache[key_fn(q)] if q else None for q in value_list]

	@staticmethod
	def load_nlp_model(spacy_model, use_cuda):
		ModelManager.logger.info('## Loading Spacy model <{}>...'.format(spacy_model))
		# go here <https://spacy.io/usage/processing-pipelines> for more information about Language Processing Pipeline (tokenizer, tagger, parser, etc..)
		try:
			if use_cuda:
				activated = spacy.prefer_gpu()
				if activated:
					ModelManager.logger.info('Running spacy on GPU')
			else:
				spacy.require_cpu()
				ModelManager.logger.info('Running spacy on CPU')
			nlp = spacy.load(spacy_model)
		except OSError:
			ModelManager.logger.warning('Downloading language model for the spaCy POS tagger\n'
				"(don't worry, this will only happen once)")
			spacy.cli.download(spacy_model)
			if use_cuda:
				activated = spacy.prefer_gpu()
				if activated:
					ModelManager.logger.info('Running spacy on GPU')
			else:
				spacy.require_cpu()
				ModelManager.logger.info('Running spacy on CPU')
			nlp = spacy.load(spacy_model)
		nlp.add_pipe("doc_cleaner")
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
		ModelManager.logger.info('## Spacy model loaded')
		return nlp
	
	@staticmethod
	def load_tf_model(tf_model_options):
		cache_dir = tf_model_options.get('cache_dir',None)
		if cache_dir:
			try:
				Path(cache_dir).mkdir(parents=True, exist_ok=True)
				os.environ["TFHUB_CACHE_DIR"] = cache_dir
			except:
				ModelManager.logger.warning(f"Couldn't create cache_dir {cache_dir}")

		use_cpu = not tf_model_options.get('use_cuda',False) or len(gpu_devices)==0

		model_url = tf_model_options['url']
		is_qa_model = ModelManager.tf_is_qa_model(tf_model_options)
		if is_qa_model:
			ModelManager.logger.info(f'## Loading TF model <{model_url}> for QA, on {"CPU" if use_cpu else "GPU"}...')
		else:
			ModelManager.logger.info(f'## Loading TF model <{model_url}>, on {"CPU" if use_cpu else "GPU"}...')

		def get_models():
			module = hub.load(model_url)
			# if use_cpu: # show temporarily hidden GPUs
			# 	tf.config.set_visible_devices(gpu_devices, 'GPU')
			# os.environ["CUDA_VISIBLE_DEVICES"] = str(len(gpu_devices)) # set the right number of available GPUs for other processes to use

			get_input = lambda y: tf.constant(tuple(map(lambda x: x[0] if is_listable(x) else x, y)))
			if is_qa_model:
				get_context = lambda y: tf.constant(tuple(map(lambda x: x[1] if is_listable(x) else '', y)))
				q_label = "query_encoder" if 'query_encoder' in module.signatures else 'question_encoder'
				q_module = lambda doc: module.signatures[q_label](input=get_input(doc))['outputs'].numpy() # The default signature is identical with the question_encoder signature.
				a_module = lambda doc: module.signatures['response_encoder'](input=get_input(doc), context=get_context(doc))['outputs'].numpy()
			else:
				q_module = a_module = lambda doc: module(get_input(doc)).numpy()
			return q_module, a_module

		if use_cpu:
			with tf.device("/cpu:0"):
				q_module, a_module = get_models()
		else:
			q_module, a_module = get_models()
		ModelManager.logger.info('## TF model loaded')
		return {
			'question': q_module,
			'answer': a_module
		}

	@staticmethod
	def load_hf_model(hf_model_options):
		model_name = hf_model_options['url']
		model_type = hf_model_options['type']
		model_framework = hf_model_options.get('framework', 'pt')
		cache_dir = hf_model_options.get('cache_dir',None)
		if cache_dir:
			model_path = os.path.join(cache_dir, model_name.replace('/','-'))
			if not os.path.isdir(model_path):
				os.mkdir(model_path)
		else:
			model_path = None
		use_cpu = (not hf_model_options.get('use_cuda',False)) or torch.cuda.device_count()==0
		ModelManager.logger.info(f'###### Loading {model_type} model <{model_name}> for {model_framework}, on {"CPU" if use_cpu else "GPU"} ######')
		config = AutoConfig.from_pretrained(model_name, cache_dir=model_path) # Download configuration from S3 and cache.
		tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
		ModelManager.logger.info(f'###### <{model_name}> loaded ######')
		return {
			'pipeline': pipeline(model_type, 
				model=model_name, 
				tokenizer=model_name, 
				framework=model_framework, 
				device=-1 if use_cpu else 0
			),
			'tokenizer': tokenizer,
			'config': config,
			'text_template': hf_model_options.get('text_template',None),
		}

	@staticmethod
	def load_sbert_model(sbert_model_options):
		model_url = sbert_model_options['url']
		use_cuda = sbert_model_options.get('use_cuda',False)
		cache_dir = sbert_model_options.get('cache_dir',None)
		is_qa_model = ModelManager.sbert_is_qa_model(sbert_model_options)
		if is_qa_model:
			ModelManager.logger.info(f"## Loading sentence_transformers model <{model_url}> for QA, cuda support {use_cuda}...")
		else:
			ModelManager.logger.info(f"## Loading sentence_transformers model <{model_url}>, cuda support {use_cuda}...")
		sbert_model = SentenceTransformer(
			model_url, 
			device='cpu' if not use_cuda or not gpu_devices else 'cuda', 
			cache_folder=cache_dir if isinstance(cache_dir,str) and os.path.isdir(cache_dir) else None
		)

		get_input = lambda y: tuple(map(lambda x: x[0] if is_listable(x) else x, y))
		if is_qa_model:
			if model_url == 'nq-distilbert-base-v1':
				a_template = lambda x: (x[0],x[1])
				ctx_sbert_model = sbert_model
			else:
				a_template = lambda x: f"{x[0]} [SEP] {x[1]}"
				ctx_sbert_model = SentenceTransformer(
					model_url.replace('question_encoder','ctx_encoder'), 
					device='cpu' if not use_cuda or not gpu_devices else 'cuda', 
					cache_folder=cache_dir if isinstance(cache_dir,str) and os.path.isdir(cache_dir) else None
				)
			get_context = lambda y: tuple(map(lambda x: x[1] if is_listable(x) else '', y))
			q_module = lambda doc,**args: sbert_model.encode(get_input(doc),**args)
			a_module = lambda doc,**args: ctx_sbert_model.encode(tuple(map(a_template, zip(get_input(doc),get_context(doc)))),**args)
		else:
			q_module = a_module = lambda doc,**args: sbert_model.encode(get_input(doc),**args)
		ModelManager.logger.info('## SBERT model loaded')
		return {
			'question': q_module,
			'answer': a_module
		}
	
	def get_nlp_model(self):
		if ModelManager.__nlp_models.get(self.__spacy_model, None) is None:
			ModelManager.__nlp_models[self.__spacy_model] = ModelManager.load_nlp_model(self.__spacy_model, self.__use_cuda)
		return ModelManager.__nlp_models[self.__spacy_model]

	def get_tf_model(self):
		model_key = self.__tf_model_options['url']
		if ModelManager.__tf_embedders.get(model_key, None) is None:
			ModelManager.__tf_embedders[model_key] = ModelManager.load_tf_model(self.__tf_model_options)
		return ModelManager.__tf_embedders[model_key]

	@staticmethod
	def tf_is_qa_model(model_dict):
		return 'qa' in model_dict['url'].lower()

	@staticmethod
	def sbert_is_qa_model(model_dict):
		model_dict_url = model_dict['url'].lower()
		return 'question_encoder' in model_dict_url or 'qa' in model_dict_url

	def get_hf_model(self):
		model_key = (self.__hf_model_options['url'],self.__hf_model_options['type'])
		if ModelManager.__hf_embedders.get(model_key, None) is None:
			ModelManager.__hf_embedders[model_key] = ModelManager.load_hf_model(self.__hf_model_options)
		return ModelManager.__hf_embedders[model_key]

	def get_sbert_model(self):
		model_key = self.__sbert_model_options['url']
		if ModelManager.__sbert_embedders.get(model_key, None) is None:
			ModelManager.__sbert_embedders[model_key] = ModelManager.load_sbert_model(self.__sbert_model_options)
		return ModelManager.__sbert_embedders[model_key]

	def nlp(self, text_list, disable=None, n_threads=None, batch_size=None, with_cache=None):
		if not disable:
			disable = self.disable_spacy_component
		if not n_threads: # real multi-processing: https://git.readerbench.com/eit/prepdoc/blob/f8e93b6d0a346e9a53dac2e70e5f1712d40d6e1e/examples/parallel_parse.py
			n_threads = self.__n_threads
		if not batch_size:
			batch_size = self.__default_batch_size
		if with_cache is None:
			with_cache = self.__with_cache
		def fetch_fn(missing_text):
			self.logger.debug(f"Processing {len(missing_text)} texts with spacy and {'with' if with_cache else 'without'} cache")
			nlp = self.get_nlp_model()
			# if len(missing_text) == 1:
			# 	return [nlp(missing_text[0])]
			result = nlp.pipe(
				missing_text, 
				disable=disable, 
				batch_size=min(batch_size, math.ceil(len(missing_text)/n_threads)),
				n_process=min(n_threads, len(missing_text)), # The keyword argument n_threads on the .pipe methods is now deprecated, as the v2.x models cannot release the global interpreter lock. (Future versions may introduce a n_process argument for parallel inference via multiprocessing.) - https://spacy.io/usage/v2-1#incompat
			)
			# print('qq', result)
			return self.tqdm(result, total=len(missing_text))
		if not with_cache:
			if isinstance(text_list, types.GeneratorType):
				text_list = tuple(text_list)
			return list(fetch_fn(text_list))
		# print('kk', text_list)
		return self.get_cached_values(text_list, self.__spacy_cache, fetch_fn)

	def run_tf_embedding(self, inputs, norm=None, without_context=False, with_cache=None):
		tf_model = self.get_tf_model()
		batch_size = self.__tf_model_options.get('batch_size', self.__default_batch_size)
		if with_cache is None:
			with_cache = self.__tf_model_options.get('with_cache', self.__with_cache)
		def fetch_fn(missing_queries):
			self.logger.debug(f"Processing {len(missing_queries)} inputs with tf and {'with' if with_cache else 'without'} cache")
			# print(missing_queries)
			# Feed missing_queries into current tf graph
			len_batch_iter = math.ceil(len(missing_queries)/batch_size)
			encoder = tf_model['question' if without_context else 'answer']
			if len_batch_iter == 1:
				embeddings = encoder(missing_queries)
			else:
				batch_iter = (
					missing_queries[i*batch_size:(i+1)*batch_size]
					for i in range(len_batch_iter)
				)
				ModelManager.logger.info(f'TF: Modelling {len(missing_queries)} sentences in {len_batch_iter} batches..')
				batched_embeddings = tuple(map(encoder, self.tqdm(batch_iter, total=len_batch_iter)))
				embeddings = np.concatenate(batched_embeddings, 0)
			# Normalize the embeddings, if required
			if norm is not None:
				embeddings = normalize(embeddings, norm=norm)
			return embeddings
		if not with_cache:
			if isinstance(inputs, types.GeneratorType):
				inputs = tuple(inputs)
			return list(fetch_fn(inputs))
		return self.get_cached_values(inputs, self.__tf_cache, fetch_fn, key_fn=lambda x:(x,without_context))

	def run_sbert_embedding(self, inputs, convert_to_tensor=False, without_context=False, with_cache=None):
		sbert_model = self.get_sbert_model()
		batch_size = self.__sbert_model_options.get('batch_size', self.__default_batch_size)
		if with_cache is None:
			with_cache = self.__sbert_model_options.get('with_cache', self.__with_cache)
		def fetch_fn(missing_inputs):
			self.logger.debug(f"Processing {len(missing_inputs)} inputs with sbert and {'with' if with_cache else 'without'} cache")
			len_batch_iter = math.ceil(len(missing_inputs)/batch_size)
			encoder = sbert_model['question' if without_context else 'answer']
			if len_batch_iter == 1:
				return encoder(missing_inputs, convert_to_tensor=convert_to_tensor)
			batch_iter = (
				missing_inputs[i*batch_size:(i+1)*batch_size]
				for i in range(len_batch_iter)
			)
			ModelManager.logger.info(f'SBert: Modelling {len(missing_inputs)} sentences in {len_batch_iter} batches..')
			batched_embeddings = tuple(
				encoder(batch, convert_to_tensor=convert_to_tensor)
				for batch in self.tqdm(batch_iter, total=len_batch_iter)
			)
			return np.concatenate(batched_embeddings, 0)
		if not with_cache:
			if isinstance(inputs, types.GeneratorType):
				inputs = tuple(inputs)
			return list(fetch_fn(inputs))
		return self.get_cached_values(inputs, self.__sbert_cache, fetch_fn, key_fn=lambda x:(x,without_context))

	def run_hf_task(self, inputs, with_cache=None, **kwargs):
		hf_model = self.get_hf_model()
		batch_size = self.__hf_model_options.get('batch_size', self.__default_batch_size)
		if with_cache is None:
			with_cache = self.__hf_model_options.get('with_cache', self.__with_cache)
		def fetch_fn(missing_inputs):
			self.logger.debug(f"Processing {len(missing_inputs)} inputs with hf and {'with' if with_cache else 'without'} cache")
			templatise_text = (lambda x: hf_model['text_template'].replace('{txt}',x)) if hf_model['text_template'] else (lambda x:x)
			missing_inputs = tuple(map(templatise_text, missing_inputs))
			len_batch_iter = math.ceil(len(missing_inputs)/batch_size)
			if len_batch_iter == 1:
				return hf_model['pipeline'](missing_inputs, **kwargs)
			
			batch_iter = (
				missing_inputs[i*batch_size:(i+1)*batch_size] 
				for i in range(len_batch_iter)
			)
			ModelManager.logger.info(f'HF: Modelling {len(missing_inputs)} sentences in {len_batch_iter} batches..')
			batched_results = (
				hf_model['pipeline'](batch, **kwargs)
				for batch in self.tqdm(batch_iter, total=len_batch_iter)
				# for batch in batch_list
			)
			return flatten(batched_results, as_list=True)
		cache_key = '.'.join(map(lambda x: '='.join(map(str,x)), sorted(kwargs.items(), key=lambda x:x[0])))
		if not with_cache:
			if isinstance(inputs, types.GeneratorType):
				inputs = tuple(inputs)
			return fetch_fn(inputs)
		return self.get_cached_values(inputs, self.__hf_cache, fetch_fn, key_fn=lambda x: '.'.join((cache_key,x)))

	def get_default_embedder(self):
		if self.__tf_model_options:
			return self.run_tf_embedding
		elif self.__sbert_model_options:
			return self.run_sbert_embedding
		return None

	def get_default_similarity_fn(self):
		if self.__tf_model_options:
			# np.inner == lambda x,y: np.matmul(x,np.transpose(y))
			return np.inner
		elif self.__sbert_model_options:
			return cosine_similarity
		return None

	def get_default_embedding(self, text_list, without_context=False, with_cache=None):
		embedding_fn = self.get_default_embedder()
		assert embedding_fn is not None, 'cannot find a proper embedding_fn, please specify a TF or SBERT model'
		return embedding_fn(text_list, without_context=without_context, with_cache=with_cache)

	def get_default_similarity(self, a, b):
		similarity_fn = self.get_default_similarity_fn()
		assert similarity_fn is not None, 'cannot find a proper similarity_fn'
		return similarity_fn(a,b)

	def get_element_wise_similarity(self, source_list, target_list, source_without_context=False, target_without_context=False, get_embedding_fn=None, get_similarity_fn=None, with_cache=None):
		if get_embedding_fn is None:
			get_embedding_fn = self.get_default_embedding
		if get_similarity_fn is None:
			get_similarity_fn = self.get_default_similarity
		assert len(source_list)==len(target_list), 'len(source_list)!=len(target_list)'
		source_embeddings = get_embedding_fn(source_list, without_context=source_without_context, with_cache=with_cache)
		target_embeddings = get_embedding_fn(target_list, without_context=target_without_context, with_cache=with_cache)
		return [
			float(get_similarity_fn([a],[b])[0][0]) if a is not None and b is not None else 0
			for a,b in zip(source_embeddings,target_embeddings)
		]

	def get_similarity_ranking(self, source_text_list, target_text_list, without_context=False, get_embedding_fn=None, get_similarity_fn=None, with_cache=None):
		if get_embedding_fn is None:
			get_embedding_fn = self.get_default_embedding
		if get_similarity_fn is None:
			get_similarity_fn = self.get_default_similarity
		source_embeddings = get_embedding_fn(source_text_list, without_context=without_context, with_cache=with_cache)
		target_embeddings = get_embedding_fn(target_text_list, without_context=False, with_cache=with_cache)
		similarity_vec = get_similarity_fn(source_embeddings,target_embeddings)
		return np.argsort(similarity_vec, kind='stable', axis=-1), similarity_vec

	def get_most_similar_idx_n_label(self, source_text_list, target_text_list, without_context=False, get_embedding_fn=None, get_similarity_fn=None, with_cache=None):
		if get_embedding_fn is None:
			get_embedding_fn = self.get_default_embedding
		if get_similarity_fn is None:
			get_similarity_fn = self.get_default_similarity
		similarity_vec = get_similarity_fn(
			get_embedding_fn(source_text_list, without_context=without_context, with_cache=with_cache),
			get_embedding_fn(target_text_list, without_context=False, with_cache=with_cache)
		)
		argmax_list = np.argmax(similarity_vec, axis=-1)
		return list(zip(argmax_list, np.take(similarity_vec,argmax_list)))

	def remove_similar_labels(self, tuple_list, threshold=0.97, key=None, without_context=False, get_embedding_fn=None, get_similarity_fn=None, sort_by_conformity=False, with_cache=None):
		if key is None:
			key = lambda x: x[0] if isinstance(x, (list,tuple)) else x
		if get_embedding_fn is None:
			get_embedding_fn = self.get_default_embedding
		if get_similarity_fn is None:
			get_similarity_fn = self.get_default_similarity
		value_list = tuple(map(key,tuple_list))
		similarity_vec = get_similarity_fn(
			get_embedding_fn(value_list, without_context=without_context, with_cache=with_cache),
			get_embedding_fn(value_list, without_context=without_context, with_cache=with_cache)
		)
		
		if sort_by_conformity:
			sorted_idx_vec = np.argsort(np.sum(similarity_vec, axis=-1), kind='stable', axis=-1)[::-1]
			sorted_similarity_vec = np.take(similarity_vec, sorted_idx_vec, axis=0)
			return [
				tuple_list[j]
				for i,j in enumerate(sorted_idx_vec.tolist())
				if not np.any(sorted_similarity_vec[i][:i] >= threshold)
			]
		# print('remove_similar_labels')
		# for i,_ in enumerate(value_list):
		# 	for j,v in enumerate(similarity_vec[i][:i]):
		# 		if v < threshold:
		# 			print(0, v, key(tuple_list[i]))
		# 			print(1, v, key(tuple_list[j]))
		# 			break
		result_list = []
		for i,v in enumerate(tuple_list):
			if not np.any(similarity_vec[i][:i] >= threshold):
				result_list.append(v)
			else: # ignore this element in next comparisons
				similarity_vec[:,i] = 0
		return result_list

	def sort_labels_by_conformity(self, tuple_list, key=None, without_context=False, get_embedding_fn=None, get_similarity_fn=None, return_conformity=False, with_cache=None):
		if key is None:
			key = lambda x: x[0] if isinstance(x, (list,tuple)) else x
		if get_embedding_fn is None:
			get_embedding_fn = self.get_default_embedding
		if get_similarity_fn is None:
			get_similarity_fn = self.get_default_similarity
		value_list = tuple(map(key,tuple_list))
		similarity_vec = get_similarity_fn(
			get_embedding_fn(value_list, without_context=without_context, with_cache=with_cache),
			get_embedding_fn(value_list, without_context=without_context, with_cache=with_cache)
		)
		similarity_vec = np.mean(similarity_vec, axis=-1)
		sorted_idx_vec = np.argsort(similarity_vec, kind='stable', axis=-1)
		sorted_label_list = np.take(tuple_list, sorted_idx_vec, axis=0).tolist()
		if not return_conformity:
			return sorted_label_list
		sorted_conformity_list = np.take(similarity_vec, sorted_idx_vec, axis=0).tolist()
		sorted_conformity_iter = map(float, sorted_conformity_list)
		return list(zip(sorted_label_list, sorted_conformity_iter))

	def filter_by_similarity_to_target(self, source_tuple_list, target_tuple_list, threshold=0.97, source_key=None, target_key=None, source_without_context=False, target_without_context=False, get_embedding_fn=None, get_similarity_fn=None, with_cache=None):
		if source_key is None:
			source_key = lambda x: x[0] if isinstance(x, (list,tuple)) else x
		if target_key is None:
			target_key = lambda x: x[0] if isinstance(x, (list,tuple)) else x
		if get_embedding_fn is None:
			get_embedding_fn = self.get_default_embedding
		if get_similarity_fn is None:
			get_similarity_fn = self.get_default_similarity
		similarity_vec = get_similarity_fn(
			get_embedding_fn(tuple(map(source_key,source_tuple_list)), without_context=source_without_context, with_cache=with_cache),
			get_embedding_fn(tuple(map(target_key,target_tuple_list)), without_context=target_without_context, with_cache=with_cache)
		)
		# print('filter_by_similarity_to_target')
		# for i,_ in enumerate(source_tuple_list):
		# 	for j,v in enumerate(similarity_vec[i]):
		# 		if v < threshold:
		# 			print(0, v, source_key(source_tuple_list[i]))
		# 			print(1, v, target_key(target_tuple_list[j]))
		# 			break
		return [
			v
			for i,v in enumerate(source_tuple_list)
			if not np.any(similarity_vec[i] >= threshold)
		]

	# def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
	# 	import nltk  # $ pip install nltk
	# 	try:
	# 		from nltk.corpus import cmudict
	# 	except OSError:
	# 		print('Downloading nltk::cmudict\n'
	# 			"(don't worry, this will only happen once)")
	# 		nltk.download('cmudict')
	# 		from nltk.corpus import cmudict
	# 	for syllables in pronunciations.get(word, []):
	# 		return syllables[0][-1].isdigit()  # use only the first one

	def resolve_texts_coreferences(self, txt_list, with_cache=False):
		new_txt_list = []
		for doc in self.nlp(txt_list, with_cache=with_cache):
			if not doc:
				new_txt_list.append('')
				continue
			token_text_list = []
			last_token_final_idx = 0
			# last_was_indefinite_article = False
			for token in doc:
				# if last_was_indefinite_article and starts_with_vowel_sound(token.text.casefold()):
				# 	token_text_list.append('n')
				# last_was_indefinite_article = False
				if last_token_final_idx != token.idx:
					token_text_list.append(' ')
				last_token_final_idx = token.idx + len(token.text)
				# Remove determiners: demonstrative, possessive
				if token.dep_=='det' and (token.lemma_.casefold() in ('that', 'those', 'this', 'these')):
					token_text_list.append('the')
					# token_text_list.append('a')
					# last_was_indefinite_article = True
				elif token.pos_ == 'PRON' and token.dep_ == 'poss':
					token_text_list.append('the')
				else:
					token_text_list.append(token.text)
			new_txt_list.append(''.join(token_text_list))
		return new_txt_list

# mm = ModelManager({'tf_model': {
# 		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
# 		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
# 		# 'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
# 		'use_cuda': True,
# 		# 'with_cache': True,
# 	}})

# d = mm.nlp(['This is your best thing for him and you know it.'])[0]
# for t in d:
# 	print(t.lemma_)
