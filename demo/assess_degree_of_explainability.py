from doxpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor
from doxpy.models.estimation.dox_estimator import DoXEstimator
from doxpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from doxpy.models.reasoning.answer_retriever import AnswerRetriever
from doxpy.misc.doc_reader import load_or_create_cache, DocParser
from doxpy.misc.graph_builder import get_betweenness_centrality, save_graphml, get_concept_set, get_concept_description_dict
from doxpy.misc.jsonld_lib import *
from doxpy.misc.utils import *

import json
import os
import sys
import logging
logger = logging.getLogger('doxpy')
logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)
# logger.addHandler(logging.StreamHandler(sys.stdout))

model_type, answer_pertinence_threshold, synonymity_threshold, explicandum_path, explainable_information_path, cache_path = sys.argv[1:]
answer_pertinence_threshold = float(answer_pertinence_threshold)
synonymity_threshold = float(synonymity_threshold)
if not os.path.exists(cache_path): os.mkdir(cache_path)

print('Assessing DoX of:', json.dumps(sys.argv[1:], indent=4))

########################################################################################################################################################################
################ Configuration ################
AVOID_JUMPS = True
# keep_the_n_most_similar_concepts = 2 
# query_concept_similarity_threshold = 0.75, 

ARCHETYPE_FITNESS_OPTIONS = {
	'one_answer_per_sentence': False,
	'answer_pertinence_threshold': answer_pertinence_threshold, 
	'answer_to_question_max_similarity_threshold': None,
	'answer_to_answer_max_similarity_threshold': 0.85,
}

KG_MANAGER_OPTIONS = {
	'spacy_model': 'en_core_web_trf',
	'n_threads': 1,
	'use_cuda': True,
	'with_cache': False,
	'with_tqdm': False,

	# 'min_triplet_len': 0,
	# 'max_triplet_len': float('inf'),
	# 'min_sentence_len': 0,
	# 'max_sentence_len': float('inf'),
	# 'min_paragraph_len': 0,
	# 'max_paragraph_len': 0, # do not use paragraphs for computing DoX
}

GRAPH_EXTRACTION_OPTIONS = {
	'add_verbs': False, 
	'add_predicates_label': False, 
	'add_subclasses': True, 
	'use_wordnet': False,
}

GRAPH_CLEANING_OPTIONS = {
	'remove_stopwords': False,
	'remove_numbers': False,
	'avoid_jumps': AVOID_JUMPS,
	'parallel_extraction': False,
}

GRAPH_BUILDER_OPTIONS = {
	'spacy_model': 'en_core_web_trf',
	'n_threads': 1,
	'use_cuda': True,

	'with_cache': False,
	'with_tqdm': False,

	'max_syntagma_length': None,
	'add_source': True,
	'add_label': True,
	'lemmatize_label': False,

	# 'default_similarity_threshold': 0.75,
	'default_similarity_threshold': 0,
	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
		# 'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
		# 'use_cuda': True,
		# 'with_cache': True,
		# 'batch_size': 100,
	},
}

CONCEPT_CLASSIFIER_OPTIONS = {
	'spacy_model': 'en_core_web_trf',
	'n_threads': 1,
	'use_cuda': True,

	'default_batch_size': 20,
	'with_tqdm':False,
	'with_cache': True,

	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
		# 'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
	},
	# 'sbert_model': {
	# 	'url': 'all-MiniLM-L12-v2',
	# 	'use_cuda': True,
	# },
	'default_similarity_threshold': synonymity_threshold,
	# 'with_stemmed_tfidf': True,
	'default_tfidf_importance': 0,
}

SENTENCE_CLASSIFIER_OPTIONS = {
	'spacy_model': 'en_core_web_trf',
	'n_threads': 1,
	'use_cuda': True,

	# 'default_batch_size': 100,
	'with_tqdm': False,
	'with_cache': False,
	
	'default_tfidf_importance': 0,
}

if model_type == 'tf':
	SENTENCE_CLASSIFIER_OPTIONS['tf_model'] = {
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder-qa/3', # English QA
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3', # Multilingual QA # 16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian)
		# 'url': 'https://tfhub.dev/google/LAReQA/mBERT_En_En/1',
		# 'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
		# 'use_cuda': True,
		'with_cache': True,
	}
else:
	SENTENCE_CLASSIFIER_OPTIONS['sbert_model'] = {
		'url': 'multi-qa-MiniLM-L6-cos-v1', # model for paraphrase identification
		# 'use_cuda': True,
		'with_cache': True,
	}

if __name__=='__main__':
	################ Initialise data structures ################
	explicandum_graph_cache = os.path.join(cache_path,f"cache_explicandum_graph_lemma-{GRAPH_BUILDER_OPTIONS['lemmatize_label']}.pkl")
	explainable_information_graph_cache = os.path.join(cache_path,f"cache_explainable_information_graph_lemma-{GRAPH_BUILDER_OPTIONS['lemmatize_label']}.pkl")
	edu_graph_cache = os.path.join(cache_path,f"cache_edu_graph.pkl")
	betweenness_centrality_cache = os.path.join(cache_path,'cache_betweenness_centrality.pkl')

	qa_cache = os.path.join(cache_path,'cache_qa_embedder.pkl')
	qa_edu_cache = os.path.join(cache_path,'cache_qa_edu_embedder.pkl')
	qa_disco_cache = os.path.join(cache_path,'cache_qa_disco_embedder.pkl')
	########################################################################
	print('Building Graph..')
	explicandum_graph = load_or_create_cache(
		explicandum_graph_cache, 
		lambda: KnowledgeGraphExtractor(GRAPH_BUILDER_OPTIONS).set_documents_path(explicandum_path, remove_stopwords=True, remove_numbers=True, avoid_jumps=True).build(**GRAPH_EXTRACTION_OPTIONS)
	)
	save_graphml(explicandum_graph, os.path.join(cache_path,'explicandum_graph'))
	print('Explicandum Graph size:', len(explicandum_graph))
	print("Explicandum Graph clauses:", len(list(filter(lambda x: '{obj}' in x[1], explicandum_graph))))
	explainable_information_graph = load_or_create_cache(
		explainable_information_graph_cache, 
		lambda: KnowledgeGraphExtractor(GRAPH_BUILDER_OPTIONS).set_documents_path(explainable_information_path, **GRAPH_CLEANING_OPTIONS).build(**GRAPH_EXTRACTION_OPTIONS)
	)
	save_graphml(explainable_information_graph, os.path.join(cache_path,'explainable_information_graph'))
	print('Explainable Information Graph size:', len(explainable_information_graph))
	print("Explainable Information Graph clauses:", len(list(filter(lambda x: '{obj}' in x[1], explainable_information_graph))))
	#############
	print('Building Question Answerer..')
	# betweenness_centrality = load_or_create_cache(
	# 	betweenness_centrality_cache, 
	# 	lambda: get_betweenness_centrality(filter(lambda x: '{obj}' in x[1], explainable_information_graph))
	# )
	kg_manager = KnowledgeGraphManager(KG_MANAGER_OPTIONS, explainable_information_graph)
	qa = AnswerRetriever( # Using qa_dict_list also for getting the archetype_fitness_dict might over-estimate the median pertinence of some archetypes (and in a different way for each), because the QA Extractor is set to prefer a higher recall to a higher precision.
		kg_manager= kg_manager, 
		concept_classifier_options= CONCEPT_CLASSIFIER_OPTIONS, 
		sentence_classifier_options= SENTENCE_CLASSIFIER_OPTIONS, 
		# betweenness_centrality= betweenness_centrality,
	)
	qa.load_cache(qa_cache)

	########################################################################################################################################################################
	########################################################################################################################################################################
	########################################################################################################################################################################

	### Get explanandum aspects
	explanandum_aspect_list = get_concept_description_dict(graph=explicandum_graph, label_predicate=HAS_LABEL_PREDICATE, valid_concept_filter_fn=lambda x: '{obj}' in x[1]).keys()
	explanandum_aspect_list = list(explanandum_aspect_list)
	print('Important explicandum aspects:', len(explanandum_aspect_list))
	print(json.dumps(explanandum_aspect_list, indent=4))

	### Define archetypal questions
	question_template_list = [
		##### AMR
		'What is {X}?',
		'Who is {X}?',
		'How is {X}?',
		'Where is {X}?',
		'When is {X}?',
		'Which {X}?',
		'Whose {X}?',
		'Why {X}?',
		##### Discourse Relations
		'In what manner is {X}?', # (25\%),
		'What is the reason for {X}?', # (19\%),
		'What is the result of {X}?', # (16\%),
		'What is an example of {X}?', # (11\%),
		'After what is {X}?', # (7\%),
		'While what is {X}?', # (6\%),
		'In what case is {X}?', # (3),
		'Despite what is {X}?', # (3\%),
		'What is contrasted with {X}?', # (2\%),
		'Before what is {X}?', # (2\%),
		'Since when is {X}?', # (2\%),
		'What is similar to {X}?', # (1\%),
		'Until when is {X}?', # (1\%),
		'Instead of what is {X}?', # (1\%),
		'What is an alternative to {X}?', # ($\leq 1\%$),
		'Except when it is {X}?', # ($\leq 1\%$),
		'{X}, unless what?', # ($\leq 1\%$).
	]

	### Define a question generator
	question_generator = lambda question_template,concept_label: question_template.replace('{X}',concept_label)

	### Initialise the DoX estimator
	dox_estimator = DoXEstimator(qa)
	### Estimate DoX
	dox = dox_estimator.estimate(
		aspect_uri_iter=list(explanandum_aspect_list), 
		query_template_list=question_template_list, 
		question_generator=question_generator,
		**ARCHETYPE_FITNESS_OPTIONS, 
	)
	print(f'DoX:', json.dumps(dox, indent=4))
	### Compute the average DoX
	# archetype_weight_dict = {
	# 	'why': 1,
	# 	'how': 0.9,
	# 	'what-for': 0.75,
	# 	'what': 0.75,
	# 	'what-if': 0.6,
	# 	'when': 0.5,
	# }
	average_dox = dox_estimator.get_weighted_degree_of_explainability(dox, archetype_weight_dict=None)
	print('Average DoX:', average_dox)
