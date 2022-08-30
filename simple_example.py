import json
import os
import sys

from doxpy.models.knowledge_extraction.ontology_builder import OntologyBuilder
from doxpy.models.estimation.explainability_estimator import ExplainabilityEstimator
from doxpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from doxpy.models.reasoning.question_answerer import QuestionAnswerer
from doxpy.misc.doc_reader import load_or_create_cache
from doxpy.misc.graph_builder import get_betweenness_centrality, save_graphml, get_concept_set, get_concept_description_dict
from doxpy.misc.jsonld_lib import *

PHI = [
	"Angina happens when some part of your heart doesn't get enough oxygen",
]
RELEVANT_ASPECTS = [
	"my:heart",
	"my:stroke",
	"my:vessel",
	"my:disease",
	"my:angina",
	"my:symptom",
]

################ Configuration ################
ARCHETYPE_FITNESS_OPTIONS = {
	'only_overview_exploration': False,
	'answer_pertinence_threshold': 0.55, 
	'answer_to_question_similarity_threshold': 0.9502, 
	'answer_to_answer_similarity_threshold': 0.9502, 
}
OVERVIEW_OPTIONS = {
	'answer_pertinence_threshold': None, # default is None
	'answer_to_question_similarity_threshold': None, # default is 0.9502
	'answer_to_answer_similarity_threshold': None, # default is 0.9502
	'minimise': False,
	'sort_archetypes_by_relevance': False,
	'set_of_archetypes_to_consider': None, # set(['why','how'])
	'answer_horizon': 10,
	'remove_duplicate_answer_sentences': True,

	'top_k': 100,
	'include_super_concepts_graph': False, 
	'include_sub_concepts_graph': True, 
	'add_external_definitions': False, 
	'consider_incoming_relations': True,
	'tfidf_importance': 0,
}

KG_MANAGER_OPTIONS = {
	# 'spacy_model': 'en_core_web_trf',
	# 'n_threads': 1,
	# 'use_cuda': True,
	'with_cache': False,
	'with_tqdm': False,
}

QA_EXTRACTOR_OPTIONS = {
	# 'models_dir': '/home/toor/Desktop/data/models', 
	'models_dir': '/Users/toor/Documents/University/PhD/Project/YAI/code/libraries/QuAnsX/data/models', 
	'use_cuda': True,

	'sbert_model': {
		'url': 'facebook-dpr-question_encoder-multiset-base', # model for paraphrase identification
		'cache_dir': '/Users/toor/Documents/Software/DLModels/sb_cache_dir',
		'use_cuda': True,
	},
}

ONTOLOGY_BUILDER_DEFAULT_OPTIONS = {
	'spacy_model': 'en_core_web_trf',
	'n_threads': 1,
	'use_cuda': True,

	'max_syntagma_length': None,
	'lemmatize_label': False,

	'default_similarity_threshold': 0.75,
	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
		# 'cache_dir': '/public/francesco_sovrano/DoX/Scripts/.env',
		'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir',
		'use_cuda': False,
	},
	'with_centered_similarity': True,
}

CONCEPT_CLASSIFIER_DEFAULT_OPTIONS = {
	# 'spacy_model': 'en_core_web_trf',
	# 'n_threads': 1,
	# 'use_cuda': True,

	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
		# 'cache_dir': '/public/francesco_sovrano/DoX/Scripts/.env',
		'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir',
		'use_cuda': False,
	},
	'with_centered_similarity': True,
	'default_similarity_threshold': 0.75,
	# 'default_tfidf_importance': 3/4,
}

SENTENCE_CLASSIFIER_DEFAULT_OPTIONS = {
	# 'spacy_model': 'en_core_web_trf',
	# 'n_threads': 1,
	# 'use_cuda': True,

	# 'tf_model': {
	# 	# 'url': 'https://tfhub.dev/google/universal-sentence-encoder-qa2/3', # English QA
	# 	'url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3', # Multilingual QA # 16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian)
	# 	# 'url': 'https://tfhub.dev/google/LAReQA/mBERT_En_En/1',
	# 	'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
	# 	'use_cuda': True,
	# }, 
	'sbert_model': {
		'url': 'facebook-dpr-question_encoder-multiset-base', # model for paraphrase identification
		# 'cache_dir': '/public/francesco_sovrano/DoX/Scripts/.env',
		'cache_dir': '/Users/toor/Documents/Software/DLModels/sb_cache_dir',
		'use_cuda': True,
	},
	'with_centered_similarity': False,
	'with_topic_scaling': False,
	'with_stemmed_tfidf': False,
	# 'default_tfidf_importance': 1/4,
}

################ Initialise data structures ################
print('Building Graph..')
explainable_information_graph = OntologyBuilder(ONTOLOGY_BUILDER_DEFAULT_OPTIONS).set_content_list(PHI).build()
# save_graphml(explainable_information_graph, 'knowledge_graph')
print('Graph size:', len(explainable_information_graph))
print("Graph's Clauses:", len(list(filter(lambda x: '{obj}' in x[1], explainable_information_graph))))
#############
print('Building Question Answerer..')
betweenness_centrality = get_betweenness_centrality(filter(lambda x: '{obj}' in x[1], explainable_information_graph))

###### QuestionAnswererEDUClause########################
kg_manager = KnowledgeGraphManager(KG_MANAGER_OPTIONS, explainable_information_graph)
qa = QuestionAnswerer( # Using qa_dict_list also for getting the archetype_fitness_dict might over-estimate the median pertinence of some archetypes (and in a different way for each), because the QA Extractor is set to prefer a higher recall to a higher precision.
	kg_manager= kg_manager, 
	concept_classifier_options= CONCEPT_CLASSIFIER_DEFAULT_OPTIONS, 
	sentence_classifier_options= SENTENCE_CLASSIFIER_DEFAULT_OPTIONS, 
	betweenness_centrality= betweenness_centrality,
)
########################################################

important_aspects = set(RELEVANT_ASPECTS)
print('Important explicandum aspects:', len(important_aspects))
print(json.dumps(list(important_aspects), indent=4))
explainability_estimator = ExplainabilityEstimator(qa)
#############
archetype_fitness_dict = explainability_estimator.get_archetype_fitness_dict(
	overview_options=OVERVIEW_OPTIONS,
	aspect_uri_list=list(important_aspects),
	**ARCHETYPE_FITNESS_OPTIONS
)
print('Archetype Fitness:', json.dumps(archetype_fitness_dict, indent=4))
dox = explainability_estimator.get_degree_of_explainability_from_archetype_fitness(archetype_fitness_dict)
print(f'DoX:', json.dumps(dox, indent=4))
weighted_degree_of_explainability = explainability_estimator.get_weighted_degree_of_explainability(dox, archetype_weight_dict=None)
print('Average DoX:', weighted_degree_of_explainability)
