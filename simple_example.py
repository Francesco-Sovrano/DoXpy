import json
import os
import sys

from doxpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor
from doxpy.models.estimation.dox_estimator import DoXEstimator
from doxpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from doxpy.models.reasoning.answer_retriever import AnswerRetriever
from doxpy.misc.doc_reader import load_or_create_cache
from doxpy.misc.graph_builder import get_betweenness_centrality, save_graphml, get_concept_set, get_concept_description_dict
from doxpy.misc.jsonld_lib import *

PHI = [ # Information whose explainability to assess
	"Angina happens when some part of your heart doesn't get enough oxygen",
]
EXPLANANDUM_ASPECTS = [ # A: the explanandum aspects
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
	'answer_to_question_max_similarity_threshold': 0.9502, 
	'answer_to_answer_max_similarity_threshold': 0.9502, 
}
OVERVIEW_OPTIONS = {
	'answer_pertinence_threshold': None, # default is None
	'answer_to_question_max_similarity_threshold': None, # default is 0.9502
	'answer_to_answer_max_similarity_threshold': None, # default is 0.9502
	'minimise': False,
	'sort_archetypes_by_relevance': False,
	'set_of_archetypes_to_consider': None, # set(['why','how'])
	'answer_horizon': None,
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

KG_BUILDER_DEFAULT_OPTIONS = {
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
explainable_information_graph = KnowledgeGraphExtractor(KG_BUILDER_DEFAULT_OPTIONS).set_content_list(PHI, remove_stopwords=False, remove_numbers=False, avoid_jumps=True).build()
# save_graphml(explainable_information_graph, 'knowledge_graph')
print('Graph size:', len(explainable_information_graph))
print("Graph's Clauses:", len(list(filter(lambda x: '{obj}' in x[1], explainable_information_graph))))
#############
print('Building Question Answerer..')
# betweenness_centrality = get_betweenness_centrality(filter(lambda x: '{obj}' in x[1], explainable_information_graph))
kg_manager = KnowledgeGraphManager(KG_MANAGER_OPTIONS, explainable_information_graph)
qa = AnswerRetriever( # Using qa_dict_list also for getting the archetype_fitness_dict might over-estimate the median pertinence of some archetypes (and in a different way for each), because the QA Extractor is set to prefer a higher recall to a higher precision.
	kg_manager= kg_manager, 
	concept_classifier_options= CONCEPT_CLASSIFIER_DEFAULT_OPTIONS, 
	sentence_classifier_options= SENTENCE_CLASSIFIER_DEFAULT_OPTIONS, 
	# betweenness_centrality= betweenness_centrality,
)
########################################################

### Get explanandum aspects
explanandum_aspect_list = EXPLANANDUM_ASPECTS
print('Important explicandum aspects:', len(explanandum_aspect_list))
print(json.dumps(explanandum_aspect_list, indent=4))

### Define archetypal questions
question_template_list = [ # Q: the archetypal questions
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
OVERVIEW_OPTIONS['question_generator'] = lambda question_template,concept_label: question_template.replace('{X}',concept_label)

### Initialise the DoX estimator
dox_estimator = DoXEstimator(qa)
### Estimate DoX
dox = dox_estimator.estimate(
	aspect_uri_iter=list(explanandum_aspect_list), 
	query_template_list=question_template_list, 
	archetype_fitness_options=ARCHETYPE_FITNESS_OPTIONS, 
	**OVERVIEW_OPTIONS
)
print(f'DoX:', json.dumps(dox, indent=4))
### Compute the average DoX
average_dox = dox_estimator.get_weighted_degree_of_explainability(dox, archetype_weight_dict=None)
print('Average DoX:', average_dox)
