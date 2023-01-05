from doxpy.models.knowledge_extraction.knowledge_graph_builder import KnowledgeGraphBuilder
from doxpy.models.estimation.explainability_estimator import ExplainabilityEstimator
from doxpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from doxpy.models.reasoning.question_answerer import QuestionAnswerer
from doxpy.misc.doc_reader import load_or_create_cache, DocParser
from doxpy.misc.graph_builder import get_betweenness_centrality, save_graphml, get_concept_set, get_concept_description_dict
from doxpy.misc.jsonld_lib import *
from doxpy.misc.utils import *

import json
import os
import sys

model_type, answer_pertinence_threshold, explicandum_path, explainable_information_path, cache_path = sys.argv[1:]
answer_pertinence_threshold = float(answer_pertinence_threshold)
if not os.path.exists(cache_path): os.mkdir(cache_path)

print('Assessing DoX of:', json.dumps(sys.argv[1:], indent=4))
# archetype_weight_dict = {
# 	'why': 1,
# 	'how': 0.9,
# 	'what-for': 0.75,
# 	'what': 0.75,
# 	'what-if': 0.6,
# 	'when': 0.5,
# }

################ Configuration ################
AVOID_JUMPS = True
# keep_the_n_most_similar_concepts = 2 
# query_concept_similarity_threshold = 0.75, 

OVERVIEW_OPTIONS = {
	'answer_horizon': None,
	'question_horizon': None,
	######################
	## QuestionAnswerer stuff
	'tfidf_importance': 0,
	'answer_pertinence_threshold': answer_pertinence_threshold, 
	'answer_to_question_max_similarity_threshold': None,
	'answer_to_answer_max_similarity_threshold': 0.85,
	'use_weak_pointers': False,
	# 'top_k': 100,
	# 'filter_fn': OQA_OPTIONS['filter_fn'],
	######################
	'include_super_concepts_graph': False, 
	'include_sub_concepts_graph': True, 
	'consider_incoming_relations': True,
	'minimise': False, 
	######################
	'sort_archetypes_by_relevance': False, 
}

ARCHETYPE_FITNESS_OPTIONS = {
	'one_answer_per_sentence': False,
	'answer_pertinence_threshold': None, 
	'answer_to_question_max_similarity_threshold': None,
	'answer_to_answer_max_similarity_threshold': None,
}


QA_EXTRACTOR_OPTIONS = {
	'models_dir': '/home/toor/Desktop/data/models', 
	# 'models_dir': '/Users/toor/Documents/University/PhD/Project/YAI/code/libraries/QuAnsX/data/models', 
	# 'use_cuda': True,

	'sbert_model': {
		'url': 'facebook-dpr-question_encoder-multiset-base', # model for paraphrase identification
		# 'cache_dir': '/public/francesco_sovrano/DoX/Scripts/.env',
		# 'use_cuda': True,
	},
}

KG_MANAGER_OPTIONS = {
	'spacy_model': 'en_core_web_trf',
	'n_threads': 1,
	# 'use_cuda': True,
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
	'parallel_extraction': True,
}

GRAPH_BUILDER_OPTIONS = {
	'spacy_model': 'en_core_web_trf',
	'n_threads': 1,
	# 'use_cuda': True,

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
	'with_centered_similarity': True,
}

CONCEPT_CLASSIFIER_OPTIONS = {
	'spacy_model': 'en_core_web_trf',
	'n_threads': 1,
	# 'use_cuda': True,

	'default_batch_size': 20,
	'with_tqdm':False,

	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
		# 'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
	},
	# 'sbert_model': {
	# 	'url': 'all-MiniLM-L12-v2',
	# 	'use_cuda': True,
	# },
	'with_centered_similarity': True,
	'default_similarity_threshold': 0.75,
	# 'default_tfidf_importance': 3/4,
	'default_tfidf_importance': 0,
}

SENTENCE_CLASSIFIER_OPTIONS = {
	'spacy_model': 'en_core_web_trf',
	'n_threads': 1,
	# 'use_cuda': True,

	# 'default_batch_size': 100,
	'with_tqdm': False,
	'with_cache': False,
	
	'with_centered_similarity': False,
	# 'with_topic_scaling': False,
	'with_stemmed_tfidf': True,
	'default_tfidf_importance': 1/2,
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
	lambda: KnowledgeGraphBuilder(GRAPH_BUILDER_OPTIONS).set_documents_path(explicandum_path, remove_stopwords=True, remove_numbers=True, avoid_jumps=True).build(**GRAPH_EXTRACTION_OPTIONS)
)
save_graphml(explicandum_graph, os.path.join(cache_path,'explicandum_graph'))
print('Explicandum Graph size:', len(explicandum_graph))
print("Explicandum Graph clauses:", len(list(filter(lambda x: '{obj}' in x[1], explicandum_graph))))
explainable_information_graph = load_or_create_cache(
	explainable_information_graph_cache, 
	lambda: KnowledgeGraphBuilder(GRAPH_BUILDER_OPTIONS).set_documents_path(explainable_information_path, **GRAPH_CLEANING_OPTIONS).build(**GRAPH_EXTRACTION_OPTIONS)
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
qa = QuestionAnswerer( # Using qa_dict_list also for getting the archetype_fitness_dict might over-estimate the median pertinence of some archetypes (and in a different way for each), because the QA Extractor is set to prefer a higher recall to a higher precision.
	kg_manager= kg_manager, 
	concept_classifier_options= CONCEPT_CLASSIFIER_OPTIONS, 
	sentence_classifier_options= SENTENCE_CLASSIFIER_OPTIONS, 
	# betweenness_centrality= betweenness_centrality,
)
qa.load_cache(qa_cache)
########################################################

important_aspects = get_concept_description_dict(graph=explicandum_graph, label_predicate=HAS_LABEL_PREDICATE, valid_concept_filter_fn=lambda x: '{obj}' in x[1]).keys()
important_aspects = set(important_aspects)
print('Important explicandum aspects:', len(important_aspects))
print(json.dumps(list(important_aspects), indent=4))

# concepts_dict = qa.concept_classifier.get_concept_dict(
# 	doc_parser=DocParser().set_content_list(important_aspects),
# 	similarity_threshold=query_concept_similarity_threshold, 
# 	size=keep_the_n_most_similar_concepts,
# )
# # Group queries by concept_uri
# concept_uri_query_dict = {}
# # print(json.dumps(concepts_dict, indent=4))
# for concept_label, concept_count_dict in concepts_dict.items():
# 	for concept_similarity_dict in itertools.islice(unique_everseen(concept_count_dict["similar_to"], key=lambda x: x["id"]), max(1,keep_the_n_most_similar_concepts)):
# 		concept_uri = concept_similarity_dict["id"]
# 		concept_query_set = concept_uri_query_dict.get(concept_uri,None)
# 		if concept_query_set is None:
# 			concept_query_set = concept_uri_query_dict[concept_uri] = set()
# 		concept_query_set.update((
# 			sent_dict["paragraph_text"]
# 			for sent_dict in concept_count_dict["source_list"]
# 		))
# print('Concept-URI-Label dict:', len(concept_uri_query_dict))
# print(concept_uri_query_dict)

explainability_estimator = ExplainabilityEstimator(qa)
#############
# if explainability_estimator.aspect_archetype_answers_dict is None:
# 	explainability_estimator.extract_archetypal_answers_per_aspect(**dict(OVERVIEW_OPTIONS))
# 	explainability_estimator.store_cache(qa_cache)
aspect_archetype_answers_dict = explainability_estimator.extract_archetypal_answers_per_aspect(
	aspect_uri_iter=list(important_aspects),
	only_overview_exploration=False,
	**OVERVIEW_OPTIONS
)
# print('Aspect-Archetype-Answers dict:', json.dumps(aspect_archetype_answers_dict, indent=4))
archetype_fitness_dict = explainability_estimator.get_archetype_fitness_dict(
	aspect_archetype_answers_dict,
	ARCHETYPE_FITNESS_OPTIONS
)
print('Archetype Fitness:', json.dumps(archetype_fitness_dict, indent=4))
dox = explainability_estimator.get_degree_of_explainability_from_archetype_fitness(archetype_fitness_dict)
print(f'DoX:', json.dumps(dox, indent=4))
weighted_degree_of_explainability = explainability_estimator.get_weighted_degree_of_explainability(dox, archetype_weight_dict=None)
print('Average DoX:', weighted_degree_of_explainability)
#############
# qa.store_cache(qa_cache)

