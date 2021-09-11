from knowpy.misc.doc_reader import DocParser
from knowpy.models.model_manager import ModelManager
from knowpy.models.knowledge_extraction.ontology_builder import OntologyBuilder
from knowpy.models.classification.concept_classifier import ConceptClassifier
from knowpy.models.classification.sentence_classifier import SentenceClassifier
from knowpy.misc.adjacency_matrix import AdjacencyMatrix
from knowpy.misc.graph_builder import get_root_set, get_concept_set, get_predicate_set, get_object_set, get_connected_graph_list, get_ancestors, filter_graph_by_root_set, tuplefy, get_concept_description_dict, get_betweenness_centrality
from knowpy.misc.levenshtein_lib import remove_similar_labels, labels_are_similar, labels_are_contained
from knowpy.misc.jsonld_lib import *

import numpy as np
from collections import Counter
import re
import time
import json
from more_itertools import unique_everseen
import itertools
import wikipedia
try:
	from nltk.corpus import wordnet as wn
	from nltk.corpus import brown
except OSError:
	print('Downloading nltk::wordnet\n'
		"(don't worry, this will only happen once)")
	import nltk
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('wordnet')
	nltk.download('brown')
	from nltk.corpus import wordnet as wn
	from nltk.corpus import brown
from nltk import FreqDist
from collections import namedtuple

ArchetypePertinence = namedtuple('ArchetypePertinence',['archetype','pertinence'])
InformationUnit = namedtuple('InformationUnit',['unit','context'])
get_information_unit = lambda x: InformationUnit(x['abstract'], x['sentence'])

word_frequency_distribution = FreqDist(i.lower() for i in brown.words())
is_common_word = lambda w: word_frequency_distribution.freq(w) >= 1e-4
singlefy = lambda s: s.strip().replace("\n"," ")#.capitalize()
wh_elements = ['why','how','what','where','when','who','which','whose','whom']
wh_elements_regexp = re.compile('('+'|'.join(map(re.escape, wh_elements))+')', re.IGNORECASE)
is_not_wh_word = lambda x: re.match(wh_elements_regexp, x) is None # use match instead of search

class QuestionAnswerer:
	archetypal_questions_dict = {
		##### Descriptive
		# 'what': 'What is a description of {X}?',
		'what': 'What is {X}?',
		# 'what': 'What is {X}?',
		'who': 'Who {X}?',
		# 'whom': 'Whom {X}?',
		##### Causal + Justificatory
		'why': 'Why {X}?',
		# 'why-not': 'Why not {X}?',
		##### Counterfactual
		'what-if': 'What if {X}?',
		##### Teleological
		'what-for': 'What is {X} for?',
		# 'what-for': 'What is {X} for?',
		##### Expository
		'how': 'How {X}?',
		##### Quantitative
		'how-much': 'How much {X}?',
		# 'how-many': 'How many {X}?',
		##### Spatial
		'where': 'Where {X}?',
		##### Temporal
		'when': 'When {X}?',
		##### Medium
		'who-by': 'Who by {X}?',
		##### Extra
		'which': 'Which {X}?',
		'whose': 'Whose {X}?',
		##### Discourse Relations
		'Expansion.Manner': 'In what manner {X}?', # (25\%),
		'Contingency.Cause': 'What is the reason {X}?', # (19\%),
		'Contingency.Effect': 'What is the result of {X}?', # (16\%),
		'Expansion.Level-of-detail': 'What is an example of {X}?', # (11\%),
		'Temporal.Asynchronous.Consequence': 'After what {X}?', # (7\%),
		'Temporal.Synchronous': 'While what {X}?', # (6\%),
		'Contingency.Condition': 'In what case {X}?', # (3),
		'Comparison.Concession': 'Despite what {X}?', # (3\%),
		'Comparison.Contrast': 'What is contrasted with {X}?', # (2\%),
		'Temporal.Asynchronous.Premise': 'Before what {X}?', # (2\%),
		'Temporal.Asynchronous.Being': 'Since when {X}?', # (2\%),
		'Comparison.Similarity': 'What is similar to {X}?', # (1\%),
		'Temporal.Asynchronous.End': 'Until when {X}?', # (1\%),
		'Expansion.Substitution': 'Instead of what {X}?', # (1\%),
		'Expansion.Disjunction': 'What is an alternative to {X}?', # ($\leq 1\%$),
		'Expansion.Exception': 'Except when {X}?', # ($\leq 1\%$),
		'Contingency.Neg.-cond.': 'Unless what {X}?', # ($\leq 1\%$).
	}

	def __init__(self, graph, concept_classifier_options, sentence_classifier_options, answer_summariser_options=None, betweenness_centrality=None, log=False):
		self.graph = graph
		self.log = log
		self.betweenness_centrality = betweenness_centrality

		self.adjacency_matrix = AdjacencyMatrix(
			graph, 
			equivalence_relation_set=set([IN_SYNSET_PREDICATE,IS_EQUIVALENT]),
			is_sorted=True,
		)
		self.adjacency_matrix_no_equivalence = AdjacencyMatrix(
			graph, 
			equivalence_relation_set=set([IS_EQUIVALENT]),
			is_sorted=True,
		)
		# self.subclass_dict = QuestionAnswerer.get_predicate_dict(get_predicate_dict, SUBCLASSOF_PREDICATE)
		self.content_dict = self.adjacency_matrix.get_predicate_dict(CONTENT_PREDICATE, singlefy)
		self.source_dict = self.adjacency_matrix.get_predicate_dict(HAS_SOURCE_PREDICATE)
		self.label_dict = self.adjacency_matrix.get_predicate_dict(HAS_LABEL_PREDICATE, singlefy)

		self.content_dict_no_equiv = self.adjacency_matrix_no_equivalence.get_predicate_dict(CONTENT_PREDICATE, singlefy)
		self.source_span_dict_no_equiv = self.adjacency_matrix_no_equivalence.get_predicate_dict(HAS_SOURCE_SPAN_PREDICATE, singlefy)
		self.source_sentence_dict_no_equiv = self.adjacency_matrix_no_equivalence.get_predicate_dict(HAS_SOURCE_SENTENCE_PREDICATE, singlefy)
		self.source_dict_no_equiv = self.adjacency_matrix_no_equivalence.get_predicate_dict(HAS_SOURCE_PREDICATE)
		self.label_dict_no_equiv = self.adjacency_matrix_no_equivalence.get_predicate_dict(HAS_LABEL_PREDICATE, singlefy)
		self.source_label_dict_no_equiv = self.adjacency_matrix_no_equivalence.get_predicate_dict(HAS_SOURCE_LABEL_PREDICATE, singlefy)

		# Concept classification
		self.concept_classifier_options = concept_classifier_options
		self._concept_classifier = None
		# Sentence classification
		self.sentence_classifier_options = sentence_classifier_options
		self._sentence_classifier = None

		self._overview_aspect_set = None
		self._relevant_aspect_set = None

	@property
	def sentence_classifier(self):
		if self._sentence_classifier is None:
			self._sentence_classifier = SentenceClassifier(self.sentence_classifier_options)
			self._init_sentence_classifier()
		return self._sentence_classifier

	@property
	def concept_classifier(self):
		if self._concept_classifier is None:
			self._concept_classifier = ConceptClassifier(self.concept_classifier_options)
			self._init_concept_classifier()
		return self._concept_classifier

	@property
	def overview_aspect_set(self):
		if self._overview_aspect_set is None:
			self._overview_aspect_set = set(filter(self.is_overview_aspect, self.concept_classifier.ids))
			# Betweenness centrality quantifies the number of times a node acts as a bridge along the shortest path between two other nodes.
			if self.betweenness_centrality is not None:
				filtered_betweenness_centrality = dict(filter(lambda x: x[-1] > 0, self.betweenness_centrality.items()))
				self._overview_aspect_set &= filtered_betweenness_centrality.keys()
		return self._overview_aspect_set

	@property
	def relevant_aspect_set(self):
		if self._relevant_aspect_set is None:
			self._relevant_aspect_set = set(filter(self.is_relevant_aspect, self.concept_classifier.ids))
		return self._relevant_aspect_set

	def _init_sentence_classifier(self):
		print('Initialising Sentence Classifier..')
		# Setup Sentence Classifier
		abstract_iter, context_iter, original_triple_iter, source_id_iter = zip(*self.get_sourced_graph())
		id_doc_iter = tuple(zip(
			zip(original_triple_iter, source_id_iter), # id
			abstract_iter # doc
		))
		self.sentence_classifier.set_documents(id_doc_iter, tuple(context_iter))

	def _init_concept_classifier(self):
		print('Initialising Concept Classifier..')
		self.concept_classifier.set_concept_description_dict(
			get_concept_description_dict(
				graph= self.graph, 
				label_predicate= HAS_LABEL_PREDICATE, 
				valid_concept_filter_fn= lambda x: '{obj}' in x[1]
			)
		)
	
	def store_cache(self, cache_name):
		self.concept_classifier.store_cache(cache_name+'.concept_classifier.pkl')
		self.sentence_classifier.store_cache(cache_name+'.sentence_classifier.pkl')

	def load_cache(self, cache_name):
		if self._concept_classifier is None:
			self._concept_classifier = ConceptClassifier(self.concept_classifier_options)
			self._concept_classifier.load_cache(cache_name+'.concept_classifier.pkl')
			self._init_concept_classifier()
		else:
			self._concept_classifier.load_cache(cache_name+'.concept_classifier.pkl')
		if self._sentence_classifier is None:
			self._sentence_classifier = SentenceClassifier(self.sentence_classifier_options)
			self._sentence_classifier.load_cache(cache_name+'.sentence_classifier.pkl')
			self._init_sentence_classifier()
		else:
			self._sentence_classifier.load_cache(cache_name+'.sentence_classifier.pkl')

	def is_overview_aspect(self, aspect_uri):
		if len(self.source_dict_no_equiv.get(aspect_uri,[])) <= 1: # not enough material for an overview; topic already covered in the previous overview
			return False
		return self.is_uncommon_aspect(aspect_uri)

	def is_uncommon_aspect(self, aspect_uri):
		commonality_gen = (not is_common_word(label.lower()) for label in self.get_label_list(aspect_uri))
		return next(filter(lambda x: x, commonality_gen), None) is not None

	def is_relevant_aspect(self, aspect_uri):
		if not self.is_uncommon_aspect(aspect_uri):
			# print('is_common_aspect', aspect_uri)
			return False
		# concepts with no sources are leaves with no triplets: safely ignore them, they are included by a super-class and they are less relevant than it
		source_set = self.get_source_set(aspect_uri)
		if not source_set:
			# print('has_no_triplets', aspect_uri)
			return False
		# concepts with the same sources of one of their sub-classes are redundant as those with no sources
		aspect_set = set([aspect_uri])
		subclass_set = self.adjacency_matrix.get_predicate_chain(
			concept_set = aspect_set, 
			predicate_filter_fn = lambda x: x == SUBCLASSOF_PREDICATE, 
			direction_set = ['in'],
			depth = 0,
		) - aspect_set
		if not subclass_set:
			return True
		is_relevant = next(filter(lambda x: source_set-self.get_source_set(x), subclass_set), None) is not None
		# if not is_relevant:
		# 	print('is_redundant', aspect_uri)
		return is_relevant

	@staticmethod
	def get_question_answer_dict_quality(question_answer_dict, top=5):
		return {
			question: {
				# 'confidence': {
				# 	'best': answers[0]['confidence'],
				# 	'top_mean': sum(map(lambda x: x['confidence'], answers[:top]))/top,
				# },
				# 'syntactic_similarity': {
				# 	'best': answers[0]['syntactic_similarity'],
				# 	'top_mean': sum(map(lambda x: x['syntactic_similarity'], answers[:top]))/top,
				# },
				# 'semantic_similarity': {
				# 	'best': answers[0]['semantic_similarity'],
				# 	'top_mean': sum(map(lambda x: x['semantic_similarity'], answers[:top]))/top,
				# },
				'valid_answers_count': len(answers),
				'syntactic_similarity': answers[0]['syntactic_similarity'],
				'semantic_similarity': answers[0]['semantic_similarity'],
			}
			for question,answers in question_answer_dict.items()
		}

	def get_source_set(self, aspect_uri):
		return set((
			source_id
			for source_span_uri in self.source_span_dict_no_equiv.get(aspect_uri,[])
			for source_sentence_uri in self.source_sentence_dict_no_equiv[source_span_uri]
			for source_id in self.source_dict_no_equiv[source_sentence_uri]
		))

	def get_label_list(self, concept_uri, explode_if_none=True):
		if concept_uri in self.label_dict:
			return self.label_dict[concept_uri]
		if concept_uri.startswith(WORDNET_PREFIX):
			return list(map(lambda x: explode_concept_key(x.name()), wn.synset(concept_uri[len(WORDNET_PREFIX):]).lemmas()))
		return [explode_concept_key(concept_uri) if explode_if_none else '']

	def get_label(self, concept_uri, explode_if_none=True):
		label_list = self.get_label_list(concept_uri, explode_if_none)
		if concept_uri in self.label_dict:
			return min(label_list, key=len)
		return label_list[0]

	def get_sub_graph(self, uri, depth=None, predicate_filter_fn=lambda x: x != SUBCLASSOF_PREDICATE and '{obj}' not in x):
		uri_set = self.adjacency_matrix.get_predicate_chain(set([uri]), direction_set=['out'], depth=depth, predicate_filter_fn=predicate_filter_fn)
		return list(unique_everseen((
			(s,p,o)
			for s in uri_set
			for p,o in self.adjacency_matrix_no_equivalence.get_outcoming_edges_matrix(s)
		)))

	def get_aspect_graph(self, concept_uri, add_external_definitions=False, include_super_concepts_graph=False, include_sub_concepts_graph=False, consider_incoming_relations=False, depth=None, filter_fn=lambda x: x):
		concept_set = set([concept_uri])
		expanded_concept_set = set(concept_set)
		# Get sub-classes
		if include_sub_concepts_graph:
			sub_concept_set = self.adjacency_matrix.get_predicate_chain(
				concept_set = concept_set, 
				predicate_filter_fn = lambda x: x == SUBCLASSOF_PREDICATE, 
				direction_set = ['in'],
				depth = depth,
			)
			expanded_concept_set |= sub_concept_set
		# Get super-classes
		if include_super_concepts_graph:
			super_concept_set = self.adjacency_matrix.get_predicate_chain(
				concept_set = concept_set, 
				predicate_filter_fn = lambda x: x == SUBCLASSOF_PREDICATE, 
				direction_set = ['out'],
				depth = depth,
			)
			expanded_concept_set |= super_concept_set
		# expanded_concept_set = sorted(expanded_concept_set) # this would improve caching, later on
		# Add outcoming relations to concept graph
		expanded_aspect_graph = [
			(s,p,o)
			for s in expanded_concept_set
			for p,o in self.adjacency_matrix_no_equivalence.get_outcoming_edges_matrix(s)
		]
		# Add incoming relations to concept graph
		if consider_incoming_relations:
			expanded_aspect_graph += [
				(s,p,o)
				for o in expanded_concept_set
				for p,s in self.adjacency_matrix_no_equivalence.get_incoming_edges_matrix(o)
			]
		# print(concept_uri, json.dumps(expanded_aspect_graph, indent=4))
		expanded_aspect_graph = list(filter(filter_fn, expanded_aspect_graph))
		# Add external definitions
		if add_external_definitions:
			# Add wordnet's definition
			for equivalent_concept_uri in filter(lambda x: x.startswith(WORDNET_PREFIX), self.adjacency_matrix.equivalence_matrix.get(concept_uri,[])):
				synset = wn.synset(equivalent_concept_uri[len(WORDNET_PREFIX):]) # remove string WORDNET_PREFIX, 3 chars
				definition = synset.definition()
				expanded_aspect_graph.append((concept_uri,HAS_DEFINITION_PREDICATE,definition))
			# Add wikipedia's (short) definition
			# try:
			# 	definition = wikipedia.summary(
			# 		self.get_label(concept_uri), 
			# 		sentences=1, # short
			# 		chars=0,
			# 		auto_suggest=True, 
			# 		redirect=True
			# 	)
			# 	expanded_aspect_graph.append((concept_uri,HAS_DEFINITION_PREDICATE,definition))
			# except:
			# 	pass
		return expanded_aspect_graph

	def get_sourced_graph_from_aspect_graph(self, aspect_graph):
		def sourced_graph_with_rdf_triplets_gen():
			# Add full triplets
			for original_triple in aspect_graph:
				s,p,o = original_triple
				s_source_span_uri_set = self.source_span_dict_no_equiv.get(s,[])
				p_source_span_uri_set = set(self.source_span_dict_no_equiv.get(p,[]))
				o_source_span_uri_set = self.source_span_dict_no_equiv.get(o,[])
				for source_span_uri in p_source_span_uri_set.intersection(s_source_span_uri_set).intersection(o_source_span_uri_set):
					triplet_text = self.source_label_dict_no_equiv[source_span_uri][0]
					for source_sentence_uri in self.source_sentence_dict_no_equiv[source_span_uri]:
						sentence_text = self.source_label_dict_no_equiv[source_sentence_uri][0]
						for source_id in self.source_dict_no_equiv[source_sentence_uri]:
							paragraph_text = self.content_dict_no_equiv[source_id][0]
							yield (
								triplet_text, # triplet's text
								paragraph_text, # paragraph's text
								original_triple,
								source_id,
							)
							if paragraph_text != sentence_text:
								yield (
									triplet_text, # triplet's text
									sentence_text, # sentence's text
									original_triple,
									source_id,
								)
		# Add source to triples
		return list(unique_everseen(sourced_graph_with_rdf_triplets_gen()))

	def get_sourced_graph(self):
		def sourced_graph_gen():
			source_span_uri_iter = unique_everseen((
				source_span_uri 
				for source_span_uri_set in self.source_span_dict_no_equiv.values()
				for source_span_uri in source_span_uri_set 
			))
			for source_span_uri in source_span_uri_iter:
				triplet_text = self.source_label_dict_no_equiv[source_span_uri][0]
				for source_sentence_uri in self.source_sentence_dict_no_equiv[source_span_uri]:
					sentence_text = self.source_label_dict_no_equiv[source_sentence_uri][0]
					for source_id in self.source_dict_no_equiv[source_sentence_uri]:
						paragraph_text = self.content_dict_no_equiv[source_id][0]
						yield (
							triplet_text, # triplet's text
							paragraph_text, # paragraph's text
							(source_span_uri,source_id),
							source_id,
						)
						if paragraph_text != sentence_text:
							yield (
								triplet_text, # triplet's text
								sentence_text, # sentence's text
								(source_span_uri,source_sentence_uri),
								source_id,
							)
		# Add source to triples
		return list(unique_everseen(sourced_graph_gen(), key=lambda x:x[2]))

	def find_answers(self, query_list, question_answer_dict=None, answer_pertinence_threshold=0.55, tfidf_importance=None, answer_to_question_similarity_threshold=0.9502, answer_to_answer_similarity_threshold=0.9502, with_annotation=True, top_k=None):
		if question_answer_dict is None: question_answer_dict = {}
		def get_formatted_answer(answer):
			triple, source_uri = answer['id']
			sentence = answer['context']
			annotation = self.get_sub_graph(source_uri) if with_annotation and source_uri else None
			return {
				'abstract': answer['doc'],
				'confidence': answer['similarity'],
				'syntactic_similarity': answer['syntactic_similarity'],
				'semantic_similarity': answer['semantic_similarity'],
				'annotation': annotation,
				'sentence': sentence, 
				'triple': triple, 
				'source_id': source_uri if source_uri else sentence, 
			}

		# classify
		classification_dict_gen = self.sentence_classifier.classify(
			query_list=query_list, 
			similarity_type='weighted', 
			similarity_threshold=answer_pertinence_threshold, 
			without_context=True, 
			tfidf_importance=tfidf_importance,
			top_k=top_k,
		)
		# Add missing questions to question_answer_dict
		for question in query_list:
			if question not in question_answer_dict:
				question_answer_dict[question] = []
		# Format Answers
		for question, answer_iter in zip(query_list, classification_dict_gen):
			# answer_list = sorted(answer_iter, key=lambda x: x['similarity'], reverse=True)
			answer_list = list(unique_everseen(answer_iter, key=lambda x: (x['doc'],x['id'][-1])))
			if len(answer_list) == 0:
				continue
			# answers contained in the question are not valid
			if answer_to_question_similarity_threshold is not None:
				answer_list = self.sentence_classifier.filter_by_similarity_to_target(
					answer_list, 
					[question], 
					threshold=answer_to_question_similarity_threshold, 
					source_key=lambda a: a['doc'], 
					target_key=lambda q: q
				)
			# ignore similar-enough sentences with lower pertinence
			if answer_to_answer_similarity_threshold is not None:
				answer_list = self.sentence_classifier.remove_similar_labels(
					answer_list, 
					threshold=answer_to_answer_similarity_threshold, 
					key=lambda x: (x['doc'],x['context']),
					without_context=False,
				)
			question_answer_dict[question] += map(get_formatted_answer, answer_list)
		return question_answer_dict

	@staticmethod
	def get_answer_question_pertinence_dict(question_answer_dict, update_answers=False):
		answer_question_pertinence_dict = {}
		for question,answers in question_answer_dict.items():
			for a in answers:
				information_unit = get_information_unit(a)
				question_pertinence_list = answer_question_pertinence_dict.get(information_unit,None)
				if question_pertinence_list is None:
					question_pertinence_list = answer_question_pertinence_dict[information_unit] = []
				question_pertinence_list.append(ArchetypePertinence(question, a['semantic_similarity']))
		if update_answers:
			for question,answers in question_answer_dict.items():
				for a in answers:
					a['question_pertinence_set'] = answer_question_pertinence_dict[get_information_unit(a)]
		return answer_question_pertinence_dict

	@staticmethod
	def minimise_question_answer_dict(question_answer_dict):
		answer_question_dict = QuestionAnswerer.get_answer_question_pertinence_dict(question_answer_dict, update_answers=True)
		get_best_answer_archetype = lambda a: max(answer_question_dict[get_information_unit(a)], key=lambda y: y.pertinence).archetype
		return {
			question: list(filter(lambda x: get_best_answer_archetype(x)==question, answers))
			for question,answers in question_answer_dict.items()
		}

	@staticmethod
	def get_question_answer_overlap_dict(question_answer_dict):
		answer_question_dict = QuestionAnswerer.get_answer_question_pertinence_dict(question_answer_dict)
		get_question_iter = lambda q,a_list: filter(lambda x: x!=q, (answer_question_pertinence_dict[get_information_unit(a)].archetype for a in a_list))
		return {
			question: Counter(get_question_iter(question,answers))
			for question,answers in question_answer_dict.items()
		}

################################################################################################################################################

	def ask(self, question_list, answer_pertinence_threshold=None, tfidf_importance=None, answer_to_question_similarity_threshold=0.9502, answer_to_answer_similarity_threshold=0.9502, with_annotation=True, top_k=None, **args):
		question_answer_dict = self.find_answers(
			query_list= question_list, 
			answer_pertinence_threshold= answer_pertinence_threshold,
			tfidf_importance= tfidf_importance,
			answer_to_question_similarity_threshold= answer_to_question_similarity_threshold,
			answer_to_answer_similarity_threshold= answer_to_answer_similarity_threshold,
			with_annotation= with_annotation,
			top_k= top_k,
		)
		# Sort answers
		print(f'Sorting answers..')
		for question, formatted_answer_list in question_answer_dict.items():
			question_answer_dict[question] = list(unique_everseen(
				sorted(
					formatted_answer_list, 
					key=lambda x: x['confidence'], reverse=True
				), 
				key=lambda x: x["source_id"]
			))
		return question_answer_dict

	def get_concept_overview(self, query_template_list=None, concept_uri=None, concept_label=None, answer_pertinence_threshold=None, tfidf_importance=None, sort_archetypes_by_relevance=True, answer_to_question_similarity_threshold=0.9502, answer_to_answer_similarity_threshold=0.9502, minimise=True, with_annotation=True, top_k=None, **args):
		assert concept_uri, f"{concept_uri} is not a valid concept_uri"
		if query_template_list is None:
			query_template_list = list(self.archetypal_questions_dict.values())
		# set consider_incoming_relations to False with concept-centred generic questions (e.g. what is it?), otherwise the answers won't be the sought ones
		if not concept_label:
			concept_label = self.get_label(concept_uri)
		question_answer_dict = self.find_answers(
			query_list= tuple(map(lambda x:x.replace('{X}',concept_label), query_template_list)), 
			answer_pertinence_threshold= answer_pertinence_threshold,
			tfidf_importance= tfidf_importance,
			answer_to_question_similarity_threshold= None,
			answer_to_answer_similarity_threshold= answer_to_answer_similarity_threshold,
			with_annotation= with_annotation,
			top_k= top_k,
		)
		question_answer_values = question_answer_dict.values()
		# answers contained in the question are not valid
		if answer_to_question_similarity_threshold is not None:
			question_answer_values = (
				self.sentence_classifier.filter_by_similarity_to_target(
					answer_list, 
					[concept_label], 
					threshold=answer_to_question_similarity_threshold, 
					source_key=lambda a: a['abstract'], 
					target_key=lambda q: q,
				)
				for answer_list in question_answer_values
			)
		# question_answer_items = question_answer_dict.items()
		question_answer_items = zip(query_template_list, question_answer_values)
		question_answer_items = filter(lambda x: x[-1], question_answer_items) # remove unanswered questions
		# re_exp = re.compile(f' *{re.escape(concept_label)}')
		# question_answer_items = map(lambda x: (re.sub(re_exp,'',x[0]), x[1]), question_answer_items)
		if sort_archetypes_by_relevance:
			question_answer_items = sorted(question_answer_items, key=lambda x: x[-1][0]['confidence'], reverse=True)
		question_answer_dict = dict(question_answer_items)
		if minimise:
			question_answer_dict = self.minimise_question_answer_dict(question_answer_dict)
		return question_answer_dict

	def annotate_taxonomical_view(self, taxonomical_view, similarity_threshold=0.8, max_concepts_per_alignment=1):
		if not taxonomical_view:
			return []
		sentence_iter = map(lambda y: y[-1], filter(lambda x: not is_url(x[-1]), taxonomical_view))
		return self.concept_classifier.annotate(
			DocParser().set_content_list(list(sentence_iter)), 
			similarity_threshold=similarity_threshold, 
			max_concepts_per_alignment=max_concepts_per_alignment,
			concept_id_filter=lambda x: x in self.overview_aspect_set,
		)
		
	def get_taxonomical_view(self, concept_uri, depth=None):
		concept_set = set((concept_uri,))
		if depth != 0:
			sub_concept_set = self.adjacency_matrix.get_predicate_chain(
				concept_set = concept_set, 
				predicate_filter_fn = lambda x: x == SUBCLASSOF_PREDICATE, 
				direction_set = ['in'],
				depth = depth,
			)
			super_concept_set = self.adjacency_matrix.get_predicate_chain(
				concept_set = concept_set, 
				predicate_filter_fn = lambda x: x == SUBCLASSOF_PREDICATE, 
				direction_set = ['out'],
				depth = depth,
			)
			concept_set |= sub_concept_set
			concept_set |= super_concept_set
		# Add subclassof relations
		taxonomical_view = set(
			(s,p,o)
			for s in concept_set
			for p,o in self.adjacency_matrix_no_equivalence.get_outcoming_edges_matrix(s)
			if p == SUBCLASSOF_PREDICATE
		).union(
			(s,p,o)
			for o in concept_set
			for p,s in self.adjacency_matrix_no_equivalence.get_incoming_edges_matrix(o)
			if p == SUBCLASSOF_PREDICATE
		)
		taxonomical_view = list(taxonomical_view)
		taxonomy_concept_set = get_concept_set(taxonomical_view).union(concept_set)
		# Add labels
		taxonomical_view += (
			(concept, HAS_LABEL_PREDICATE, self.get_label(concept, explode_if_none=False))
			for concept in taxonomy_concept_set
		)
		# for concept in taxonomy_concept_set:
		# 	if not concept.startswith(WORDNET_PREFIX):
		# 		print(concept, self.label_dict[concept])
		# Add sources
		taxonomical_view += (
			(concept, HAS_SOURCE_PREDICATE, source)
			for concept in taxonomy_concept_set
			for source in self.source_dict.get(concept,())
		)
		for concept in taxonomy_concept_set:
			for source in self.source_dict.get(concept,()):
				taxonomical_view += self.get_sub_graph(source)
		# Add wordnet definitions
		taxonomical_view += (
			(concept, HAS_DEFINITION_PREDICATE, wn.synset(concept[3:]).definition())
			for concept in filter(lambda x: x.startswith(WORDNET_PREFIX), taxonomy_concept_set)
		)
		# Add definitions
		taxonomical_view += unique_everseen(
			(concept_uri,p,o)
			for p,o in self.adjacency_matrix_no_equivalence.get_outcoming_edges_matrix(concept_uri)
			if p == HAS_DEFINITION_PREDICATE
		)
		# Add types
		sub_types_set = self.adjacency_matrix.get_predicate_chain(
			concept_set = concept_set, 
			predicate_filter_fn = lambda x: x == HAS_TYPE_PREDICATE, 
			direction_set = ['out'],
			depth = 0,
		)
		super_types_set = self.adjacency_matrix.get_predicate_chain(
			concept_set = concept_set, 
			predicate_filter_fn = lambda x: x == HAS_TYPE_PREDICATE, 
			direction_set = ['in'],
			depth = 0,
		)
		taxonomical_view += (
			(concept_uri,HAS_TYPE_PREDICATE,o)
			for o in sub_types_set - concept_set
		)
		taxonomical_view += (
			(s,HAS_TYPE_PREDICATE,concept_uri)
			for s in super_types_set - concept_set
		)
		taxonomical_view += unique_everseen(
			(s, HAS_LABEL_PREDICATE, self.get_label(s, explode_if_none=False))
			for s in (super_types_set | sub_types_set) - concept_set
		)
		taxonomical_view = filter(lambda x: x[0] and x[1] and x[2], taxonomical_view)
		# print(taxonomical_view)
		return list(taxonomical_view)
	