from doxpy.misc.doc_reader import DocParser

from doxpy.models.classification.concept_classifier import ConceptClassifier
from doxpy.models.classification.sentence_classifier import SentenceClassifier
from doxpy.models.knowledge_extraction.couple_extractor import filter_invalid_sentences

# from doxpy.misc.graph_builder import get_concept_description_dict
from doxpy.misc.levenshtein_lib import remove_similar_labels
from doxpy.misc.jsonld_lib import *
from doxpy.misc.utils import *
from doxpy.models.reasoning import is_not_wh_word

import numpy as np
from collections import Counter
import re
import time
import json
from more_itertools import unique_everseen
import itertools
import wikipedia
from collections import namedtuple
import logging
from doxpy.models.model_manager import ModelManager

ArchetypePertinence = namedtuple('ArchetypePertinence',['archetype','pertinence'])
InformationUnit = namedtuple('InformationUnit',['unit','context'])
get_information_unit = lambda x: InformationUnit(x['abstract'], x['sentence'])

class QuestionAnswererBase(ModelManager):
	archetypal_questions_dict = {
		##### Descriptive
		# 'what': 'What is a description of {X}?',
		'what': 'What is {X}?',
		# 'what': 'What is {X}?',
		'who': 'Who is {X}?',
		# 'whom': 'Whom {X}?',
		##### Causal + Justificatory
		'why': 'Why is {X}?',
		# 'why-not': 'Why not {X}?',
		##### Counterfactual
		# 'what-if': 'What if {X}?',
		##### Teleological
		# 'what-for': 'What is {X} for?',
		# 'what-for': 'What is {X} for?',
		##### Expository
		'how': 'How is {X}?',
		##### Quantitative
		# 'how-much': 'How much {X}?',
		# 'how-many': 'How many {X}?',
		##### Spatial
		'where': 'Where is {X}?',
		##### Temporal
		'when': 'When is {X}?',
		##### Medium
		# 'who-by': 'Who by {X}?',
		##### Extra
		'which': 'Which is {X}?',
		'whose': 'Whose is {X}?',
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

	def __init__(self, kg_manager, concept_classifier_options, sentence_classifier_options, betweenness_centrality=None, **args):
		super().__init__(sentence_classifier_options)
		self.disable_spacy_component = ["ner","textcat"]
		
		self.betweenness_centrality = betweenness_centrality
		self.kg_manager = kg_manager

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
			self._overview_aspect_set = set(filter(lambda x: self.kg_manager.is_relevant_aspect(x,ignore_leaves=True), self.concept_classifier.ids))
			# Betweenness centrality quantifies the number of times a node acts as a bridge along the shortest path between two other nodes.
			if self.betweenness_centrality is not None:
				filtered_betweenness_centrality = dict(filter(lambda x: x[-1] > 0, self.betweenness_centrality.items()))
				self._overview_aspect_set &= filtered_betweenness_centrality.keys()
		return self._overview_aspect_set

	@property
	def relevant_aspect_set(self):
		if self._relevant_aspect_set is None:
			self._relevant_aspect_set = set(filter(self.kg_manager.is_relevant_aspect, self.concept_classifier.ids))
		return self._relevant_aspect_set

	@property
	def adjacency_list(self):
		return self.kg_manager.adjacency_list

	def _init_sentence_classifier(self):
		self.logger.info('Initialising Sentence Classifier..')
		# Setup Sentence Classifier
		abstract_iter, context_iter, original_triple_iter, source_id_iter = zip(*filter(lambda x: x[0].strip() and x[1].strip(), self.kg_manager.get_sourced_graph()))
		id_doc_iter = tuple(zip(
			zip(original_triple_iter, source_id_iter), # id
			abstract_iter # doc
		))
		self._sentence_classifier.set_documents(id_doc_iter, tuple(context_iter))

	def _init_concept_classifier(self):
		self.logger.info('Initialising Concept Classifier..')
		self._concept_classifier.set_concept_description_dict(self.kg_manager.concept_description_dict)
	
	def store_cache(self, cache_name):
		self._concept_classifier.store_cache(cache_name+'.concept_classifier.pkl')
		self._sentence_classifier.store_cache(cache_name+'.sentence_classifier.pkl')

	def load_cache(self, cache_name, save_if_init=True, **args):
		if self._sentence_classifier is None:
			self._sentence_classifier = SentenceClassifier(self.sentence_classifier_options)
			loaded_sentence_classifier = self._sentence_classifier.load_cache(cache_name+'.sentence_classifier.pkl')
			self._init_sentence_classifier()
			if not loaded_sentence_classifier and save_if_init:
				self._sentence_classifier.store_cache(cache_name+'.sentence_classifier.pkl')
		#######
		if self._concept_classifier is None:
			self._concept_classifier = ConceptClassifier(self.concept_classifier_options)
			loaded_concept_classifier = self._concept_classifier.load_cache(cache_name+'.concept_classifier.pkl')
			self._init_concept_classifier()
			if not loaded_concept_classifier and save_if_init:
				self._concept_classifier.store_cache(cache_name+'.concept_classifier.pkl')

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
				'syntactic_similarity': answers[0]['syntactic_similarity'] if answers else 0,
				'semantic_similarity': answers[0]['semantic_similarity'] if answers else 0,
			}
			for question,answers in question_answer_dict.items()
		}

	def find_answers(self, query_list, question_answer_dict=None, answer_pertinence_threshold=0.55, tfidf_importance=None, answer_to_question_max_similarity_threshold=0.9502, answer_to_answer_max_similarity_threshold=0.9502, top_k=None):
		if question_answer_dict is None: question_answer_dict = {}
		def get_formatted_answer(answer):
			triple, source_uri = answer['id']
			sentence = answer['context']
			extra_info = self.kg_manager.get_sub_graph(source_uri) if source_uri else None
			return {
				'abstract': answer['doc'],
				'confidence': answer['similarity'],
				'syntactic_similarity': answer['syntactic_similarity'],
				'semantic_similarity': answer['semantic_similarity'],
				'extra_info': extra_info,
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
			if answer_to_question_max_similarity_threshold is not None:
				answer_list = self.sentence_classifier.filter_by_similarity_to_target(
					answer_list, 
					[question], 
					threshold=answer_to_question_max_similarity_threshold, 
					source_key=lambda a: a['doc'].split('?')[-1] if '?' in a['doc'] else a['doc'], 
					target_key=lambda q: q
				)
			# ignore similar-enough sentences with lower pertinence
			if answer_to_answer_max_similarity_threshold is not None:
				answer_list = self.sentence_classifier.remove_similar_labels(
					answer_list, 
					threshold=answer_to_answer_max_similarity_threshold, 
					key=lambda x: (x['doc'],x['context']),
					# without_context=False,
				)
			valid_answers = map(get_formatted_answer, answer_list)
			valid_answers = unique_everseen(valid_answers, key=lambda x: x['sentence']) # apparently not covered by "remove_similar_labels"
			question_answer_dict[question] += valid_answers
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
		QuestionAnswererBase.logger.info('Minimising question answer dict')
		answer_question_dict = QuestionAnswererBase.get_answer_question_pertinence_dict(question_answer_dict, update_answers=True)
		get_best_answer_archetype = lambda a: max(answer_question_dict[get_information_unit(a)], key=lambda y: y.pertinence).archetype
		return {
			question: list(filter(lambda x: get_best_answer_archetype(x)==question, answers))
			for question,answers in question_answer_dict.items()
		}

	@staticmethod
	def get_question_answer_overlap_dict(question_answer_dict):
		answer_question_dict = QuestionAnswererBase.get_answer_question_pertinence_dict(question_answer_dict)
		get_question_iter = lambda q,a_list: filter(lambda x: x!=q, (answer_question_pertinence_dict[get_information_unit(a)].archetype for a in a_list))
		return {
			question: Counter(get_question_iter(question,answers))
			for question,answers in question_answer_dict.items()
		}

	def get_answer_relatedness_to_question(self, question_list, answer_list): 
		question_list = list(map(lambda x: x if question.endswith('?') else x+'?', question_list))
		return self.sentence_classifier.get_element_wise_similarity(question_list,answer_list, source_without_context=True, target_without_context=False)

################################################################################################################################################

	def ask(self, question_list, answer_pertinence_threshold=None, tfidf_importance=None, answer_to_question_max_similarity_threshold=0.9502, answer_to_answer_max_similarity_threshold=0.9502, top_k=None, **args):
		question_answer_dict = self.find_answers(
			query_list= question_list, 
			answer_pertinence_threshold= answer_pertinence_threshold,
			tfidf_importance= tfidf_importance,
			answer_to_question_max_similarity_threshold= answer_to_question_max_similarity_threshold,
			answer_to_answer_max_similarity_threshold= answer_to_answer_max_similarity_threshold,
			top_k= top_k,
		)
		# Sort answers
		self.logger.info(f'Sorting answers..')
		for question, formatted_answer_list in question_answer_dict.items():
			question_answer_dict[question] = list(unique_everseen(
				sorted(
					formatted_answer_list, 
					key=lambda x: x['confidence'], reverse=True
				), 
				key=lambda x: x["source_id"]
			))
		return question_answer_dict

	def get_concept_overview(self, query_template_list=None, concept_uri=None, concept_label=None, answer_pertinence_threshold=None, tfidf_importance=None, sort_archetypes_by_relevance=True, answer_to_question_max_similarity_threshold=0.9502, answer_to_answer_max_similarity_threshold=0.9502, minimise=True, top_k=None, **args):
		assert concept_uri, f"{concept_uri} is not a valid concept_uri"
		if query_template_list is None:
			query_template_list = list(QuestionAnswererBase.archetypal_questions_dict.values())
		# set consider_incoming_relations to False with concept-centred generic questions (e.g. what is it?), otherwise the answers won't be the sought ones
		if not concept_label:
			concept_label = self.kg_manager.get_label(concept_uri)
		question_answer_dict = self.find_answers(
			query_list= tuple(map(lambda x:x.replace('{X}',concept_label), query_template_list)), 
			answer_pertinence_threshold= answer_pertinence_threshold,
			tfidf_importance= tfidf_importance,
			answer_to_question_max_similarity_threshold= None,
			answer_to_answer_max_similarity_threshold= answer_to_answer_max_similarity_threshold,
			top_k= top_k,
		)
		question_answer_values = question_answer_dict.values()
		# answers contained in the question are not valid
		if answer_to_question_max_similarity_threshold is not None:
			question_answer_values = (
				self.sentence_classifier.filter_by_similarity_to_target(
					answer_list, 
					[concept_label], 
					threshold=answer_to_question_max_similarity_threshold, 
					source_key=lambda a: a['abstract'], 
					target_key=lambda q: q,
				) if answer_list else answer_list
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

	def annotate_taxonomical_view(self, taxonomical_view, similarity_threshold=0.8, max_concepts_per_alignment=1, tfidf_importance=None, is_preprocessed_content=False):
		if not taxonomical_view:
			return []
		sentence_iter = map(lambda y: y[-1], filter(lambda x: not is_url(x[-1]), taxonomical_view))
		return self.concept_classifier.annotate(
			DocParser().set_content_list(list(sentence_iter)), 
			similarity_threshold=similarity_threshold, 
			max_concepts_per_alignment=max_concepts_per_alignment,
			tfidf_importance=tfidf_importance,
			concept_id_filter=lambda x: x in self.overview_aspect_set,
			is_preprocessed_content=is_preprocessed_content,
		)

	def get_taxonomical_view(self, *arg, **args):
		return self.kg_manager.get_taxonomical_view(*arg, **args, concept_id_filter=lambda x: x in self.overview_aspect_set)
	