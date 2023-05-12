import json
from more_itertools import unique_everseen
import itertools
import logging

from doxpy.misc.doc_reader import DocParser

from doxpy.misc.jsonld_lib import *
from doxpy.models.reasoning.answer_retriever_base import *
from doxpy.models.reasoning import is_not_wh_word
from doxpy.models.knowledge_extraction.couple_extractor import filter_invalid_sentences

class AnswerRetriever(AnswerRetrieverBase):
	def __init__(self, kg_manager, concept_classifier_options, sentence_classifier_options, default_query_template_list=None, **args):
		super().__init__(kg_manager, concept_classifier_options, sentence_classifier_options, **args)
		self.default_query_template_list = default_query_template_list

	def _init_sentence_classifier(self):
		pass

	def get_formatted_answer(self, answer):
		triple, (source_sentence_uri,source_uri), abstract = answer['id']
		sentence = answer['context']
		# paragraph = None
		# if source_sentence_uri:
		# 	paragraph_iter = map(self.kg_manager.get_paragraph_text, self.kg_manager.source_dict[source_sentence_uri])
		# 	paragraph_iter = filter(lambda x: x, paragraph_iter)
		# 	paragraph_iter = filter(lambda x: len(x) <= self.kg_manager.max_paragraph_len, paragraph_iter)
		# 	paragraph_iter = list(paragraph_iter)
		# 	paragraph = max(paragraph_iter, key=len) if paragraph_iter else None

		# abstract = answer['doc']
		return {
			'abstract': abstract,
			'confidence': answer['similarity'],
			'syntactic_similarity': answer['syntactic_similarity'],
			'semantic_similarity': answer['semantic_similarity'],
			# 'extra_info': self.kg_manager.get_sub_graph(source_uri) if source_uri else None,
			'sentence': sentence, # paragraph if paragraph else sentence, 
			'triple': triple, 
			'source_id': source_uri if source_uri else sentence, 
			'source_sentence_uri': source_sentence_uri,
		}

	def find_answers_in_concept_graph(self, query_list, concept_uri, question_answer_dict, answer_pertinence_threshold=0.55, add_external_definitions=False, include_super_concepts_graph=False, include_sub_concepts_graph=False, consider_incoming_relations=False, tfidf_importance=None, answer_to_question_max_similarity_threshold=0.97, answer_to_answer_max_similarity_threshold=0.97, use_weak_pointers=False, top_k=None):
		self.logger.info(f"Getting aspect_graph for {concept_uri}")
		concept_graph = self.kg_manager.get_aspect_graph(
			concept_uri=concept_uri, 
			add_external_definitions=add_external_definitions, 
			include_super_concepts_graph=include_super_concepts_graph, 
			include_sub_concepts_graph=include_sub_concepts_graph, 
			consider_incoming_relations=consider_incoming_relations,
			filter_fn=lambda x: '{obj}' in x[1],
		)
		self.logger.debug('######## Concept Graph ########')
		self.logger.debug(f"{concept_uri} has {len(concept_graph)} triplets")
		self.logger.debug(json.dumps(concept_graph, indent=4))
		# Extract sourced triples
		self.logger.info(f"Getting get_sourced_graph_from_aspect_graph for {concept_uri}")
		sourced_natural_language_triples_set = self.kg_manager.get_sourced_graph_from_aspect_graph(concept_graph)
		if len(sourced_natural_language_triples_set) <= 0:
			self.logger.warning(f'Missing: {concept_uri}')
			# Add missing questions to question_answer_dict
			for question in query_list:
				if question not in question_answer_dict:
					question_answer_dict[question] = []
			return question_answer_dict
		# sourced_natural_language_triples_set.sort(key=str) # only for better summary caching
		# Setup Sentence Classifier
		abstract_iter, context_iter, original_triple_iter, source_id_iter = zip(*sourced_natural_language_triples_set)
		id_doc_iter = tuple(zip(
			zip(original_triple_iter, source_id_iter, abstract_iter), # id
			map(lambda x: x.split('?')[-1] if '?' in x else x, abstract_iter) if use_weak_pointers else abstract_iter # doc
		))
		self.sentence_classifier.set_documents(id_doc_iter, tuple(context_iter))
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
		for i,(question, answer_iter) in enumerate(zip(query_list, classification_dict_gen)):
			answer_iter = filter(lambda x: x['doc'], answer_iter)
			question_answer_dict[question] += map(self.get_formatted_answer, answer_iter)
		return question_answer_dict

	def sort_question_answer_dict(self, question_answer_dict, answer_to_question_max_similarity_threshold=None, answer_to_answer_max_similarity_threshold=None):
		for question, answer_list in question_answer_dict.items():
			answer_list = sorted(answer_list, key=lambda x: x['confidence'], reverse=True)
			answer_list = list(unique_everseen(answer_list, key=lambda x: (x['source_id'],x['abstract']))) # Filtering by source_id and triplet (a.k.a. level of detail) is needed by the DoXEstimator and it does not filter out good answers because it only keeps either the sentence or the paragraph as context. In fact, KGManager::get_sourced_graph_from_aspect_graph outputs triplets with both sentence and paragraph as context (if these are different).
			if answer_list and answer_to_question_max_similarity_threshold: # Answers contained in the question are not valid
				answer_list = self.sentence_classifier.filter_by_similarity_to_target(
					answer_list, 
					[question], 
					threshold=answer_to_question_max_similarity_threshold, 
					source_key=lambda a: a['sentence'], 
					target_key=lambda q: q
				)
			if answer_list and answer_to_answer_max_similarity_threshold: # Ignore similar-enough sentences with lower pertinence
				answer_list = self.sentence_classifier.remove_similar_labels(
					answer_list, 
					threshold=answer_to_answer_max_similarity_threshold, 
					key=lambda x: x['sentence'],
					without_context=True,
				)
			question_answer_dict[question] = answer_list
		return question_answer_dict

################################################################################################################################################

	def ask(self, question_list, query_concept_similarity_threshold=0.55, answer_pertinence_threshold=0.55, with_numbers=True, remove_stopwords=False, lemmatized=False, keep_the_n_most_similar_concepts=None, add_external_definitions=False, include_super_concepts_graph=True, include_sub_concepts_graph=True, consider_incoming_relations=True, tfidf_importance=None, concept_label_filter=is_not_wh_word, answer_to_question_max_similarity_threshold=0.97, answer_to_answer_max_similarity_threshold=0.97, use_weak_pointers=False, filter_fn=None, top_k=None, answer_horizon=None, minimise=False, **args):
		# set consider_incoming_relations to False with concept-centred generic questions (e.g. what is it?), otherwise the answers won't be the sought ones
		self.logger.info(f'Extracting concepts from question_list: {json.dumps(question_list, indent=4)}..')
		concepts_dict = self.concept_classifier.get_concept_dict(
			doc_parser=DocParser().set_content_list(question_list),
			similarity_threshold=query_concept_similarity_threshold, 
			with_numbers=with_numbers, 
			remove_stopwords=remove_stopwords, 
			lemmatized=lemmatized,
			concept_label_filter=concept_label_filter,
			size=keep_the_n_most_similar_concepts,
		)
		self.logger.debug('######## Concepts Dict ########')
		self.logger.debug(json.dumps(concepts_dict, indent=4))
		# Group queries by concept_uri
		concept_uri_query_dict = {}
		# print(json.dumps(concepts_dict, indent=4))
		for concept_label, concept_count_dict in concepts_dict.items():
			for concept_similarity_dict in itertools.islice(unique_everseen(concept_count_dict["similar_to"], key=lambda x: x["id"]), max(1,keep_the_n_most_similar_concepts)):
				concept_uri = concept_similarity_dict["id"]
				concept_query_set = concept_uri_query_dict.get(concept_uri,None)
				if concept_query_set is None:
					concept_query_set = concept_uri_query_dict[concept_uri] = set()
				concept_query_set.update((
					sent_dict["paragraph_text"]
					for sent_dict in concept_count_dict["source_list"]
				))
		# For every aligned concept, extract from the ontology all the incoming and outgoing triples, thus building a partial graph (a view).
		question_answer_dict = {}
		for concept_uri, concept_query_set in concept_uri_query_dict.items():
			self.logger.info(f'Extracting answers related to {concept_uri}..')
			self.find_answers_in_concept_graph(
				query_list= list(concept_query_set), 
				concept_uri= concept_uri, 
				question_answer_dict= question_answer_dict, 
				answer_pertinence_threshold= answer_pertinence_threshold,
				add_external_definitions= add_external_definitions,
				include_super_concepts_graph= include_super_concepts_graph, 
				include_sub_concepts_graph= include_sub_concepts_graph, 
				consider_incoming_relations= consider_incoming_relations,
				tfidf_importance= tfidf_importance,
				use_weak_pointers= use_weak_pointers,
				top_k= top_k,
			)
		####################################
		## Sort and filter duplicated answers
		self.logger.info(f'Sorting answers..')
		question_answer_dict = self.sort_question_answer_dict(question_answer_dict, answer_to_question_max_similarity_threshold=answer_to_question_max_similarity_threshold, answer_to_answer_max_similarity_threshold=answer_to_answer_max_similarity_threshold)
		####################################
		question_answer_items = zip(question_list, question_answer_dict.values())
		if filter_fn: # remove unwanted answers
			question_answer_items = map(lambda x: (x[0], list(filter(filter_fn,x[-1]))), question_answer_items) # remove unwanted answers
		if answer_horizon: # keep only answer_horizon answers
			question_answer_items = [
				(q,a_list[:answer_horizon])
				for q,a_list in question_answer_items
			]
		question_answer_dict = dict(question_answer_items)
		if minimise:
			question_answer_dict = self.merge_duplicated_answers(question_answer_dict)
		return question_answer_dict

	def get_default_template_list(self, concept_uri, **args):
		if self.default_query_template_list:
			return self.default_query_template_list
		# set consider_incoming_relations to False with concept-centred generic questions (e.g. what is it?), otherwise the answers won't be the sought ones
		return [
			'What is {X}?',
			'Why {X}?',
			'How is {X}?',
			'Where is {X}?',
			# 'When is {X}?',
			# # 'Who is {X}?',
			# 'Which {X}?',
			# # 'Whose {X}?',
		]

	def get_concept_overview(self, query_template_list=None, concept_uri=None, concept_label=None, answer_pertinence_threshold=0.3, add_external_definitions=True, include_super_concepts_graph=True, include_sub_concepts_graph=True, consider_incoming_relations=True, tfidf_importance=None, sort_archetypes_by_relevance=True, answer_to_question_max_similarity_threshold=0.97, answer_to_answer_max_similarity_threshold=0.97, minimise=True, use_weak_pointers=False, question_horizon=None, filter_fn=None, answer_horizon=None, top_k=None, question_generator=None, keep_the_n_most_similar_concepts=None, query_concept_similarity_threshold=None, **args):
		assert concept_uri, f"{concept_uri} is not a valid concept_uri"
		if query_template_list is None:
			query_template_list = self.get_default_template_list(concept_uri)
		elif not query_template_list:
			return {}
		if question_generator is None:
			question_generator = lambda x,l: x.replace('{X}',l)
		# set consider_incoming_relations to False with concept-centred generic questions (e.g. what is it?), otherwise the answers won't be the sought ones
		if not concept_label:
			concept_label = self.kg_manager.get_label(concept_uri)
		concept_uri_set = set([concept_uri])
		if keep_the_n_most_similar_concepts!=0 and (query_concept_similarity_threshold or (query_concept_similarity_threshold is None and self.concept_classifier.default_similarity_threshold)):
			self.logger.info(f'Extracting concepts from concept_label_list: {concept_uri}..')
			concepts_iter = flatten(self.concept_classifier.classify(
				query_list=self.kg_manager.get_label_list(concept_uri), 
				similarity_type='weighted', 
				similarity_threshold=query_concept_similarity_threshold, 
				without_context=True, 
			))

			super_n_sub_classes = concept_uri_set | self.kg_manager.get_sub_classes(concept_uri_set) | self.kg_manager.get_super_classes(concept_uri_set)
			concepts_iter = filter(lambda x: x["id"] not in super_n_sub_classes, concepts_iter)
			
			if keep_the_n_most_similar_concepts:
				concepts_iter = itertools.islice(concepts_iter, max(1,keep_the_n_most_similar_concepts))
			concepts_list = list(concepts_iter)
			self.logger.info('######## Concepts Dict ########')
			self.logger.info(json.dumps(concepts_list, indent=4))
			concept_uri_set |= set((
				concept_similarity_dict["id"]
				for concept_similarity_dict in concepts_list
			))	
						
		# For every aligned concept, extract from the ontology all the incoming and outgoing triples, thus building a partial graph (a view).
		question_answer_dict = {}
		for concept_uri in concept_uri_set:
			self.logger.info(f'get_concept_overview for "{concept_label}": finding answers in concept graph of <{concept_uri}>..')
			self.find_answers_in_concept_graph(
				query_list= tuple(map(lambda x: question_generator(x,concept_label), query_template_list)), 
				concept_uri= concept_uri, 
				question_answer_dict= question_answer_dict, 
				answer_pertinence_threshold= answer_pertinence_threshold,
				add_external_definitions= add_external_definitions,
				include_super_concepts_graph= include_super_concepts_graph, 
				include_sub_concepts_graph= include_sub_concepts_graph, 
				consider_incoming_relations= consider_incoming_relations,
				tfidf_importance= tfidf_importance,
				use_weak_pointers= use_weak_pointers,
				top_k= top_k,
			)
		question_answer_items = zip(query_template_list, question_answer_dict.values())
		if filter_fn:
			question_answer_items = map(lambda x: (x[0], list(filter(filter_fn,x[-1]))), question_answer_items) # remove unwanted answers
		####################################
		## Sort and filter duplicated answers
		self.logger.info(f'Sorting answers..')
		question_answer_items = self.sort_question_answer_dict(dict(question_answer_items), answer_to_question_max_similarity_threshold=answer_to_question_max_similarity_threshold, answer_to_answer_max_similarity_threshold=answer_to_answer_max_similarity_threshold).items()
		####################################
		# # The answers contained in other answers are removed.
		# question_answer_items = map(lambda x: (x[0], list(unique_everseen(x[-1], key=lambda y: y["source_id"]))), question_answer_items) # remove unwanted answers

		# if question_to_question_max_similarity_threshold: # ignore too similar questions with lower pertinence
		# question_answer_items = sorted(question_answer_items, key=lambda x: sum(map(lambda y: y['confidence'], x[-1])), reverse=True)
		# question_answer_items = self.sentence_classifier.remove_similar_labels(
		# 	question_answer_items,
		# 	threshold=0.85, 
		# 	key=lambda x: x[0],
		# 	without_context=False,
		# )

		if minimise:
			# The answers contained in other answers (even if from other questions) are replaced with the containing answer.
			# The answers that could be assigned to several questions are given to the question with the highest estimated pertinence.
			question_answer_items = list(
				self.minimise_question_answer_dict(
					self.merge_duplicated_answers(
						dict(question_answer_items)
					)
				).items()
			)

		# The questions with no answers are removed.
		# question_answer_items = list(filter(lambda x: x[-1], question_answer_items)) # remove unanswered questions

		if sort_archetypes_by_relevance:
			# # The questions are sorted by decreasing cumulative pertinence of their answers.
			# question_answer_dict = dict(sorted(question_answer_dict.items(), key=lambda x: sum(map(lambda y: y['confidence'], x[-1])), reverse=True))
			# The questions are sorted by decreasing relevance of the first answer.
			question_answer_items = sorted(question_answer_items, key=lambda x: x[-1][0]['confidence'] if x[-1] else float('-inf'), reverse=True)

		if question_horizon:
			# Only $q$ questions are kept.
			question_answer_items = question_answer_items[:question_horizon]

		if answer_horizon:
			# Only the best $a$ answers are kept.
			question_answer_items = [
				(q,a_list[:answer_horizon])
				for q,a_list in question_answer_items
			]

		return dict(question_answer_items)

