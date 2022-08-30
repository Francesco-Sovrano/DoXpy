import json
from more_itertools import unique_everseen
import itertools
import logging

from doxpy.misc.doc_reader import DocParser

from doxpy.misc.jsonld_lib import *
from doxpy.models.reasoning.question_answerer_base import *
from doxpy.models.reasoning import is_not_wh_word
from doxpy.models.knowledge_extraction.couple_extractor import filter_invalid_sentences

class QuestionAnswerer(QuestionAnswererBase):

	def _init_sentence_classifier(self):
		pass
	
	def find_answers_in_concept_graph(self, query_list, concept_uri, question_answer_dict, answer_pertinence_threshold=0.55, add_external_definitions=False, include_super_concepts_graph=False, include_sub_concepts_graph=False, consider_incoming_relations=False, tfidf_importance=None, answer_to_question_max_similarity_threshold=0.97, answer_to_answer_max_similarity_threshold=0.97):
		def get_formatted_answer(answer):
			triple, (_,source_uri), abstract = answer['id']
			sentence = answer['context']
			# abstract = answer['doc']
			return {
				'abstract': abstract,
				'confidence': answer['similarity'],
				'syntactic_similarity': answer['syntactic_similarity'],
				'semantic_similarity': answer['semantic_similarity'],
				'extra_info': self.kg_manager.get_sub_graph(source_uri) if source_uri else None,
				'sentence': sentence, 
				'triple': triple, 
				'source_id': source_uri if source_uri else sentence, 
			}

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
		# Extract sourced triples
		self.logger.info(f"Getting get_sourced_graph_from_aspect_graph for {concept_uri}")
		sourced_natural_language_triples_set = self.kg_manager.get_sourced_graph_from_aspect_graph(concept_graph)
		if len(sourced_natural_language_triples_set) <= 0:
			self.logger.warning(f'Missing: {concept_uri}')
			return
		# sourced_natural_language_triples_set.sort(key=str) # only for better summary caching
		# Setup Sentence Classifier
		abstract_iter, context_iter, original_triple_iter, source_id_iter = zip(*sourced_natural_language_triples_set)
		id_doc_iter = tuple(zip(
			zip(original_triple_iter, source_id_iter, abstract_iter), # id
			map(lambda x: x.split('?')[-1] if '?' in x else x, abstract_iter) # doc
		))
		self.sentence_classifier.set_documents(id_doc_iter, tuple(context_iter))
		# classify
		classification_dict_gen = self.sentence_classifier.classify(
			query_list=query_list, 
			similarity_type='weighted', 
			similarity_threshold=answer_pertinence_threshold, 
			without_context=True, 
			tfidf_importance=tfidf_importance
		)
		# Add missing questions to question_answer_dict
		for question in query_list:
			if question not in question_answer_dict:
				question_answer_dict[question] = []
		# Format Answers
		for i,(question, answer_iter) in enumerate(zip(query_list, classification_dict_gen)):
			answer_iter = filter(lambda x: x['doc'], answer_iter)
			answer_list = sorted(answer_iter, key=lambda x: x['similarity'], reverse=True)
			answer_list = tuple(unique_everseen(answer_list, key=lambda x: (x['doc'],x['id'][1])))
			if len(answer_list) == 0:
				continue
			# answers contained in the question are not valid
			if answer_to_question_max_similarity_threshold:
				# print(list(map(lambda a: a['doc'], answer_list)))
				answer_list = self.sentence_classifier.filter_by_similarity_to_target(
					answer_list, 
					[question], 
					threshold=answer_to_question_max_similarity_threshold, 
					source_key=lambda a: a['doc'], 
					target_key=lambda q: q
				)
				if len(answer_list) == 0:
					continue
			# ignore similar-enough sentences with lower pertinence
			if answer_to_answer_max_similarity_threshold:
				answer_list = self.sentence_classifier.remove_similar_labels(
					answer_list, 
					threshold=answer_to_answer_max_similarity_threshold, 
					key=lambda x: x['context'],
					without_context=True,
				)
			# filter invalid sentences not being useful for any overview
			valid_answers = map(get_formatted_answer, answer_list)
			valid_answers = unique_everseen(valid_answers, key=lambda x: x['sentence'])
			question_answer_dict[question] += valid_answers
		return question_answer_dict

################################################################################################################################################

	def ask(self, question_list, query_concept_similarity_threshold=0.55, answer_pertinence_threshold=0.55, with_numbers=True, remove_stopwords=False, lemmatized=False, keep_the_n_most_similar_concepts=1, add_external_definitions=False, include_super_concepts_graph=True, include_sub_concepts_graph=True, consider_incoming_relations=True, tfidf_importance=None, concept_label_filter=is_not_wh_word, answer_to_question_max_similarity_threshold=0.97, answer_to_answer_max_similarity_threshold=0.97, **args):
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
		self.logger.debug(len(concepts_dict))
		# Group queries by concept_uri
		concept_uri_query_dict = {}
		# print(json.dumps(concepts_dict, indent=4))
		for concept_label, concept_count_dict in concepts_dict.items():
			for concept_similarity_dict in itertools.islice(unique_everseen(concept_count_dict["similar_to"], key=lambda x: x["id"]), min(1,keep_the_n_most_similar_concepts)):
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
				answer_to_question_max_similarity_threshold= answer_to_question_max_similarity_threshold,
				answer_to_answer_max_similarity_threshold = answer_to_answer_max_similarity_threshold,
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

	def get_concept_overview_fast(self, **args):
		return super().get_concept_overview(**args)

	def get_concept_overview(self, query_template_list=None, concept_uri=None, concept_label=None, answer_pertinence_threshold=0.3, add_external_definitions=True, include_super_concepts_graph=True, include_sub_concepts_graph=True, consider_incoming_relations=True, tfidf_importance=None, sort_archetypes_by_relevance=True, answer_to_question_max_similarity_threshold=0.97, answer_to_answer_max_similarity_threshold=0.97, minimise=True, **args):
		assert concept_uri, f"{concept_uri} is not a valid concept_uri"
		if query_template_list is None:
			query_template_list = list(QuestionAnswererBase.archetypal_questions_dict.values())
		elif not query_template_list:
			return {}
		# set consider_incoming_relations to False with concept-centred generic questions (e.g. what is it?), otherwise the answers won't be the sought ones
		if not concept_label:
			concept_label = self.kg_manager.get_label(concept_uri)
		question_answer_dict = {}
		self.logger.info(f'get_concept_overview {concept_uri}: finding answers in concept graph..')
		self.find_answers_in_concept_graph(
			query_list= tuple(map(lambda x:x.replace('{X}',concept_label), query_template_list)), 
			concept_uri= concept_uri, 
			question_answer_dict= question_answer_dict, 
			answer_pertinence_threshold= answer_pertinence_threshold,
			add_external_definitions= add_external_definitions,
			include_super_concepts_graph= include_super_concepts_graph, 
			include_sub_concepts_graph= include_sub_concepts_graph, 
			consider_incoming_relations= consider_incoming_relations,
			tfidf_importance= tfidf_importance,
			answer_to_question_max_similarity_threshold= answer_to_question_max_similarity_threshold,
			answer_to_answer_max_similarity_threshold = answer_to_answer_max_similarity_threshold,
		)
		question_answer_values = question_answer_dict.values()
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

