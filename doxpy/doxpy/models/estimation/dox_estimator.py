import json
import numpy as np
from more_itertools import unique_everseen
from collections import Counter
from itertools import islice

from doxpy.misc.cache_lib import load_or_create_cache, create_cache, load_cache
from doxpy.misc.levenshtein_lib import remove_similar_labels, labels_are_contained

# from doxpy.models.reasoning.answer_retriever import AnswerRetriever
from doxpy.misc.jsonld_lib import *
from doxpy.models.reasoning import is_not_wh_word
from doxpy.misc.utils import *

get_archetype_set = lambda x: [qp.archetype for qp in x['question_pertinence_set']]

class DoXEstimator:
	
	@staticmethod
	def get_explanatory_illocution_from_archetype_fitness(fitness_dict):
		# tot_relevance describes well how all the information units are relevant together, the more details, the higher is explanatory illocution. If the minimum relevance is high (long-tail distribution), tot_relevance is more sensitive to the number of information units than to their relevance. For example, according to this metric, many low-relevance units beat a single high-relevance unit even if the best option would be a single high-relevance unit.
		# weighted_level_of_detail = fitness_dict["tot_relevance"]["median"]
		# top5_relevance is as tot_relevance, but with the long-tail of information units cut.
		# top5_weighted_level_of_detail = fitness_dict["top5_relevance"]["median"]
		# max_relevance does not take into account the number of different information units, but it's good to describe what is the maximum precision of information units for this aspect.
		# maximum_explanatory_relevance = fitness_dict["max_relevance"]["median"]
		# explanatory_illocution = weighted_level_of_detail + top1_weighted_level_of_detail, because this way higher importance is given to the best 10 information units
		return fitness_dict["tot_relevance"]["mean"] if isinstance(fitness_dict["tot_relevance"],dict) else fitness_dict["tot_relevance"]

	def __init__(self, answer_retriever, archetypal_questions_dict=None):
		self.answer_retriever = answer_retriever
		self.archetypal_questions_dict = self.answer_retriever.archetypal_questions_dict if archetypal_questions_dict is None else archetypal_questions_dict

	def get_aspect_wedox_dict(self, aspect_archetype_answers_dict, archetype_fitness_options=None, archetype_weight_dict=None):
		if archetype_fitness_options is None:
			archetype_fitness_options = {}
		aspect_archetype_fitness_dict = self.get_aspect_archetype_fitness_dict(aspect_archetype_answers_dict, archetype_fitness_options)
		aspect_dox_dict = {
			aspect_uri: self.get_degree_of_explainability_from_archetype_fitness(archetype_fitness_dict)
			for aspect_uri, archetype_fitness_dict in aspect_archetype_fitness_dict.items()
		}
		# print('Aspects DoX:', json.dumps(aspect_dox_dict, indent=4))
		aspect_wedox_dict = {
			aspect_uri: self.get_weighted_degree_of_explainability(dox, archetype_weight_dict)
			for aspect_uri, dox in aspect_dox_dict.items()
		}
		return aspect_wedox_dict

	def extract_archetypal_answers_per_aspect(self, aspect_uri_iter=None, query_template_list=None, question_generator=None, archetypal_qa_options=None):
		_archetypal_qa_options = {
			'answer_horizon': None,
			'question_horizon': None,
			######################
			## AnswerRetriever stuff
			'tfidf_importance': 0,
			'answer_pertinence_threshold': None, 
			'answer_to_question_max_similarity_threshold': None,
			'answer_to_answer_max_similarity_threshold': None,
			######################
			'include_super_concepts_graph': False, 
			'include_sub_concepts_graph': True, 
			'consider_incoming_relations': True,
			'minimise': False, 
			'use_weak_pointers': False,
			######################
			'sort_archetypes_by_relevance': False, 
			'top_k': None,
			# 'keep_the_n_most_similar_concepts': 3,
		}
		if archetypal_qa_options:
			_archetypal_qa_options.update(archetypal_qa_options)
			
		# _archetypal_qa_options['filter_fn'] = lambda a: '?' not in a['abstract'] # use only most granular information units available: clauses
		# self.answer_retriever.logger.info(f"Extracting aspect_archetype_answers_dict with top_k {_archetypal_qa_options.get('top_k',None)}..")
		self.answer_retriever.logger.info('Extract archetypal answer_list for each aspect')
		if not aspect_uri_iter:
			aspect_uri_iter = unique_everseen(self.answer_retriever.kg_manager.aspect_uri_list)
		if query_template_list is None:
			query_template_list = list(self.archetypal_questions_dict.values())
		# ##################################################################
		# ## Important: do not consider paragraphs during concept overview, they may duplicate the level of detail, causing over-estimated DoX scores
		# old_max_paragraph_len = self.answer_retriever.kg_manager.max_paragraph_len # memorise current max_paragraph_len
		# self.answer_retriever.kg_manager.max_paragraph_len = 0 # set max_paragraph_len to zero
		# ##################################################################
		archetypal_answers_per_aspect = {
			aspect_uri: self.answer_retriever.get_concept_overview(
				query_template_list=query_template_list, 
				concept_uri=aspect_uri, 
				question_generator=question_generator,
				**_archetypal_qa_options,
			)
			for aspect_uri in self.answer_retriever.tqdm(aspect_uri_iter)
		}
		# ##################################################################
		# self.answer_retriever.kg_manager.max_paragraph_len = old_max_paragraph_len # restore old max_paragraph_len
		# ##################################################################
		return archetypal_answers_per_aspect

	@staticmethod
	def get_fitness_from_answer_list(answer_list=None):
		if not answer_list:
			return { # candidate main statistics: max_relevance, tot_relevance, top10_relevance, mean_relevance
				"max_relevance": 0, # max_relevance does not take into account the number of different information units, but it's good to describe what is the maximum precision of information units for this aspect.
				"min_relevance": 0,
				"tot_relevance": 0, # tot_relevance describes well how all the information units are relevant together. If the minimum relevance is high (long-tail distribution), tot_relevance is more sensitive to the number of information units than to their relevance (i.e. many low-relevance units beat a single high-relevance unit).
				"top5_relevance": 0, # As tot_relevance, but the long-tail of information units is cut.
				# "head_relevance": 0,
				# "tail_relevance": 0,
				"mean_relevance": 0, # Differently to tot_relevance, a single high-relevance unit is weighted more than many low-relevance units, but the relevance distribution of information units has a long-tail. Therefore the more information units, the lower this value is likely to be.
				"median_relevance": 0,
				"relevant_answers": 0,
				# "unit_LoD": 0,
				# "context_LoD": 0,
			}
		answers_relevance = [a['confidence'] for a in answer_list]
		median_relevance = np.median(answers_relevance)
		tot_relevance = sum(answers_relevance)
		return { # candidate main statistics: max_relevance, tot_relevance, top10_relevance, mean_relevance
			"max_relevance": max(answers_relevance), # max_relevance does not take into account the number of different information units, but it's good to describe what is the maximum precision of information units for this aspect.
			"min_relevance": min(answers_relevance),
			"tot_relevance": tot_relevance, # tot_relevance describes well how all the information units are relevant together. If the minimum relevance is high (long-tail distribution), tot_relevance is more sensitive to the number of information units than to their relevance (i.e. many low-relevance units beat a single high-relevance unit).
			"top5_relevance": sum(answers_relevance[:5]), # As tot_relevance, but the long-tail of information units is cut.
			# "head_relevance": sum(filter(lambda x: x >= median_relevance, answers_relevance)),
			# "tail_relevance": sum(filter(lambda x: x < median_relevance, answers_relevance)),
			"mean_relevance": tot_relevance/len(answers_relevance), # Differently to tot_relevance, a single high-relevance unit is weighted more than many low-relevance units, but the relevance distribution of information units has a long-tail. Therefore the more information units, the lower this value is likely to be.
			"median_relevance": median_relevance,
			"relevant_answers": len(answers_relevance),
			# "unit_LoD": len(unit_qa_list),
			# "context_LoD": len(context_qa_list),
		}

	def get_filtered_aspect_archetype_answers_dict(self, aspect_archetype_answers_dict, answer_pertinence_threshold=None, answer_to_question_max_similarity_threshold=None, answer_to_answer_max_similarity_threshold=None, one_answer_per_sentence=False, **args):
		self.answer_retriever.logger.info("Adding question_pertinence_set to answer_list..")
		for archetype_dict in aspect_archetype_answers_dict.values():
			self.answer_retriever.get_answer_question_pertinence_dict(archetype_dict, update_answers=True)
		self.answer_retriever.logger.info("Getting cumulative pertinence and LoD, per archetype..")
		filtered_aspect_archetype_answers_dict = {}
		for aspect_uri, archetype_dict in self.answer_retriever.tqdm(aspect_archetype_answers_dict.items(), total=len(aspect_archetype_answers_dict)):
			archetype_fitness_dict = filtered_aspect_archetype_answers_dict[aspect_uri] = {}
			for archetype,answer_iter in archetype_dict.items():
				# Considered only sentences that can be seen
				if answer_pertinence_threshold is not None:
					answer_iter = filter(lambda x: x['confidence']>=answer_pertinence_threshold, answer_iter)
				if one_answer_per_sentence:
					answer_iter = unique_everseen(answer_iter, key=lambda x: x['sentence'])
				answer_list = list(answer_iter)
				if len(answer_list) == 0:
					archetype_fitness_dict[archetype] = answer_list
					continue
				# Answers contained in the question are not valid
				if answer_to_question_max_similarity_threshold is not None:
					answer_list = self.answer_retriever.sentence_classifier.filter_by_similarity_to_target(
						answer_list, 
						[self.answer_retriever.kg_manager.get_label(aspect_uri)], 
						threshold=answer_to_question_max_similarity_threshold, 
						source_key=lambda a: a['abstract'], 
						target_key=lambda q: q,
					)
				if len(answer_list) == 0:
					archetype_fitness_dict[archetype] = answer_list
					continue
				# Ignore similar-enough sentences with lower pertinence
				if answer_to_answer_max_similarity_threshold is not None:
					answer_list = self.answer_retriever.sentence_classifier.remove_similar_labels(
						answer_list, 
						threshold=answer_to_answer_max_similarity_threshold, 
						key=lambda x: (x['abstract'],x['sentence']),
						without_context=False,
					)
				archetype_fitness_dict[archetype] = answer_list
		return filtered_aspect_archetype_answers_dict

	def get_aspect_archetype_fitness_dict(self, aspect_archetype_answers_dict, archetype_fitness_options):
		aspect_archetype_answers_dict = self.get_filtered_aspect_archetype_answers_dict(aspect_archetype_answers_dict, **archetype_fitness_options)
		return {
			aspect_uri: {
				archetype: self.get_fitness_from_answer_list(answer_list)
				for archetype, answer_list in archetype_dict.items()
			}
			for aspect_uri, archetype_dict in aspect_archetype_answers_dict.items()
		}

	def get_archetype_fitness_dict(self, aspect_archetype_answers_dict):
		archetype_aspects_fitness_dict = {}
		archetype_overlap_dict = {}
		for aspect_uri, archetype_dict in aspect_archetype_answers_dict.items():
			for archetype, answer_list in archetype_dict.items():
				aspect_fitness_list = archetype_aspects_fitness_dict.get(archetype, None)
				if aspect_fitness_list is None:
					aspect_fitness_list = archetype_aspects_fitness_dict[archetype] = []
				# Get explanatory illocution
				aspect_fitness_list.append(self.get_fitness_from_answer_list(answer_list))
				# Keep track of the list of covered archetypes in a different structure
				overlap_list = archetype_overlap_dict.get(archetype, None)
				if overlap_list is None:
					overlap_list = archetype_overlap_dict[archetype] = []
				covered_archetypes = flatten((get_archetype_set(answer_dict) for answer_dict in answer_list), as_list=True)
				overlap_list.append((covered_archetypes,len(answer_list)))
		self.answer_retriever.logger.info("Computing total relevance and LoD per archetype..")
		archetype_fitness_dict = {}
		for archetype, aspect_fitness_list in archetype_aspects_fitness_dict.items():
			aspect_fitness = aspect_fitness_list[0]
			archetype_fitness = archetype_fitness_dict[archetype] = {}
			for k in aspect_fitness.keys():
				k_vector = np.fromiter(map(lambda x:x[k], aspect_fitness_list), dtype=np.float32)
				archetype_fitness[k] = {
					"sum": float(np.sum(k_vector)),
					"lower_q": float(np.quantile(k_vector,0.25)),
					"median": float(np.quantile(k_vector,0.5)),
					"upper_q": float(np.quantile(k_vector,0.75)),
					"min": float(np.min(k_vector)),
					"mean": float(np.mean(k_vector)),
					"std": float(np.std(k_vector)),
					"max": float(np.max(k_vector)),
				} if k_vector.size > 0 else {
					"sum": 0,
					"lower_q": 0,
					"median": 0,
					"upper_q": 0,
					"min": 0,
					"mean": 0,
					"std": 0,
					"max": 0,
				}
		self.answer_retriever.logger.info("Adding overlap ratios to fitness_dict..")
		for archetype, overlap_list in archetype_overlap_dict.items():
			answers_count = sum(map(lambda x:x[1], overlap_list))
			overlap_dict = Counter(flatten(map(lambda x:x[0], overlap_list)))
			archetype_fitness_dict[archetype]['overlap_ratios'] = {
				k: v/answers_count
				for k,v in sorted(overlap_dict.items(), key=lambda x: x[-1], reverse=True)
			}
		# print("Sorting archetypes by explanatory illocution..")
		# sorted_archetype_fitness_dict = dict(sorted(archetype_fitness_dict.items(), key=lambda x: self.get_explanatory_illocution_from_archetype_fitness(x[-1]), reverse=True))
		# return sorted_archetype_fitness_dict
		return archetype_fitness_dict

	def get_abstract_dox_dict(self, aspect_dox_dict): # normalise abstract concepts' WeDoX (it should not be lower than their subclasses)
		self.answer_retriever.logger.info('Get WeDoX of best sub-class, a.k.a. Abstract WeDoX')
		return {
			aspect_uri: max((
				aspect_dox_dict.get(a,0) 
				for a in self.answer_retriever.kg_manager.get_sub_classes(
					set(
						self.answer_retriever.kg_manager.get_equivalent_concepts(aspect_uri)
					)
				)
			))
			for aspect_uri in aspect_dox_dict.keys()
		}

	def get_sentence_aspect_dox_dict(self, aspect_uri_list, query_template_list=False, question_generator=None, overview_options=None, archetype_fitness_options=None):
		if archetype_fitness_options is None:
			archetype_fitness_options = {}
		if overview_options is None:
			overview_options = {}
		aspect_archetype_answers_dict = self.extract_archetypal_answers_per_aspect(
			aspect_uri_iter=aspect_uri_list,
			query_template_list=query_template_list,
			question_generator=question_generator,
			archetypal_qa_options=overview_options
		)
		aspect_archetype_answers_dict = self.get_filtered_aspect_archetype_answers_dict(aspect_archetype_answers_dict, **archetype_fitness_options)
		sentence_aspect_dox_dict = {}
		for aspect_uri, archetype_dict in aspect_archetype_answers_dict.items():
			for archetype, answer_list in archetype_dict.items():
				# if is_new_content_fn is not None:
				# 	answer_list = map(is_new_content_fn, answer_list)
				for answer in answer_list:
					sentence = answer['sentence']
					sentence_dox = answer['confidence'] 
					aspect_dox_dict = sentence_aspect_dox_dict.get(sentence, None)
					if aspect_dox_dict is None:
						aspect_dox_dict = sentence_aspect_dox_dict[sentence] = {}
					aspect_dox = aspect_dox_dict.get(aspect_uri, None)
					if aspect_dox is None:
						aspect_dox = aspect_dox_dict[aspect_uri] = {}
					aspect_dox[archetype] = sentence_dox
		return sentence_aspect_dox_dict

	def estimate(self, aspect_uri_iter=None, query_template_list=None, question_generator=None, archetypal_qa_options=None, **archetype_fitness_options):
		aspect_archetype_answers_dict = self.extract_archetypal_answers_per_aspect(
			aspect_uri_iter=aspect_uri_iter,
			query_template_list=query_template_list,
			question_generator=question_generator,
			archetypal_qa_options=archetypal_qa_options,
		)
		aspect_archetype_answers_dict = self.get_filtered_aspect_archetype_answers_dict(aspect_archetype_answers_dict, **archetype_fitness_options)
		self.answer_retriever.logger.info(f'Aspect-Archetype-Answers dict: {json.dumps(aspect_archetype_answers_dict, indent=4)}')
		self.answer_retriever.logger.info(f'Aspect-Archetype dict: {json.dumps(dict((q, dict(sorted([(k, np.mean([a["confidence"] for a in v]) if v else 0) for k,v in w.items()], key=lambda x: x[-1], reverse=True))) for q,w in aspect_archetype_answers_dict.items()), indent=4)}')
		archetype_fitness_dict = self.get_archetype_fitness_dict(aspect_archetype_answers_dict)
		# print('Archetype Fitness:', json.dumps(archetype_fitness_dict, indent=4))
		return self.get_degree_of_explainability_from_archetype_fitness(archetype_fitness_dict)

	@staticmethod
	def get_degree_of_explainability_from_archetype_fitness(archetype_fitness_dict):
		# print(json.dumps(archetype_fitness_dict, indent=4))
		dox_archetype_iter = zip(archetype_fitness_dict.keys(), map(DoXEstimator.get_explanatory_illocution_from_archetype_fitness, archetype_fitness_dict.values()))
		dox_archetype_iter = sorted(dox_archetype_iter, key=lambda x: x[-1], reverse=True)
		return dict(dox_archetype_iter)

	@staticmethod
	def get_weighted_degree_of_explainability(dox, archetype_weight_dict=None):
		if not dox:
			return 0
		if not archetype_weight_dict:
			return sum(dox.values())/len(dox)
		weighted_degree_of_explainability = 0
		for archetype, weight in archetype_weight_dict.items():
			if archetype in dox:
				weighted_degree_of_explainability += dox[archetype]*weight
		return weighted_degree_of_explainability

