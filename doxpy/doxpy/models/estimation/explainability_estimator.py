import json
import numpy as np
from more_itertools import unique_everseen
from tqdm import tqdm
from collections import Counter
from itertools import islice

from doxpy.misc.cache_lib import load_or_create_cache, create_cache, load_cache
from doxpy.misc.levenshtein_lib import remove_similar_labels, labels_are_contained

from doxpy.misc.doc_reader import DocParser
from doxpy.misc.jsonld_lib import *
from doxpy.models.reasoning.question_answerer import QuestionAnswerer
from doxpy.models.reasoning import is_not_wh_word, singlefy

def get_stats_dict(v):
	return {
		"lower_q": np.quantile(v,0.25),
		"median": np.quantile(v,0.5),
		"upper_q": np.quantile(v,0.75),
		"min": min(v),
		"mean": np.mean(v),
		"std": np.std(v),
		"max": max(v)
	}

class ExplainabilityEstimator:
	@staticmethod
	def get_explanatory_illocution_from_archetype_fitness(fitness_dict):
		# tot_relevance describes well how all the information units are relevant together, the more details, the higher is explanatory illocution. If the minimum relevance is high (long-tail distribution), tot_relevance is more sensitive to the number of information units than to their relevance. For example, according to this metric, many low-relevance units beat a single high-relevance unit even if the best option would be a single high-relevance unit.
		# weighted_level_of_detail = fitness_dict["tot_relevance"]["median"]
		# top5_relevance is as tot_relevance, but with the long-tail of information units cut.
		# top5_weighted_level_of_detail = fitness_dict["top5_relevance"]["median"]
		# max_relevance does not take into account the number of different information units, but it's good to describe what is the maximum precision of information units for this aspect.
		# maximum_explanatory_relevance = fitness_dict["max_relevance"]["median"]
		# explanatory_illocution = weighted_level_of_detail + top1_weighted_level_of_detail, because this way higher importance is given to the best 10 information units
		return fitness_dict["tot_relevance"]["median"]

	def __init__(self, question_answerer:QuestionAnswerer, qa_extractor_options=None, archetypal_questions_dict=None):
		self.question_answerer = question_answerer
		self._qa_extractor_options = qa_extractor_options

		self.concept2qa_dict = None
		self.paragraph2qas_dict = None
		self.aspect_archetype_answers_dict = None

		self.archetypal_questions_dict = self.question_answerer.archetypal_questions_dict if archetypal_questions_dict is None else archetypal_questions_dict

	def store_cache(self, cache_name):
		if self.concept2qa_dict is not None:
			create_cache(cache_name+'.concept2qa_dict.pkl', lambda: self.concept2qa_dict)
		if self.paragraph2qas_dict is not None:
			create_cache(cache_name+'.paragraph2qas_dict.pkl', lambda: self.paragraph2qas_dict)
		if self.aspect_archetype_answers_dict is not None:
			create_cache(cache_name+'.aspect_archetype_answers_dict.pkl', lambda: self.aspect_archetype_answers_dict)

	def load_cache(self, cache_name):
		concept2qa_dict = load_cache(cache_name+'.concept2qa_dict.pkl')
		if concept2qa_dict is not None:
			self.concept2qa_dict = concept2qa_dict

		paragraph2qas_dict = load_cache(cache_name+'.paragraph2qas_dict.pkl')
		if paragraph2qas_dict is not None:
			self.paragraph2qas_dict = paragraph2qas_dict

		aspect_archetype_answers_dict = load_cache(cache_name+'.aspect_archetype_answers_dict.pkl')
		if aspect_archetype_answers_dict is not None:
			self.aspect_archetype_answers_dict = aspect_archetype_answers_dict

	def extract_archetypal_answers_per_aspect(self, aspect_uri_list=None, **archetypal_qa_options):
		archetypal_qa_options['sort_archetypes_by_relevance'] = False # set it to False, important!
		archetypal_qa_options['minimise'] = False # set it to False, important!
		archetypal_qa_options['answer_pertinence_threshold'] = None
		archetypal_qa_options['answer_to_answer_similarity_threshold'] = None
		archetypal_qa_options['answer_to_question_similarity_threshold'] = None
		archetypal_qa_options['with_annotation'] = False
		print(f"Extracting aspect_archetype_answers_dict with top_k {archetypal_qa_options.get('top_k',None)}..")
		question_archetype_dict = {
			v:k
			for k,v in self.archetypal_questions_dict.items()
		}
		# print('Extract archetypal answer_list for each aspect')
		self.aspect_archetype_answers_dict = {}
		if not aspect_uri_list:
			aspect_uri_list = list(unique_everseen(self.question_answerer.concept_classifier.ids))
		query_template_list = list(self.archetypal_questions_dict.values())
		for aspect_uri in tqdm(aspect_uri_list):
			question_answer_dict = self.question_answerer.get_concept_overview(
				query_template_list=query_template_list, 
				concept_uri=aspect_uri, 
				**archetypal_qa_options,
			)
			self.aspect_archetype_answers_dict[aspect_uri] = {
				question_archetype_dict[question]: answer_list
				for question,answer_list in question_answer_dict.items()
			}
		return self.aspect_archetype_answers_dict

	def get_level_of_detail_dict(self, answer_to_answer_similarity_threshold=0.9502):
		# print('Get the list of unique QAs related to each aspect')
		if self.concept2qa_dict is None:
			self.build_concept2qa_dict(answer_to_answer_similarity_threshold)
		# print('Filter out non important aspects')
		important_concept2qa_iter = filter(lambda x:x[0] in self.question_answerer.relevant_aspect_set, self.concept2qa_dict.items())
		# print(f"LoD per aspect: {json.dumps(dict(important_concept2qa_iter), indent=4)}")ù
		important_concept_lod_iter = map(lambda x:(x[0],len(x[1])), important_concept2qa_iter)
		important_concept_lod_dict = dict(important_concept_lod_iter)
		return {
			'aspect_lod': important_concept_lod_dict,
			'global_lod': get_stats_dict(list(important_concept_lod_dict.values())),
		}

	def get_seen_sentence_set(self, aspect_archetype_answers_dict, answer_pertinence_threshold=0.15, answer_to_question_similarity_threshold=None, answer_to_answer_similarity_threshold=None, minimise=False, sort_archetypes_by_relevance=False, set_of_archetypes_to_consider=None, answer_horizon=10, remove_duplicate_answer_sentences=True, **args):
		seen_sentence_set = set()
		if minimise:
			aspect_archetype_answers_dict = {
				aspect_uri: self.question_answerer.minimise_question_answer_dict(archetype_dict)
				for aspect_uri, archetype_dict in aspect_archetype_answers_dict.items()
			}
		for aspect_uri, archetype_dict in aspect_archetype_answers_dict.items():
			if aspect_uri not in self.question_answerer.overview_aspect_set:
				continue
			archetype_dict_items = archetype_dict.items()
			if sort_archetypes_by_relevance:
				archetype_dict_items = sorted(archetype_dict_items, key=lambda x: x[-1][0]['confidence'], reverse=True)
			if set_of_archetypes_to_consider is not None:
				archetype_dict_items = filter(lambda x: x[0] in set_of_archetypes_to_consider, archetype_dict_items)
			for archetype,answer_list in archetype_dict_items:
				if answer_pertinence_threshold is not None:
					answer_list = list(filter(lambda x: x['semantic_similarity']>=answer_pertinence_threshold, answer_list))
				if len(answer_list) == 0:
					continue
				# Answers contained in the question are not valid
				if answer_to_question_similarity_threshold is not None:
					answer_list = self.question_answerer.sentence_classifier.filter_by_similarity_to_target(
						answer_list, 
						[self.question_answerer.get_label(aspect_uri)], 
						threshold=answer_to_question_similarity_threshold, 
						source_key=lambda a: a['abstract'], 
						target_key=lambda q: q,
					)
				if len(answer_list) == 0:
					continue
				# Ignore similar-enough sentences with lower pertinence
				if answer_to_answer_similarity_threshold is not None:
					answer_list = self.question_answerer.sentence_classifier.remove_similar_labels(
						answer_list, 
						threshold=answer_to_answer_similarity_threshold, 
						key=lambda x: (x['abstract'],x['sentence']),
						without_context=False,
					)
				answer_sentence_iter = map(lambda x: x['sentence'], answer_list)
				answer_sentence_iter = unique_everseen(answer_sentence_iter)
				if remove_duplicate_answer_sentences:
					answer_sentence_iter = filter(lambda x: x not in seen_sentence_set, answer_sentence_iter)
				if answer_horizon:
					answer_sentence_iter = islice(answer_sentence_iter, answer_horizon)
				seen_sentence_set |= set(answer_sentence_iter)
		return seen_sentence_set

	def get_fitness_from_answer_list(self, answer_list=None):
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
		answers_relevance = [a['semantic_similarity'] for a in answer_list]
		median_relevance = np.median(answers_relevance)
		return { # candidate main statistics: max_relevance, tot_relevance, top10_relevance, mean_relevance
			"max_relevance": max(answers_relevance), # max_relevance does not take into account the number of different information units, but it's good to describe what is the maximum precision of information units for this aspect.
			"min_relevance": min(answers_relevance),
			"tot_relevance": sum(answers_relevance), # tot_relevance describes well how all the information units are relevant together. If the minimum relevance is high (long-tail distribution), tot_relevance is more sensitive to the number of information units than to their relevance (i.e. many low-relevance units beat a single high-relevance unit).
			"top5_relevance": sum(answers_relevance[:5]), # As tot_relevance, but the long-tail of information units is cut.
			# "head_relevance": sum(filter(lambda x: x >= median_relevance, answers_relevance)),
			# "tail_relevance": sum(filter(lambda x: x < median_relevance, answers_relevance)),
			"mean_relevance": np.mean(answers_relevance), # Differently to tot_relevance, a single high-relevance unit is weighted more than many low-relevance units, but the relevance distribution of information units has a long-tail. Therefore the more information units, the lower this value is likely to be.
			"median_relevance": median_relevance,
			"relevant_answers": len(answers_relevance),
			# "unit_LoD": len(unit_qa_list),
			# "context_LoD": len(context_qa_list),
		}

	def get_archetype_fitness_dict(self, overview_options, aspect_uri_list=None, answer_pertinence_threshold=None, answer_to_question_similarity_threshold=0.9502, answer_to_answer_similarity_threshold=0.9502, only_overview_exploration=False):
		if self.aspect_archetype_answers_dict is None:
			self.extract_archetypal_answers_per_aspect(aspect_uri_list=aspect_uri_list, **dict(overview_options))
		print("Adding question_pertinence_set to answer_list..")
		aspect_archetype_answers_dict = self.aspect_archetype_answers_dict
		for archetype_dict in aspect_archetype_answers_dict.values():
			self.question_answerer.get_answer_question_pertinence_dict(archetype_dict, update_answers=True)
		print("Getting cumulative pertinence and LoD, per archetype..")
		archetype_aspects_fitness_dict = {}
		archetype_overlap_dict = {}
		get_archetype_set = lambda x: [qp.archetype for qp in x['question_pertinence_set']]
		seen_sentence_set = self.get_seen_sentence_set(
			aspect_archetype_answers_dict, 
			**dict(overview_options),
		) if only_overview_exploration else None
		set_of_archetypes_to_consider = overview_options.get('set_of_archetypes_to_consider', None)
		if set_of_archetypes_to_consider:
			print('Considering only these archetypes:', set_of_archetypes_to_consider)
		for aspect_uri, archetype_dict in tqdm(list(aspect_archetype_answers_dict.items())):
			# if aspect_uri not in self.question_answerer.relevant_aspect_set:
			# 	continue
			answer_archetype_dict = aspect_archetype_answers_dict[aspect_uri]
			for archetype,answer_iter in archetype_dict.items():
				aspect_fitness_list = archetype_aspects_fitness_dict.get(archetype, None)
				if aspect_fitness_list is None:
					aspect_fitness_list = archetype_aspects_fitness_dict[archetype] = []
				# Considered only sentences that can be seen
				if answer_pertinence_threshold is not None:
					answer_iter = filter(lambda x: x['semantic_similarity']>=answer_pertinence_threshold, answer_iter)
				if seen_sentence_set:
					answer_iter = filter(lambda x: x['sentence'] in seen_sentence_set, answer_iter)
				if set_of_archetypes_to_consider:
					answer_iter = filter(lambda x: len(set_of_archetypes_to_consider.intersection(get_archetype_set(x))) > 0, answer_iter)
				answer_list = list(answer_iter)
				if len(answer_list) == 0:
					aspect_fitness_list.append(self.get_fitness_from_answer_list(answer_list))
					continue
				# Answers contained in the question are not valid
				if answer_to_question_similarity_threshold is not None:
					answer_list = self.question_answerer.sentence_classifier.filter_by_similarity_to_target(
						answer_list, 
						[self.question_answerer.get_label(aspect_uri)], 
						threshold=answer_to_question_similarity_threshold, 
						source_key=lambda a: a['abstract'], 
						target_key=lambda q: q,
					)
				if len(answer_list) == 0:
					aspect_fitness_list.append(self.get_fitness_from_answer_list(answer_list))
					continue
				# Ignore similar-enough sentences with lower pertinence
				if answer_to_answer_similarity_threshold is not None:
					answer_list = self.question_answerer.sentence_classifier.remove_similar_labels(
						answer_list, 
						threshold=answer_to_answer_similarity_threshold, 
						key=lambda x: (x['abstract'],x['sentence']),
						without_context=False,
					)
				# Get explanatory illocution
				aspect_fitness_list.append(self.get_fitness_from_answer_list(answer_list))
				# Keep track of the list of covered archetypes in a different structure
				overlap_list = archetype_overlap_dict.get(archetype, None)
				if overlap_list is None:
					overlap_list = archetype_overlap_dict[archetype] = []
				covered_archetypes = sum((get_archetype_set(answer_dict) for answer_dict in answer_list), [])
				overlap_list.append((covered_archetypes,len(answer_list)))
		print("Computing total relevance and LoD per archetype..")
		archetype_fitness_dict = {}
		for archetype, aspect_fitness_list in archetype_aspects_fitness_dict.items():
			aspect_fitness = aspect_fitness_list[0]
			archetype_fitness = archetype_fitness_dict[archetype] = {}
			for k in aspect_fitness.keys():
				k_list = list(map(lambda x:x[k], aspect_fitness_list))
				archetype_fitness[k] = get_stats_dict(k_list)
		print("Adding overlap ratios to fitness_dict..")
		for archetype, overlap_list in archetype_overlap_dict.items():
			answers_count = sum(map(lambda x:x[1], overlap_list))
			overlap_dict = Counter(sum(map(lambda x:x[0], overlap_list), []))
			archetype_fitness_dict[archetype]['overlap_ratios'] = {
				k: v/answers_count
				for k,v in sorted(overlap_dict.items(), key=lambda x: x[-1], reverse=True)
			}
		# print("Sorting archetypes by explanatory illocution..")
		# sorted_archetype_fitness_dict = dict(sorted(archetype_fitness_dict.items(), key=lambda x: self.get_explanatory_illocution_from_archetype_fitness(x[-1]), reverse=True))
		# return sorted_archetype_fitness_dict
		return archetype_fitness_dict

	@staticmethod
	def get_degree_of_explainability_from_archetype_fitness(archetype_fitness_dict):
		dox_archetype_iter = zip(archetype_fitness_dict.keys(), map(ExplainabilityEstimator.get_explanatory_illocution_from_archetype_fitness, archetype_fitness_dict.values()))
		dox_archetype_iter = sorted(dox_archetype_iter, key=lambda x: x[-1], reverse=True)
		return dict(dox_archetype_iter)

	@staticmethod
	def get_weighted_degree_of_explainability(dox, archetype_weight_dict=None):
		if not archetype_weight_dict:
			return np.mean(dox.values())
		weighted_degree_of_explainability = 0
		for archetype, weight in archetype_weight_dict.items():
			if archetype in dox:
				weighted_degree_of_explainability += dox[archetype]*weight
		return weighted_degree_of_explainability
