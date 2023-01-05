from doxpy.misc.doc_reader import DocParser
from doxpy.models.knowledge_extraction.concept_extractor import ConceptExtractor as CE
from doxpy.models.model_manager import ModelManager
from doxpy.misc.utils import *
import re
# from pathos.threading import ThreadPool as Pool
from pathos.multiprocessing import ProcessPool as Pool

class CoupleExtractor(CE):
	# PREDICATE_COMPONENT = [ # https://universaldependencies.org/u/dep/all.html
	# 	'prt',		# particle
	# 	'neg',		# negation modifier
	# 	'auxpass',	# auxiliary (passive)
	# 	'advcl',	# adverbial clause modifier
	# 	'agent',	# agent
	# 	'acomp',	# adjectival complement
	# 	'xcomp',	# open clausal complement
	# 	'pcomp',	# complement of preposition
	# 	'ccomp',	# clausal complement
	# 	'prep',		# prepositional modifier
	# ]
	# HIDDEN_PREDICATE_COMPONENT = [
	# 	'aux',		# auxiliaries
	# 	'mark', 	# marker - https://universaldependencies.org/docs/en/dep/mark.html
	# 	'advmod',	# adverbial modifier
	# 	'cc',		# coordinating conjunction
	# ]
	# PREDICATE_REGEXP = re.compile('|'.join(PREDICATE_COMPONENT+HIDDEN_PREDICATE_COMPONENT))
	PREDICATE_EXPANSION_COMPONENT = set([ # https://universaldependencies.org/u/dep/all.html
		'det',		# determiner
		'poss',		# possession modifier
		'case',		# case marking
	])
	PREDICATE_EXPANSION_REGEXP = re.compile('|'.join(PREDICATE_EXPANSION_COMPONENT))
	CC_FILTER_FN = lambda x: x.pos_=='PUNCT' or x.dep_=='cc' # punctuation and conjunctions
	
	@staticmethod
	def is_passive(span): # return true if the sentence is passive - at the moment a sentence is assumed to be passive if it has an auxpass verb
		for token in span:
			if CE.get_token_dependency(token) == "auxpass":
				return True
		return False

	@staticmethod
	def is_verbal(span):
		for token in span:
			if token.pos_ == 'VERB':
				return True
			if token.pos_ == 'AUX':
				return True
		return False

	@staticmethod
	def has_subject(span): # return true if the sentence has a subject
		# ancestors = list(CE.get_token_ancestors(span[0]))
		# if not ancestors:
		# 	ancestors = list(CE.get_token_ancestors(span[1]))
		# if not ancestors:
		# 	return False
		# root_node = ancestors[-1]
		for token in span:
			if 'subj' in token.dep_:
				return True
		return False

	@staticmethod
	def is_coreferencing(span):
		for token in span:
			# if token.pos_ == 'DET': # if token is a determiner
			if re.match('PRON|DET', token.pos_): # if token is a pronoun or a determiner
				# if re.search('nsubj', token.dep_): # and if token is a direct subject
				if re.search('nsubj|dobj', token.dep_): # and if token is a direct subject or object
					return True
		return False

	@staticmethod
	def is_at_core(concept):
		concept_span = concept['concept']['span']
		return len(concept_span)==1 and len(concept['concept_core'])==1 and concept['concept_core'][0]['span'][0] == concept_span[0]

	@staticmethod
	def get_couple_uid(couple):
		return (CE.get_concept_dict_uid(couple['concept']), CE.get_concept_dict_uid(couple['predicate']), couple['dependency'])

	@staticmethod
	def is_in_predicate(x,predicate_span):
		return x.idx > predicate_span[0].idx and x.idx < predicate_span[-1].idx

	@staticmethod
	def trim_noise(token_list):
		def trim_fn(x):
			dep = CE.get_token_dependency(x)
			return dep == 'cc' or dep == 'prep' or dep == 'punct'
		return CE.trim(token_list, trim_fn)

	@staticmethod
	def expand_predicate_core(predicate_set, subj_obj_set): # enrich predicate set with details, adding hidden related concepts (n-ary relations)
		hidden_related_concept_set = set((
			hidden_related_concept
			for predicate_element in predicate_set
			for hidden_related_concept in CE.get_token_descendants(predicate_element, lambda x: x not in subj_obj_set and x not in predicate_set)
		))
		# add missing determiners and possession modifiers
		subj_obj_determiners = set()
		for x in subj_obj_set:
			subj_obj_determiners |= set(CE.get_token_descendants(x, lambda x: re.match(CoupleExtractor.PREDICATE_EXPANSION_REGEXP, x.dep_)))
		# merge all
		return predicate_set | hidden_related_concept_set | subj_obj_determiners

	@staticmethod
	def get_common_verb(subj, obj):
		obj_super_set = set(obj.ancestors)
		if subj in obj_super_set: # if true, no verb can connect subj and obj directly
			return None

		subj_super_set = set(subj.ancestors)
		if obj in subj_super_set: # if true, no verb can connect subj and obj directly
			return None

		subj_super_set.add(subj)
		obj_super_set.add(obj)
		inter_set = subj_super_set.intersection(obj_super_set)
		if len(inter_set)==0: # subj and obj have no common predicate
			return None

		# subj_obj_set = set([subj,obj])
		subj_path_to_inter,subj_junction = CE.find_path_to_closest_in_set(subj,inter_set)
		if subj_junction:
			subj_path_to_inter.add(subj_junction)
		# if len(subj_path_to_inter.difference(subj_obj_set).intersection(core_set)) > 0: # if true, no verb connects subj and obj directly
		# 	return None
		obj_path_to_inter,obj_junction = CE.find_path_to_closest_in_set(obj,inter_set)
		if obj_junction:
			obj_path_to_inter.add(obj_junction)
		# if len(obj_path_to_inter.difference(subj_obj_set).intersection(core_set)) > 0: # if true, no verb connects subj and obj directly
		# 	return None

		# Get predicate set
		predicate_core_set = subj_path_to_inter.union(obj_path_to_inter)
		if len(predicate_core_set)==0:
			return None
		if not CoupleExtractor.is_verbal(predicate_core_set):
			return None
		# Ignore predicates composed by multiple sub-predicates
		verb_iter = filter(lambda x: x.pos_ == "VERB", predicate_core_set)
		last_v = next(verb_iter, None)
		if not last_v:
			verb_iter = filter(lambda x: x.pos_ == "AUX", predicate_core_set)
			last_v = next(verb_iter, None)
		last_v_i = last_v.i
		for v in verb_iter:
			if last_v_i+1 != v.i: # if true, no verb connects subj and obj directly
				return None
			last_v_i = v.i
		# Enrich predicate set with details, adding hidden related concepts (n-ary relations)
		predicate_set = CoupleExtractor.expand_predicate_core(predicate_core_set, subj_obj_set=set((subj,obj)))
		return sorted(predicate_set, key=lambda x: x.idx)

	@staticmethod
	def identify_cores_role(core, other_core):
		core_is_obj = re.match(CE.OBJ_REGEXP, CE.get_token_dependency(core)) is not None
		other_core_is_obj = re.match(CE.OBJ_REGEXP, CE.get_token_dependency(other_core)) is not None
		core_is_subj = re.match(CE.SUBJ_REGEXP, CE.get_token_dependency(core)) is not None
		other_core_is_subj = re.match(CE.SUBJ_REGEXP, CE.get_token_dependency(other_core)) is not None
		# Handle ambiguous dependencies
		# ambiguous_dep = core_is_obj != other_core_is_subj or core_is_subj != other_core_is_obj or core_is_obj == core_is_subj # or other_core_is_obj == other_core_is_subj
		if core_is_obj!=other_core_is_obj:
			if core_is_obj:
				subj = other_core
				obj = core
			else:
				subj = core
				obj = other_core
		elif core_is_subj!=other_core_is_subj:
			if core_is_subj:
				subj = core
				obj = other_core
			else:
				subj = other_core
				obj = core
		else: # position-based decision
			if core.idx < other_core.idx:
				subj = core
				obj = other_core
			else:
				subj = other_core
				obj = core
		return subj,obj

	@staticmethod
	def get_predicate_core_between_concepts(core, other_core, core_set, avoid_jumps=False):
		# Search for indirect connections with other concepts
		core_super_set = set(core.ancestors) # do not use CE.get_token_ancestors here, it messes up with conjunctions
		core_super_set.add(core)
		other_core_super_set = set(other_core.ancestors)
		other_core_super_set.add(other_core)
		inter = core_super_set.intersection(other_core_super_set)
		if len(inter)==0: # core and other_core are not connected, continue
			return None
		del core_super_set
		del other_core_super_set
		# get paths connecting cores to each other
		core_path_to_inter,core_junction = CE.find_path_to_closest_in_set(core,inter)
		if core_junction:
			core_path_to_inter.add(core_junction)
		subj_obj_set = set((core,other_core))
		core_path_to_inter = core_path_to_inter.difference(subj_obj_set)
		is_jumping = len(core_path_to_inter.intersection(core_set)) > 0
		if avoid_jumps and is_jumping:
			return None
		other_core_path_to_inter,other_core_junction = CE.find_path_to_closest_in_set(other_core,inter)
		del inter
		if other_core_junction:
			other_core_path_to_inter.add(other_core_junction)
		other_core_path_to_inter = other_core_path_to_inter.difference(subj_obj_set)
		is_jumping = len(other_core_path_to_inter.intersection(core_set)) > 0
		if avoid_jumps and is_jumping:
			return None
		# Get predicate set
		predicate_core_set = core_path_to_inter.union(other_core_path_to_inter)
		if len(predicate_core_set)==0:
			return None
		return predicate_core_set

	@staticmethod
	def templatize_predicate_span(predicate_span, subj, obj):
		get_concept_dict = lambda span: CoupleExtractor.get_concept_dict_from_span(span)#, hidden_dep_list=CoupleExtractor.HIDDEN_PREDICATE_COMPONENT)
		# templatize predicate_dict
		triple_span = sorted(predicate_span + [subj,obj], key=lambda x:x.idx)
		# triple_span = CoupleExtractor.trim_noise(triple_span)
		subj_pos = triple_span.index(subj)
		obj_pos = triple_span.index(obj)
		if subj_pos < obj_pos:
			left_pivot = subj_pos
			right_pivot = obj_pos
			left_is_subj = True
		else:
			left_pivot = obj_pos
			right_pivot = subj_pos
			left_is_subj = False
		templatized_lemma = []
		templatized_text = []
		if left_pivot > 0:
			left_pdict = get_concept_dict(triple_span[:left_pivot])
			templatized_lemma.append(left_pdict['lemma'])
			templatized_text.append(left_pdict['text'])
		templatized_lemma.append('{subj}' if left_is_subj else '{obj}')
		templatized_text.append('{subj}' if left_is_subj else '{obj}')
		if right_pivot > left_pivot+1:
			middle_pdict = get_concept_dict(triple_span[left_pivot+1:right_pivot])
			templatized_lemma.append(middle_pdict['lemma'])
			templatized_text.append(middle_pdict['text'])
		templatized_lemma.append('{obj}' if left_is_subj else '{subj}')
		templatized_text.append('{obj}' if left_is_subj else '{subj}')
		if right_pivot < len(triple_span)-1:
			right_pdict = get_concept_dict(triple_span[right_pivot+1:])
			templatized_lemma.append(right_pdict['lemma'])
			templatized_text.append(right_pdict['text'])
		return templatized_text, templatized_lemma

	@staticmethod
	def get_triplet(core_concept, other_core_concept, core_set, avoid_jumps=False): # can be one per core in core_set
		core, concept_dict_list = core_concept
		other_core, other_concept_dict_list = other_core_concept
		predicate_core_set = CoupleExtractor.get_predicate_core_between_concepts(core, other_core, core_set, avoid_jumps=avoid_jumps)
		if not predicate_core_set:
			return None
		# Enrich predicate set with details, adding hidden related concepts (n-ary relations)
		predicate_set = CoupleExtractor.expand_predicate_core(predicate_core_set, subj_obj_set=set((core,other_core)))
		# Add missing conjunctions
		if core in other_core.children:
			predicate_set |= set(filter(CoupleExtractor.CC_FILTER_FN, other_core.children))
		elif other_core in core.children:
			predicate_set |= set(filter(CoupleExtractor.CC_FILTER_FN, core.children))
		# Get predicate spans
		predicate_span = sorted(predicate_set, key=lambda x: x.idx)
		del predicate_set
		predicate_core_span = sorted(predicate_core_set, key=lambda x: x.idx)
		del predicate_core_set
		# # Remove consecutive punctuations
		# non_consecutive_puncts = set([',',';'])
		# predicate_span = [
		# 	v 
		# 	for i, v in enumerate(predicate_span) 
		# 	if i == 0 
		# 	or v.pos_ != 'PUNCT' 
		# 	or v.pos_ != predicate_span[i-1].pos_ 
		# 	or v.text not in non_consecutive_puncts 
		# 	or predicate_span[i-1].text not in non_consecutive_puncts
		# ]
		
		subj_core, obj_core = CoupleExtractor.identify_cores_role(core, other_core)
		verb_span = CoupleExtractor.get_common_verb(subj_core, obj_core)
		
		get_concept_dict = lambda span: CoupleExtractor.get_concept_dict_from_span(span)#, hidden_dep_list=CoupleExtractor.HIDDEN_PREDICATE_COMPONENT)
		
		# get predicate_dict
		predicate_dict = get_concept_dict(predicate_span)
		predicate_text, predicate_lemma = CoupleExtractor.templatize_predicate_span(predicate_span, subj_core, obj_core)
		predicate_dict['text'] = ' '.join(predicate_text)
		predicate_dict['lemma'] = ' '.join(predicate_lemma)
		
		# print(f'Discarding concept "{concept_dict["concept"]["text"]}", because it intersects its predicate: "{predicate_dict["predicate"]["text"]}".')
		concept_dict_list = sorted(concept_dict_list, key=lambda x: CE.get_concept_dict_size(x['concept']), reverse=True)
		other_concept_dict_list = sorted(other_concept_dict_list, key=lambda x: CE.get_concept_dict_size(x['concept']), reverse=True)
		predicate_dict_span_set = set(predicate_dict['span'])
		concept_dict_iter = filter(lambda x: not predicate_dict_span_set.intersection(x['concept']['span']), concept_dict_list)
		other_concept_dict_iter = filter(lambda x: not predicate_dict_span_set.intersection(x['concept']['span']), other_concept_dict_list)
		
		if subj_core == core:
			subj_dict = next(concept_dict_iter,None)
			obj_dict = next(other_concept_dict_iter,None)
		else:
			obj_dict = next(concept_dict_iter,None)
			subj_dict = next(other_concept_dict_iter,None)

		if not subj_dict or not obj_dict:
			return None

		subj_span, obj_span = list(subj_dict['concept']['span']), list(obj_dict['concept']['span'])
		# predicate_dict['source_text'] = get_concept_dict(sorted(predicate_span + subj_span + obj_span, key=lambda x: x.idx))['text']
		predicate_dict['source_text'] = get_concept_dict(sorted(predicate_span + [subj_core, obj_core], key=lambda x: x.idx))['text']
		predicate_dict['predicate_core'] = get_concept_dict(predicate_core_span) # get predicate_core_dict
		predicate_dict['source'] = subj_dict['source']
		# get verb
		if verb_span:
			verb_dict = get_concept_dict(verb_span)
			verb_text, verb_lemma = CoupleExtractor.templatize_predicate_span(verb_span, subj_core, obj_core)
			verb_dict['text'] = ' '.join(verb_text)
			verb_dict['lemma'] = ' '.join(verb_lemma)
			# verb_dict['source_text'] = get_concept_dict(sorted(verb_span + subj_span + obj_span, key=lambda x: x.idx))['text']
			verb_dict['source_text'] = get_concept_dict(sorted(verb_span + [subj_core, obj_core], key=lambda x: x.idx))['text']
		else:
			verb_dict = None
		predicate_dict['verb'] = verb_dict
		
		# #### free memory
		# CE.clean_concepts_from_tokens(subj_dict, remove_source=True)
		# CE.clean_concepts_from_tokens(obj_dict, remove_source=True)
		# ################
		return (subj_dict, predicate_dict, obj_dict)

	@staticmethod
	def get_triplet_list_by_concept_list(concept_list, avoid_jumps=False):
		ModelManager.logger.debug('Starting get_triplet_dict_list')
		core_concept_dict = {}
		for concept in concept_list:
			core = concept['concept_core'][-1]['span'][0]
			if core not in core_concept_dict:
				core_concept_dict[core] = []
			core_concept_dict[core].append(concept)
		ModelManager.logger.debug(f'	core_concept_dict has {len(core_concept_dict)} elements')

		core_concept_list = tuple(core_concept_dict.items())
		core_set = set(core_concept_dict.keys())
		# find the paths that connect the core concepts each other
		triplet_iter = (
			CoupleExtractor.get_triplet(core_concept, other_core_concept, core_set, avoid_jumps=avoid_jumps)
			for i,core_concept in enumerate(core_concept_list)
			for other_core_concept in core_concept_list[i+1:]
		)
		triplet_iter = filter(lambda x: x, triplet_iter)
		triplet_iter = tuple(triplet_iter)
		for s,p,o in triplet_iter:
			if 'source' in s:
				del s['source']
			if 'source' in o:
				del o['source']
		return triplet_iter

	@staticmethod
	def get_couple_list_by_concept_list(concept_list, avoid_jumps=False):
		ModelManager.logger.debug('Starting get_couple_list_by_concept_list')
		
		triplet_list = CoupleExtractor.get_triplet_list_by_concept_list(concept_list, avoid_jumps=avoid_jumps)
		ModelManager.logger.debug(f'	triplet_list has {len(triplet_list)} elements')
		couple_list = []
		for subj_dict, predicate_dict, obj_dict in triplet_list:
			# subj_dict['is_at_core'] = CoupleExtractor.is_at_core(subj_dict)
			subj_dict['dependency'] = 'subj'
			subj_dict.update(predicate_dict)
			couple_list.append(subj_dict)

			# obj_dict['is_at_core'] = CoupleExtractor.is_at_core(obj_dict)
			obj_dict['dependency'] = 'obj'
			obj_dict.update(predicate_dict)
			couple_list.append(obj_dict)

		# print([(c['dependency'],c['concept']['text'],c['predicate']['text']) for c in couple_list])
		ModelManager.logger.debug('Ending get_couple_list_by_concept_list')
		return couple_list

	def item_list_extraction_handler(self, get_item_fn, doc_parser: DocParser, parallel_extraction=True):
		doc_iter = doc_parser.get_doc_iter()
		annotation_iter = doc_parser.get_annotation_iter()
		content_iter = self.nlp(doc_parser.get_content_iter())
		doc_content_annotation_iter = list(zip(doc_iter, content_iter, annotation_iter))
		del content_iter
		total = len(doc_content_annotation_iter)
		self.logger.debug(f'parallelised_item_list_extraction: Extracting item list from corpus with {get_item_fn}..')
		if not parallel_extraction:
		# if True:
			return flatten(
				map(
					lambda x: get_item_fn(self.get_concept_list_by_doc(*x)), 
					self.tqdm(doc_content_annotation_iter, total=total)
				), as_list=True
			)
		# model_options = self.model_options
		def get_couple_list_by_doc(chunk):
			# model_manager = ModelManager(model_options)
			couple_list = flatten((
				get_item_fn(CE.get_concept_list_by_doc(doc,content,annotation))
				for doc,content,annotation in self.tqdm(chunk)
			), as_list=True)
			return couple_list

		processes = min(self.n_threads, total)
		pool = Pool(nodes=processes)

		chunks = tuple(get_chunks(doc_content_annotation_iter, number_of_chunks=processes))
		assert len(doc_content_annotation_iter) == sum(map(len, chunks))
		del doc_content_annotation_iter

		self.logger.debug('get_couple_list: Processing chunks..')
		partial_solutions = self.tqdm(pool.imap(get_couple_list_by_doc, chunks), total=len(chunks))
		pool.close()
		pool.join()
		pool.clear()

		self.logger.info('get_couple_list: Flattening solutions..')
		return flatten(partial_solutions, as_list=True)

	def get_couple_list(self, doc_parser: DocParser, avoid_jumps=False, parallel_extraction=True, remove_predicate_cores=True, remove_source_paragraph=False, remove_idx=True, remove_span=True):
		def get_item_fn(x):
			return CoupleExtractor.clean_couples_from_tokens(
				CoupleExtractor.get_couple_list_by_concept_list(x, avoid_jumps=avoid_jumps),
				remove_predicate_cores=remove_predicate_cores, # minimise memory usage
				remove_source_paragraph=remove_source_paragraph, # minimise memory usage
				remove_idx=remove_idx, # minimise memory usage
				remove_span=remove_span, # minimise memory usage
			)
		return self.item_list_extraction_handler(
			get_item_fn,
			doc_parser, 
			parallel_extraction=parallel_extraction,
		)

	@staticmethod
	def clean_couples_from_tokens(couple_iter, remove_predicate_cores=False, remove_source_paragraph=False, remove_idx=False, remove_span=False):
		for couple in couple_iter:
			couple['predicate'] = CE.clean_concept_dict_from_tokens(couple['predicate'], remove_idx=remove_idx, remove_span=remove_span)
			if 'predicate_core' in couple:
				if remove_predicate_cores:
					del couple['predicate_core']
				else:
					couple['predicate_core'] = CE.clean_concept_dict_from_tokens(couple['predicate_core'], remove_idx=remove_idx, remove_span=remove_span)
			if 'verb' in couple:
				couple['verb'] = CE.clean_concept_dict_from_tokens(couple['verb'], remove_idx=remove_idx, remove_span=remove_span)
			tuple(CE.clean_concepts_from_tokens([couple], remove_source_paragraph=remove_source_paragraph, remove_idx=remove_idx, remove_span=remove_span))
			yield couple

	def get_triplet_list(self, doc_parser: DocParser, avoid_jumps=False, parallel_extraction=True, remove_predicate_cores=True, remove_source_paragraph=False, remove_idx=True, remove_span=True):
		def get_item_fn(x):
			return CoupleExtractor.clean_triplets_from_tokens(
				CoupleExtractor.get_triplet_list_by_concept_list(x, avoid_jumps=avoid_jumps),
				remove_predicate_cores=remove_predicate_cores, # minimise memory usage
				remove_source_paragraph=remove_source_paragraph, # minimise memory usage
				remove_idx=remove_idx, # minimise memory usage
				remove_span=remove_span, # minimise memory usage
			)
		return self.item_list_extraction_handler(
			get_item_fn,
			doc_parser, 
			parallel_extraction=parallel_extraction,
		)

	@staticmethod
	def clean_triplets_from_tokens(triplet_iter, remove_predicate_cores=False, remove_source_paragraph=False, remove_idx=False, remove_span=False):
		for subj_dict, predicate_dict, obj_dict in triplet_iter:
			predicate_dict = CE.clean_concept_dict_from_tokens(predicate_dict, remove_idx=remove_idx, remove_span=remove_span)
			if 'predicate_core' in predicate_dict:
				if remove_predicate_cores:
					del predicate_dict['predicate_core']
				else:
					predicate_dict['predicate_core'] = CE.clean_concept_dict_from_tokens(predicate_dict['predicate_core'], remove_idx=remove_idx, remove_span=remove_span)
			if 'verb' in predicate_dict:
				predicate_dict['verb'] = CE.clean_concept_dict_from_tokens(predicate_dict['verb'], remove_idx=remove_idx, remove_span=remove_span)
			tuple(CE.clean_concepts_from_tokens([subj_dict,obj_dict], remove_source_paragraph=remove_source_paragraph, remove_idx=remove_idx, remove_span=remove_span))
			yield (subj_dict, predicate_dict, obj_dict)

def get_validated_sentence_list(self: ModelManager, sentence_list, avoid_coreferencing=False):
	validated_sentence_list = []
	# sentence_list = [ # faster here than in the for-loop
	# 	s if not s.strip('.').endswith(':') else ''
	# 	for s in sentence_list
	# ]
	for sent_span, sent in zip(self.nlp(sentence_list), sentence_list):
		if not sent:
			validated_sentence_list.append(None)
		# elif sent.strip('.').endswith(':'):
		# 	validated_sentence_list.append(None)
		elif not CoupleExtractor.is_verbal(sent_span):
			validated_sentence_list.append(None)
		elif not CoupleExtractor.has_subject(sent_span):
			validated_sentence_list.append(None)
		elif avoid_coreferencing and CoupleExtractor.is_coreferencing(sent_span):
			validated_sentence_list.append(None)
		else:
			validated_sentence_list.append(sent)
	return validated_sentence_list

def filter_invalid_sentences(self: ModelManager, obj_list, key=lambda x:x, avoid_coreferencing=False):
	sentence_list = list(map(key, obj_list))
	validated_sentence_list = get_validated_sentence_list(self, sentence_list, avoid_coreferencing=avoid_coreferencing)
	# print(0, json.dumps(sentence_list, indent=4))
	# print(1, json.dumps(validated_sentence_list, indent=4))
	return [
		obj
		for obj, is_valid in zip(obj_list,validated_sentence_list)
		if is_valid
	]
