from doxpy.misc.doc_reader import DocParser
from doxpy.misc.jsonld_lib import *
from doxpy.models.model_manager import ModelManager
from doxpy.misc.utils import *
from collections import Counter, deque
import re
from more_itertools import unique_everseen
import logging
# from pathos.threading import ThreadPool as Pool
from pathos.multiprocessing import ProcessPool as Pool
import string


### https://spacy.io/api/annotation
# 'acl',		# clausal modifier of noun (adjectival clause)
# 'acomp',		# adjectival complement
# 'advcl',		# adverbial clause modifier
# 'advmod',		# adverbial modifier
# 'agent',		# agent
# 'amod',		# adjectival modifier
# 'appos',		# appositional modifier
# 'attr',		# attribute
# 'aux',		# auxiliary
# 'auxpass',	# auxiliary (passive)
# 'case',		# case marking
# 'cc',			# coordinating conjunction
# 'ccomp',		# clausal complement
# 'compound',	# compound
# 'conj',		# conjunct
# 'cop',		# copula
# 'csubj',		# clausal subject
# 'csubjpass',	# clausal subject (passive)
# 'dative',		# dative
# 'dep',		# unclassified dependent
# 'det',		# determiner
# 'dobj',		# direct object
# 'expl',		# expletive
# 'intj',		# interjection
# 'mark',		# marker
# 'meta',		# meta modifier
# 'neg',		# negation modifier
# 'nn',			# noun compound modifier
# 'nounmod',	# modifier of nominal
# 'npmod',		# noun phrase as adverbial modifier
# 'nsubj',		# nominal subject
# 'nsubjpass',	# nominal subject (passive)
# 'nummod',		# numeric modifier
# 'oprd',		# object predicate
# 'obj',		# object
# 'obl',		# oblique nominal
# 'parataxis',	# parataxis
# 'pcomp',		# complement of preposition
# 'pobj',		# object of preposition
# 'poss',		# possession modifier
# 'preconj',	# pre-correlative conjunction
# 'prep',		# prepositional modifier
# 'prt',		# particle
# 'punct',		# punctuation
# 'quantmod',	# modifier of quantifier
# 'relcl',		# relative clause modifier
# 'root',		# root
# 'xcomp',		# open clausal complement

class ConceptExtractor(ModelManager):

	SUBJ_IDENTIFIER = [ # dependency markers for subjects
		'csubj',		# clausal subject
		'csubjpass',	# clausal subject (passive)
		'nsubj',		# nominal subject
		'nsubjpass',	# nominal subject (passive)
		'expl',			# expletive
	]
	OBJ_IDENTIFIER = [ # dependency markers for objects
		'oprd',		# object predicate
		'obj',		# object
		'dobj',		# direct object
		'pobj',		# object of preposition
		'obl',		# oblique nominal
		# 'intj',		# interjection
		'attr',		# attribute - http://www.english-for-students.com/an-attribute-1.html
		# # Verbal objects
		'acomp',	# adjectival complement
		'xcomp',	# open clausal complement
		# 'nounmod', 'nmod',	# modifier of nominal
	]
	CONCEPT_IDENTIFIER = SUBJ_IDENTIFIER + OBJ_IDENTIFIER
	AMOD_IDENTIFIER = [
		'amod',		# adjectival modifier
	]
	# do not extend chunk with 'det' or 'poss', it is noise causing issues with determiners like 'no', etc. Better to add determiners and possession modifiers to predicates instead
	EXTENDED_CHUNK_IDENTIFIER = [ # https://universaldependencies.org/u/dep/all.html
		# 'det',	# determiner
		'punct',	# punctuation
		# Multiword Expression
		'fixed', 'flat', 'compound',	# compound
		'subtok', 	# eg. the 'non' and the '-' in 'non-contractual obligations'
		'case',		# case marking
		'dep',		# unclassified dependent
		# Normal Modifiers
		'advmod',	# adverbial modifier
		'amod',		# adjectival modifier
		'nounmod', 'nmod',	# modifier of nominal
		'npmod',	# noun phrase as adverbial modifier
		'nummod',	# numeric modifier
		'quantmod',	# modifier of quantifier
		'nn',		# noun compound modifier
		'meta',		# meta modifier
		'neg',		# negation modifier
		# 'poss',	# possession modifier
		'appos',	# appositional modifier
	]
	PREP_IDENTIFIER = [
		'prep',		# prepositional modifier
	]
	# CLAUSE_IDENTIFIER = [
	# 	# Clausal Modifiers (noun + verb)
	# 	'acl',		# clausal modifier of noun (adjectival clause)
	# 	'relcl',	# relative clause modifier
	# 	'advcl',	# adverbial clause modifier
	# 	# Verbal objects
	# 	'mark', 	# marker - https://universaldependencies.org/docs/en/dep/mark.html
	# 	'prt',		# particle
	# 	'agent',	# agent
	# 	'dative',	# dative
	# 	'advmod',	# adverbial modifier
	# 	'aux',		# auxiliaries
	# 	# Complements
	# 	# 'acomp',	# adjectival complement
	# 	# 'xcomp',	# open clausal complement
	# 	# 'pcomp',	# complement of preposition
	# 	# 'ccomp',	# clausal complement
	# ]
	SUBJ_REGEXP = re.compile('|'.join(SUBJ_IDENTIFIER))
	OBJ_REGEXP = re.compile('|'.join(OBJ_IDENTIFIER))
	CONCEPT_REGEXP = re.compile('|'.join(CONCEPT_IDENTIFIER))
	AMOD_REGEXP = re.compile('|'.join(AMOD_IDENTIFIER))
	EXTENDED_CHUNK_REGEXP = re.compile('|'.join(EXTENDED_CHUNK_IDENTIFIER))
	GROUP_REGEXP = re.compile('|'.join(CONCEPT_IDENTIFIER+PREP_IDENTIFIER+EXTENDED_CHUNK_IDENTIFIER))
	# CLAUSE_REGEXP = re.compile('|'.join(CONCEPT_IDENTIFIER+PREP_IDENTIFIER+EXTENDED_CHUNK_IDENTIFIER+CLAUSE_IDENTIFIER))
	
	def __init__(self, model_options):
		super().__init__(model_options)
		self.min_sentence_token_count = model_options.get('min_sentence_token_count',3)
		self.min_sentence_whitespace_count = self.min_sentence_token_count-1
		self.disable_spacy_component = ["ner","textcat"]

	@staticmethod
	def get_referenced_span(token):
		if token.pos_ == 'PRON' and token._.in_coref:
			#for cluster in token._.coref_clusters:
			#	print(token.text + " => " + cluster.main.text)
			return ConceptExtractor.trim_prepositions(list(token._.coref_clusters[0].main))
		# return [token]

	@staticmethod
	def get_token_lemma(token, prevent_verb_lemmatization=False):
		return (token.lemma_ if not prevent_verb_lemmatization or token.pos_!='VERB' else token.text).casefold()

	@staticmethod
	def get_span_lemma(span, prevent_verb_lemmatization=False, hidden_dep_list=None):
		span = filter(lambda e: e.pos_ != 'DET', span) # no determiners
		span = filter(lambda e: e.pos_ != 'PUNCT', span) # no punctuation
		span = filter(lambda e: e.pos_ != 'CCONJ', span) # no conjunctions
		span = filter(lambda e: e.pos_ != 'PRON', span) # no pronouns
		# span = filter(lambda e: e.text not in string.punctuation)
		if hidden_dep_list:
			span = filter(lambda e: ConceptExtractor.get_token_dependency(e) not in hidden_dep_list, span)
		lemma_iter = (
			ConceptExtractor.get_token_lemma(e, prevent_verb_lemmatization=prevent_verb_lemmatization)
			for e in span 
		)
		span_lemma = ' '.join(lemma_iter)#.strip()
		span_lemma = span_lemma.translate(str.maketrans('', '', string.punctuation)) # enforce no punctuation regardless the errors of the dependency parser
		span_lemma = span_lemma.strip()
		return span_lemma

	@staticmethod
	def get_span_text(concept):
		return add_missing_brackets_to_string(
			' '.join(
				# ' '.join(
				# 	t.text 
				# 	for t in ConceptExtractor.get_referenced_span(c)
				# ).strip() 
				c.text 
				for c in concept
			)#.strip()#.replace(' - ','') # replace subtokens
		) 

	@staticmethod
	def trim(token_list, trim_fn):
		a = 0
		while len(token_list)+(a-1) >= 0 and trim_fn(token_list[a-1]):
			a -= 1
			# del token_list[-1]
		b = 0
		while len(token_list)+a-(b+1) >= 0 and trim_fn(token_list[b]):
			b += 1
			# del token_list[0]
		if not a and not b:
			return token_list
		return token_list[b:a] if a else token_list[b:]

	@staticmethod
	def trim_prepositions(token_list):
		# punct_to_remove = set([',','.',';','"',"'"])
		def trim_fn(x):
			dep = ConceptExtractor.get_token_dependency(x)
			return dep == 'prep' or dep == 'punct'
		return ConceptExtractor.trim(token_list, trim_fn)

	@staticmethod
	def get_token_dependency(token):
		if token.dep_ != 'conj':
			return token.dep_
		for t in token.ancestors:
			if t.dep_ != 'conj':
				return t.dep_

	@staticmethod
	def get_token_ancestors(token):
		conjunction_count = 1 if token.dep_ == 'conj' else 0
		for ancestor in token.ancestors:
			if ancestor.dep_ == 'conj':
				conjunction_count += 1
				if conjunction_count == 1:
					yield ancestor
				else:
					continue
			elif conjunction_count > 0:
				conjunction_count = 0
				continue
			yield ancestor

	@staticmethod
	def get_token_descendants(token, filter_fn=lambda x:x):
		children_to_check = deque(token.children);
		seen_tokens = set(token.children)
		while len(children_to_check) > 0:
			c = children_to_check.pop()
			if filter_fn(c):
				valid_children = tuple(filter(lambda x: x not in seen_tokens, c.children))
				children_to_check.extend(valid_children)
				seen_tokens.update(valid_children)
				# del valid_children
				yield c
		# del children_to_check
		# del seen_tokens

	@staticmethod
	def get_consecutive_tokens(core_concept, concept_span):
		return tuple((
			t
			for i,t in enumerate(concept_span)
			if abs(t.i-core_concept.i) == i
		))

	@staticmethod
	def get_composite_concept(core_concept, dep_regexp=None, fellow_filter_fn=lambda x: x.dep_ != 'conj'):
		if dep_regexp:
			filter_fn = lambda x: fellow_filter_fn(x) and re.match(dep_regexp, ConceptExtractor.get_token_dependency(x))
		else:
			filter_fn = fellow_filter_fn
		concept_span = [core_concept]
		concept_span += ConceptExtractor.get_token_descendants(core_concept, filter_fn)
		concept_span = sorted(
			unique_everseen(
				concept_span, 
				key=lambda x: x.idx
			), 
			key=lambda x: x.idx
		)
		# concept_span = ConceptExtractor.get_consecutive_tokens(core_concept, concept_span)
		# concept_span = ConceptExtractor.trim_prepositions(concept_span)
		return tuple(concept_span)

	@staticmethod
	def get_concept_dict_from_span(span, prevent_verb_lemmatization=False, hidden_dep_list=None):
		# print(list(map(lambda x: (x.text, x.pos_, x.dep_), span)))
		return {
			'span': tuple(span),
			'text': ConceptExtractor.get_span_text(span),
			'lemma': ConceptExtractor.get_span_lemma(span, prevent_verb_lemmatization, hidden_dep_list).lower(),
			# 'idx': tuple((s.idx,s.idx+len(s)) for s in span),
			'idx': next((s.idx for s in span), -1),
		}

	@staticmethod
	def clean_concept_dict_from_tokens(concept_dict, remove_idx=False, remove_span=False):
		if concept_dict and 'span' in concept_dict:
			if not remove_span:
				concept_dict['span'] = tuple((
					x.text if not isinstance(x, str) else x 
					for x in concept_dict['span']
				))
			else:
				del concept_dict['span']
		if remove_idx:
			if concept_dict and 'idx' in concept_dict:
				del concept_dict['idx']
		return concept_dict

	@staticmethod
	def clean_concepts_from_tokens(concepts_iter, remove_source_paragraph=False, remove_idx=False, remove_span=False):
		for concept in concepts_iter:
			if remove_source_paragraph and 'paragraph_text' in concept['source']:
				del concept['source']['paragraph_text']
			# if 'concept' in concept:
			ConceptExtractor.clean_concept_dict_from_tokens(concept['concept'], remove_idx=remove_idx, remove_span=remove_span)
			# if 'concept_core' in concept:
			for c in concept['concept_core']:
				ConceptExtractor.clean_concept_dict_from_tokens(c, remove_idx=remove_idx, remove_span=remove_span)
			yield concept

	@staticmethod
	def get_concept_dict_uid(concept_dict):
		return (concept_dict['text'], concept_dict.get('idx',None))

	@staticmethod
	def get_source_dict_uid(source_dict):
		return (source_dict['sentence_text'], source_dict['doc'])

	@staticmethod
	def get_concept_dict_size(concept_dict):
		concept_span = concept_dict.get('span', None)
		concept_text = concept_dict['text']
		return (
			len(concept_span) if concept_span else concept_text.count(' ')+1,
			len(concept_text)
		)

	@staticmethod
	def get_related_concept_iter(token):
		if not token:
			return []
		# Get composite concepts
		core_concept = (token,)
		concept_chunk = tuple(next(filter(lambda nc: token in nc, token.sent.noun_chunks), core_concept))
		concept_dict_iter = map(
			ConceptExtractor.get_concept_dict_from_span, 
			unique_everseen(
				map(
					ConceptExtractor.trim_prepositions,
					(
						core_concept, 
						ConceptExtractor.get_composite_concept(token, ConceptExtractor.AMOD_REGEXP), 
						concept_chunk, 
						ConceptExtractor.get_composite_concept(token, ConceptExtractor.EXTENDED_CHUNK_REGEXP), 
						ConceptExtractor.get_composite_concept(token, ConceptExtractor.GROUP_REGEXP),
						ConceptExtractor.get_composite_concept(token, None, lambda x: x), # get the whole subtree
					)
				),
				key=lambda x: tuple(map(lambda y: y.idx, x))
			)
		)
		concept_dict_iter = filter(lambda x: x['lemma'], concept_dict_iter)
		concept_dict_list = sorted(concept_dict_iter, key=ConceptExtractor.get_concept_dict_size) # tokens' length + whitespaces
		# Build concept iter
		return (
			{
				'source': { # Get sentece
					'sentence_text': token.sent.text,
					'paragraph_text': token.doc.text,
					# 'sent_idx': token.sent[0].idx, # the position of the sentence inside the documents
				},
				'concept': concept_dict, 
				'concept_core': tuple(reversed(concept_dict_list[:i]) if i > 1 else concept_dict_list[:1])
			}
			for i,concept_dict in enumerate(concept_dict_list)
		)

	@staticmethod
	def is_core_concept(token):
		dep = ConceptExtractor.get_token_dependency(token)
		return (dep =='ROOT' and token.pos_=='NOUN') or re.match(ConceptExtractor.CONCEPT_REGEXP, dep)

	@staticmethod
	def get_concept_list_by_doc(doc_id, processed_doc, annotation_dict):
		ModelManager.logger.debug('Starting get_concept_list_by_doc')
		# print(0)
		if not processed_doc:
			# print(1, processed_doc)
			return []
		# print(2)
		core_concept_iter = filter(ConceptExtractor.is_core_concept, processed_doc)
		# print(3, core_concept_iter)
		concept_iter = flatten(map(ConceptExtractor.get_related_concept_iter, core_concept_iter))
		# print(4, concept_iter)
		concept_iter = unique_everseen(concept_iter, key=lambda c: ConceptExtractor.get_concept_dict_uid(c['concept']))
		concept_list = list(concept_iter)
		# print(5, concept_list)
		for concept_dict in concept_list:
			concept_dict['source']['doc'] = doc_id
			concept_dict['source']['annotation'] = annotation_dict
		ModelManager.logger.debug('Ending get_concept_list_by_doc')
		return concept_list

	def get_concept_list(self, doc_parser: DocParser, parallel_extraction=True, remove_source_paragraph=False, remove_idx=True, remove_span=False):
		self.logger.info(f'Extracting concept list from corpus with parallel_extraction={parallel_extraction}..')
		doc_iter = doc_parser.get_doc_iter()
		# print('aa', list(doc_parser.get_doc_iter()))
		annotation_iter = doc_parser.get_annotation_iter()
		# print('bb', list(doc_parser.get_annotation_iter()))
		content_iter = self.nlp(doc_parser.get_content_iter())
		# print('cc', len(content_iter), content_iter)
		doc_content_annotation_iter = tuple(zip(doc_iter, content_iter, annotation_iter))
		# print('dd', len(doc_content_annotation_iter))
		del content_iter
		if not parallel_extraction:
			return flatten(
				map(
					lambda x: ConceptExtractor.get_concept_list_by_doc(*x), 
					self.tqdm(doc_content_annotation_iter)
				), 
				as_list=True
			)

		def get_concept_list_by_doc(chunk):
			# model_manager = ModelManager(model_options)
			return flatten((
				ConceptExtractor.clean_concepts_from_tokens(
					ConceptExtractor.get_concept_list_by_doc(doc,content,annotation),
					remove_source_paragraph=remove_source_paragraph, # minimise memory usage
					remove_idx=remove_idx, # minimise memory usage
					remove_span=remove_span, # minimise memory usage
				)
				for doc,content,annotation in self.tqdm(chunk)
			), as_list=True)

		# processes = min(math.ceil(self.n_threads/2), len(doc_content_annotation_iter))
		processes = min(self.n_threads, len(doc_content_annotation_iter))
		pool = Pool(nodes=processes)

		chunks = tuple(get_chunks(doc_content_annotation_iter, number_of_chunks=processes))
		assert len(doc_content_annotation_iter) == sum(map(len, chunks))
		del doc_content_annotation_iter

		partial_solutions = self.tqdm(pool.imap(get_concept_list_by_doc, chunks), total=len(chunks))
		pool.close()
		pool.join() 
		pool.clear()

		return flatten(partial_solutions, as_list=True)

	@staticmethod
	def find_path_to_closest_in_set(core, core_set):
		path = set()
		if core in core_set:
			return (path,core)
		for ancestor in ConceptExtractor.get_token_ancestors(core):
			if ancestor in core_set:
				return (path,ancestor)
			path.add(ancestor)
		return (path,None)

	@staticmethod
	def get_concept_counter_dict(concept_list):
		return {
			concept: {'count': count}
			for concept, count in Counter(concept_list).items()
		}
