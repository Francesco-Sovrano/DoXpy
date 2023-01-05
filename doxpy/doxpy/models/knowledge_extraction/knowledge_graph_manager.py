from doxpy.models.model_manager import ModelManager
from doxpy.misc.adjacency_list import AdjacencyList
from doxpy.models.knowledge_extraction.couple_extractor import CoupleExtractor
from doxpy.misc.graph_builder import get_concept_description_dict
from doxpy.misc.jsonld_lib import *
from doxpy.misc.utils import *

from doxpy.misc.graph_builder import get_concept_set, get_predicate_set, get_object_set, get_ancestors, filter_graph_by_root_set, tuplefy

try:
	from nltk.corpus import wordnet as wn
	from nltk.corpus import brown
	from nltk.corpus import stopwords
except OSError:
	print('Downloading nltk::wordnet\n'
		"(don't worry, this will only happen once)")
	import nltk
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('wordnet')
	nltk.download('brown')
	nltk.download('stopwords')
	from nltk.corpus import stopwords
	from nltk.corpus import wordnet as wn
	from nltk.corpus import brown
from nltk import FreqDist

import unidecode
singlefy = lambda s: unidecode.unidecode(s.strip().replace("\n"," "))#.capitalize()
# singlefy = lambda s: s.strip().replace("\n"," ")#.capitalize()

word_frequency_distribution = FreqDist(i.lower() for i in brown.words())
is_common_word = lambda w: word_frequency_distribution.freq(w) >= 1e-4

class KnowledgeGraphManager(ModelManager):
	def __init__(self, model_options, graph):
		super().__init__(model_options)
		ModelManager.logger.info('Initialising KnowledgeGraphManager..')
		# self.graph = graph

		# self.adjacency_list = AdjacencyList(
		# 	graph, 
		# 	equivalence_relation_set=set([IN_SYNSET_PREDICATE,IS_EQUIVALENT_PREDICATE]),
		# 	is_sorted=True,
		# )
		self.min_triplet_len = model_options.get('min_triplet_len',0)
		self.max_triplet_len = model_options.get('max_triplet_len',float('inf'))
		self.min_sentence_len = model_options.get('min_sentence_len',0)
		self.max_sentence_len = model_options.get('max_sentence_len',float('inf'))
		self.min_paragraph_len = model_options.get('min_paragraph_len',0)
		self.max_paragraph_len = model_options.get('max_paragraph_len',float('inf'))
		self.adjacency_list = AdjacencyList(
			graph, 
			equivalence_relation_set=set([IS_EQUIVALENT_PREDICATE]),
			is_sorted=False,
		)
		self._content_dict = None
		self._source_dict = None
		self._label_dict = None

		self._source_span_dict = None
		self._source_sentence_dict = None
		self._source_label_dict = None
		# self.verb_dict = self.adjacency_list.get_predicate_dict(HAS_VERB_PREDICATE)

		ModelManager.logger.info('KnowledgeGraphManager initialised!')

	@property
	def content_dict(self):
		if not self._content_dict:
			self.logger.info("Building content_dict..")
			self._content_dict = self.adjacency_list.get_predicate_dict(HAS_CONTENT_PREDICATE, singlefy)
			self.logger.info("content_dict built")
		return self._content_dict

	@property
	def source_dict(self):
		if not self._source_dict:
			self.logger.info("Building source_dict..")
			self._source_dict = self.adjacency_list.get_predicate_dict(HAS_PARAGRAPH_ID_PREDICATE)
			self.logger.info("source_dict built")
		return self._source_dict

	@property
	def label_dict(self):
		if not self._label_dict:
			self.logger.info("Building label_dict..")
			self._label_dict = self.adjacency_list.get_predicate_dict(HAS_LABEL_PREDICATE, singlefy)
			self.logger.info("label_dict built")
		return self._label_dict

	@property
	def source_span_dict(self):
		if not self._source_span_dict:
			self.logger.info("Building source_span_dict..")
			self._source_span_dict = self.adjacency_list.get_predicate_dict(HAS_SPAN_ID_PREDICATE)
			self.logger.info("source_span_dict built")
		return self._source_span_dict

	@property
	def source_sentence_dict(self):
		if not self._source_sentence_dict:
			self.logger.info("Building source_sentence_dict..")
			self._source_sentence_dict = self.adjacency_list.get_predicate_dict(HAS_SOURCE_ID_PREDICATE)
			self.logger.info("source_sentence_dict built")
		return self._source_sentence_dict

	@property
	def source_label_dict(self):
		if not self._source_label_dict:
			self.logger.info("Building source_label_dict..")
			self._source_label_dict = self.adjacency_list.get_predicate_dict(HAS_SOURCE_LABEL_PREDICATE, singlefy)
			self.logger.info("source_label_dict built")
		return self._source_label_dict	

	# @staticmethod
	# def build_from_edus_n_clauses(model_options, graph=None, kg_builder_options=None, qa_dict_list=None, qa_extractor_options=None, qa_type_to_use=None, use_only_elementary_discourse_units=False, edu_graph=None):
	# 	ModelManager.logger.info('build_from_edus_n_clauses')
	# 	if edu_graph is None:
	# 		assert qa_extractor_options is not None, 'if no edu_graph is passed, then qa_extractor_options must not be None'
	# 		assert kg_builder_options is not None, 'if no edu_graph is passed, then kg_builder_options must not be None'
	# 		assert graph is not None, 'if no edu_graph is passed, then graph must not be None'
	# 		edu_graph = QuestionAnswerExtractor(qa_extractor_options).extract_aligned_graph_from_qa_dict_list(graph, kg_builder_options, qa_dict_list=qa_dict_list, qa_type_to_use=qa_type_to_use)
	# 	assert graph, 'graph is missing'
	# 	# Remove invalid labels: All the valid labels of EDU-graph are included in the original graph, except for the extra ones coming from questions (i.e., templates). Thus, removing all the labels from EDU-graph will prevent to consider invalid labels as important aspects
	# 	if use_only_elementary_discourse_units:
	# 		edu_graph_label_set = get_subject_set(filter(lambda x: x[1]==HAS_LABEL_PREDICATE, edu_graph))
	# 		graph_label_set = get_subject_set(filter(lambda x: x[1]==HAS_LABEL_PREDICATE, graph))
	# 		valid_label_set = edu_graph_label_set.intersection(graph_label_set)
	# 		edu_graph = list(filter(lambda x: x[1]!=HAS_LABEL_PREDICATE or x[0] in valid_label_set, edu_graph))
	# 		# paragraph_id_graph = list(filter(lambda x: x[1]==HAS_PARAGRAPH_ID_PREDICATE, graph))
	# 		# edu_graph += paragraph_id_graph
	# 		# source_span_uri_set = get_subject_set(paragraph_id_graph)
	# 		# edu_graph += list(filter(lambda x: x[1]==HAS_SOURCE_LABEL_PREDICATE and x[0] in source_span_uri_set, graph))
	# 		# edu_graph += filter(lambda x: x[1]==HAS_CONTENT_PREDICATE, graph)
	# 	else: # Merge with graph
	# 		edu_graph = list(filter(lambda x: x[1]!=HAS_LABEL_PREDICATE, edu_graph))
	# 		edu_graph = list(unique_everseen(edu_graph + graph))
	# 	return KnowledgeGraphManager(model_options, edu_graph)

	@property
	def concept_description_dict(self):
		return {
			uri: list(unique_everseen(filter(lambda x: x.strip(), label_list), key=lambda x: x.lower()))
			for uri, label_list in self.label_dict.items()
			if '{obj}' not in uri # no predicates
			and not (uri.startswith(DOC_PREFIX) or uri.startswith(ANONYMOUS_PREFIX)) # no files or anonymous entities
		}

	@property
	def aspect_uri_list(self):
		return list(self.concept_description_dict.keys())

	def get_source_set(self, uri):
		return set((
			source_id
			for source_span_uri in self.source_span_dict.get(uri,[])
			for source_sentence_uri in self.source_sentence_dict[source_span_uri]
			for source_id in self.source_dict[source_sentence_uri]
		))

	def get_source_span_set(self, uri):
		return set(self.source_span_dict.get(uri,[]))

	def get_source_span_label_iter(self, uri):
		return (
			source_span_label
			for source_span_uri in self.source_span_dict.get(uri,[])
			for source_span_label in self.source_label_dict[source_span_uri]
		)

	def get_source_span_label_set(self, uri):
		return set(self.get_source_span_label_iter(uri))

	def get_source_span_label(self, uri):
		return next(iter(self.get_source_span_label_iter(uri)), None)

	def get_edge_source_span_label_set(self, s,p,o):
		return self.get_source_span_label_set(p).intersection(self.get_source_span_label_set(s)).intersection(self.get_source_span_label_set(o))

	def get_edge_source_span_label(self, s,p,o):
		edge_source_span_label_set = self.get_edge_source_span_label_set(s,p,o)
		if edge_source_span_label_set:
			return min(edge_source_span_label_set, key=len)
		return None

	def get_label_list(self, concept_uri, explode_if_none=True):
		label_list = []
		for c in self.get_equivalent_concepts(concept_uri):
			if c in self.label_dict:
				label_list += self.label_dict[c]
			elif c.startswith(WORDNET_PREFIX):
				label_list += list(map(lambda x: explode_concept_key(x.name()), wn.synset(c[len(WORDNET_PREFIX):]).lemmas()))
			else:
				label_list.append(explode_concept_key(concept_uri) if explode_if_none else '')
		return label_list

	def get_label(self, concept_uri, explode_if_none=True):
		label_list = self.get_label_list(concept_uri, explode_if_none)
		if concept_uri in self.label_dict:
			return min(label_list, key=len)
		return label_list[0]

	def is_uncommon_aspect(self, aspect_uri):
		return not self.is_common_aspect(aspect_uri)

	def is_common_aspect(self, aspect_uri):
		commonality_gen = (
			is_common_word(label) or label in stopwords.words('english')
			for label in map(lambda x: x.lower(), self.get_label_list(aspect_uri))
		)
		return next(filter(lambda x: x, commonality_gen), False)

	def is_relevant_aspect(self, aspect_uri, ignore_leaves=False):
		# if aspect_uri == CONCEPT_PREFIX:
		# 	return False
		# ignore commonsense aspects
		if self.is_common_aspect(aspect_uri):
			# print('is_common_aspect', aspect_uri)
			return False
		# concepts with less than 2 sources are leaves with no triplets: safely ignore them, they are included by a super-class that is necessarily explored before them and they are less relevant than it
		source_set = self.get_source_set(aspect_uri)
		if not source_set:
			# print('has_no_triplets', aspect_uri)
			return False
		if ignore_leaves:
			if len(source_set) <= 1:
				return False
		# concepts with the same sources of one of their sub-classes are redundant as those with no sources
		aspect_set = set([aspect_uri])
		subclass_set = self.get_sub_classes(aspect_set, depth=0) - aspect_set
		if not subclass_set:
			return True
		is_relevant = next(filter(lambda x: source_set-self.get_source_set(x), subclass_set), None) is not None
		# if not is_relevant:
		# 	print('is_redundant', aspect_uri)
		return is_relevant

	def get_sub_graph(self, uri, depth=None, predicate_filter_fn=lambda x: x != SUBCLASSOF_PREDICATE and '{obj}' not in x):
		uri_set = self.adjacency_list.get_predicate_chain(
			set([uri]), 
			direction_set=['out'], 
			depth=depth, 
			predicate_filter_fn=predicate_filter_fn
		)
		return list(unique_everseen((
			(s,p,o)
			for s in uri_set
			for p,o in self.adjacency_list.get_outcoming_edges_matrix(s)
		)))

	def get_equivalent_concepts(self, concept_uri):
		return list(self.adjacency_list.equivalence_matrix.get(concept_uri,[]))+[concept_uri]

	def get_sub_classes(self, concept_set, depth=None):
		return self.adjacency_list.get_predicate_chain(
			concept_set = concept_set, 
			predicate_filter_fn = lambda x: x == SUBCLASSOF_PREDICATE, 
			direction_set = ['in'],
			depth = depth,
		)

	def get_super_classes(self, concept_set, depth=None):
		return self.adjacency_list.get_predicate_chain(
			concept_set = concept_set, 
			predicate_filter_fn = lambda x: x == SUBCLASSOF_PREDICATE, 
			direction_set = ['out'],
			depth = depth,
		)

	def get_aspect_graph(self, concept_uri, add_external_definitions=False, include_super_concepts_graph=False, include_sub_concepts_graph=False, consider_incoming_relations=False, depth=None, filter_fn=lambda x: x):
		concept_set = set(self.get_equivalent_concepts(concept_uri))
		expanded_concept_set = set(concept_set)
		# Get sub-classes
		if include_sub_concepts_graph:
			expanded_concept_set |= self.get_sub_classes(concept_set, depth=depth)
		# Get super-classes
		if include_super_concepts_graph:
			expanded_concept_set |= self.get_super_classes(concept_set, depth=depth)
		# expanded_concept_set = sorted(expanded_concept_set) # this would improve caching, later on
		# Add outcoming relations to concept graph
		expanded_aspect_graph = [
			(s,p,o)
			for s in expanded_concept_set
			for p,o in self.adjacency_list.get_outcoming_edges_matrix(s)
		]
		# Add incoming relations to concept graph
		if consider_incoming_relations:
			expanded_aspect_graph += (
				(s,p,o)
				for o in expanded_concept_set
				for p,s in self.adjacency_list.get_incoming_edges_matrix(o)
			)
		# print(concept_uri, json.dumps(expanded_aspect_graph, indent=4))
		expanded_aspect_graph = list(filter(filter_fn, expanded_aspect_graph))
		# Add external definitions
		if add_external_definitions:
			# Add wordnet's definition
			for equivalent_concept_uri in filter(lambda x: x.startswith(WORDNET_PREFIX), self.adjacency_list.equivalence_matrix.get(concept_uri,[])):
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

	def get_paragraph_text(self, source_id):
		paragraph_text_list = self.content_dict.get(source_id, None) # check if any paragraph is available
		return paragraph_text_list[0] if paragraph_text_list else None

	def get_sourced_graph_from_aspect_graph(self, aspect_graph, **args):
		self.logger.info("Running get_sourced_graph_from_aspect_graph")

		def sourced_graph_with_rdf_triplets_gen(sub_graph):
			# Add full triplets
			for original_triplet in self.tqdm(sub_graph):
				# if not (self.min_triplet_len <= len(self.get_label(original_triplet[1], explode_if_none=False)) <= self.max_triplet_len):
				# 	continue
				# s,p,o = original_triplet
				for source_span_uri in self.source_span_dict.get(original_triplet,[]):
					triplet_text = self.source_label_dict[source_span_uri][0]
					if not (self.min_triplet_len <= len(triplet_text) <= self.max_triplet_len):
						continue
					for source_sentence_uri in self.source_sentence_dict[source_span_uri]:
						sentence_text_list = self.source_label_dict.get(source_sentence_uri,None)
						sentence_text = sentence_text_list[0] if sentence_text_list else None
						if sentence_text and self.min_sentence_len <= len(sentence_text) <= self.max_sentence_len:
							yield (
								triplet_text, # triplet text
								sentence_text, # sentence text
								original_triplet,
								(source_sentence_uri, self.source_dict[source_sentence_uri][0]),
							)
						for source_id in self.source_dict[source_sentence_uri]:
							paragraph_text = self.get_paragraph_text(source_id)
							if paragraph_text and paragraph_text != sentence_text and self.min_paragraph_len <= len(paragraph_text) <= self.max_paragraph_len:
								yield (
									triplet_text, # triplet text
									paragraph_text, # paragraph text
									original_triplet,
									(None, source_id),
								)
		# Add source to triples
		return list(unique_everseen(sourced_graph_with_rdf_triplets_gen(aspect_graph)))

	def get_sourced_graph(self):
		def sourced_graph_gen():
			source_span_uri_iter = unique_everseen((
				source_span_uri 
				for source_span_uri_set in self.source_span_dict.values()
				for source_span_uri in source_span_uri_set 
			))
			for source_span_uri in source_span_uri_iter:
				triplet_text = self.source_label_dict[source_span_uri][0]
				if not (self.min_triplet_len <= len(triplet_text) <= self.max_triplet_len):
					continue
				for source_sentence_uri in self.source_sentence_dict[source_span_uri]:
					sentence_text_list = self.source_label_dict.get(source_sentence_uri,None)
					sentence_text = sentence_text_list[0] if sentence_text_list else None
					if sentence_text and self.min_sentence_len <= len(sentence_text) <= self.max_sentence_len:
						yield (
							triplet_text, # triplet text
							sentence_text, # sentence text
							source_span_uri,
							(source_sentence_uri, self.source_dict[source_sentence_uri][0]),
						)
					for source_id in self.source_dict[source_sentence_uri]:
						paragraph_text = self.get_paragraph_text(source_id)
						if paragraph_text and paragraph_text != sentence_text and self.min_paragraph_len <= len(paragraph_text) <= self.max_paragraph_len:
							yield (
								triplet_text, # triplet text
								paragraph_text, # paragraph text
								source_span_uri,
								(None, source_id),
							)
		# Add source to triples
		return list(unique_everseen(sourced_graph_gen()))

	def get_taxonomical_view(self, concept_uri, depth=None, with_internal_definitions=True, concept_id_filter=None):
		if not concept_id_filter:
			concept_id_filter = lambda x: x
		concept_set = set((concept_uri,))
		if depth != 0:
			concept_set |= self.get_sub_classes(concept_set, depth=depth)
			concept_set |= self.get_super_classes(concept_set, depth=depth)
		concept_set = set(filter(concept_id_filter,concept_set))
		# Add subclassof relations
		taxonomical_view = set(
			(s,p,o)
			for s in concept_set
			for p,o in self.adjacency_list.get_outcoming_edges_matrix(s)
			if p == SUBCLASSOF_PREDICATE and concept_id_filter(o)
		).union(
			(s,p,o)
			for o in concept_set
			for p,s in self.adjacency_list.get_incoming_edges_matrix(o)
			if p == SUBCLASSOF_PREDICATE and concept_id_filter(s)
		)
		taxonomical_view = list(taxonomical_view)
		taxonomy_concept_set = get_concept_set(taxonomical_view).union(concept_set)
		# Add labels
		taxonomical_view += (
			(concept, HAS_LABEL_PREDICATE, self.get_label(concept, explode_if_none=False))
			for concept in taxonomy_concept_set
		)
		# # Add sources
		# taxonomical_view += (
		# 	(concept, HAS_PARAGRAPH_ID_PREDICATE, source)
		# 	for concept in taxonomy_concept_set
		# 	for source in self.source_dict.get(concept,())
		# )
		# for concept in taxonomy_concept_set:
		# 	for source in self.source_dict.get(concept,()):
		# 		taxonomical_view += self.get_sub_graph(source)
		# # Add wordnet definitions
		# taxonomical_view += (
		# 	(concept, HAS_DEFINITION_PREDICATE, wn.synset(concept[3:]).definition())
		# 	for concept in filter(lambda x: x.startswith(WORDNET_PREFIX), taxonomy_concept_set)
		# )
		# Add definitions
		if with_internal_definitions:
			taxonomical_view += unique_everseen(
				(concept_uri,p,o)
				for p,o in self.adjacency_list.get_outcoming_edges_matrix(concept_uri)
				if p == HAS_DEFINITION_PREDICATE
			)
		# Add types
		sub_types_set = self.adjacency_list.get_predicate_chain(
			concept_set = concept_set, 
			predicate_filter_fn = lambda x: x == HAS_TYPE_PREDICATE, 
			direction_set = ['out'],
			depth = 0,
		)
		super_types_set = self.adjacency_list.get_predicate_chain(
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

	def get_paragraph_id_from_concept_id(self, x):
		span_id = self.source_span_dict.get(x,[None])[0]
		if not span_id:
			return []
		sentence_id = self.source_sentence_dict.get(span_id,[None])[0]
		if not sentence_id:
			return []
		return self.source_dict[sentence_id][0]

	def get_sourced_graph_from_labeled_graph(self, label_graph):
		sourced_natural_language_triples_set = []

		def extract_sourced_graph(label_graph, str_fn):
			result = []
			for labeled_triple, original_triple in label_graph:
				naturalized_triple = str_fn(labeled_triple)
				# print(naturalized_triple, labeled_triple)
				if not naturalized_triple:
					continue
				context_set = self.get_paragraph_id_from_concept_id(original_triple)
				# print(s,p,o)
				# print(context_set)
				if not context_set:
					context_set = [None]
				result += (
					(
						naturalized_triple, 
						self.get_paragraph_text(source_id) if source_id else naturalized_triple, # source_label
						original_triple,
						(None, source_id if source_id else None), # source_id
					)
					for source_id in context_set
				)
			return result
		sourced_natural_language_triples_set += extract_sourced_graph(label_graph, get_string_from_triple)
		sourced_natural_language_triples_set += extract_sourced_graph(label_graph, lambda x: x[0])
		sourced_natural_language_triples_set += extract_sourced_graph(label_graph, lambda x: x[-1])
		sourced_natural_language_triples_set = list(unique_everseen(sourced_natural_language_triples_set))
		return sourced_natural_language_triples_set

	def get_labeled_graph_from_concept_graph(self, concept_graph):
		def labeled_triples_gen():
			# Get labeled triples
			for original_triple in concept_graph:
				s,p,o = original_triple
				# if s == o:
				# 	continue
				# p_is_subclassof = p == SUBCLASSOF_PREDICATE
				# if p_is_subclassof: # remove redundant triples not directly concerning the concept
				# 	if o!=concept_uri and s!=concept_uri:
				# 		continue
				for label_p in self.label_dict.get(p,[p]):
					label_p_context = set(self.get_paragraph_id_from_concept_id(p)) # get label sources
					for label_s in self.label_dict.get(s,[s]):
						if label_p_context: # triple with sources
							label_s_context = self.get_paragraph_id_from_concept_id(s) # get label sources
							label_context = label_p_context.intersection(label_s_context)
							if not label_context: # these two labels do not appear together, skip
								continue
						for label_o in self.label_dict.get(o,[o]):
							if label_p_context: # triple with sources
								label_o_context = self.get_paragraph_id_from_concept_id(o) # get label sources
								if not label_context.intersection(label_o_context): # these labels do not appear together, skip
									continue
							# if p_is_subclassof and labels_are_similar(label_s,label_o):
							# 	continue
							labeled_triple = (label_s,label_p,label_o)
							yield (labeled_triple, original_triple)
		return unique_everseen(labeled_triples_gen())

	def get_subclass_replacer(self, superclass):
		superclass_set = set([superclass])
		subclass_set = self.get_sub_classes(superclass_set, depth=1).difference(superclass_set)
		# print(subclass_set)
		exploded_superclass = explode_concept_key(superclass).strip().lower()
		exploded_subclass_iter = map(lambda x: explode_concept_key(x).strip().lower(), subclass_set)
		exploded_subclass_iter = filter(lambda x: x and not x.startswith(exploded_superclass), exploded_subclass_iter)
		exploded_subclass_list = sorted(exploded_subclass_iter, key=lambda x:len(x), reverse=True)
		# print(exploded_subclass_list)
		if len(exploded_subclass_list) == 0:
			return None
		# print(exploded_superclass, exploded_subclass_list)
		subclass_regexp = re.compile('|'.join(exploded_subclass_list))
		return lambda x,triple: re.sub(subclass_regexp, exploded_superclass, x) if triple[1]!=SUBCLASSOF_PREDICATE else x

	def get_noun_set(self, graph):
		noun_set = set()
		concept_list = list(get_concept_set(graph))
		concept_label_list = list(map(self.get_label, concept_list))
		span_list = self.nlp(concept_label_list)
		for concept, span in zip(concept_list, span_list):
			# print(concept, span)
			if not span:
				continue
			for token in span:
				if token.pos_ == 'NOUN':
					noun_set.add(concept)
					break
		return noun_set
