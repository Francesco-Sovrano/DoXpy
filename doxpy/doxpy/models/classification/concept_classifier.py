from doxpy.misc.doc_reader import DocParser
from doxpy.models.classification.sentence_classifier import SentenceClassifier
from doxpy.models.knowledge_extraction.concept_extractor import ConceptExtractor
from more_itertools import unique_everseen
from collections import Counter
try:
	from nltk.corpus import stopwords
except Exception as e:
	import nltk
	print('Downloading nltk::stopwords\n'
		"(don't worry, this will only happen once)")
	nltk.download('stopwords')
	from nltk.corpus import stopwords
import re
import json
import itertools

class ConceptClassifier(SentenceClassifier):
	def __init__(self, model_options):
		self.logger.info('Initialising ConceptClassifier')
		super().__init__(model_options)
		self.concept_extractor = ConceptExtractor(model_options)

	def set_concept_description_dict(self, concept_description_dict):
		id_doc_list = tuple((
			(key,description) 
			for key, value in concept_description_dict.items() 
			for description in value
			if description
		))
		self.set_documents(id_doc_list)
		return self
	
	def lemmatize_spacy_document(self, doc):
		return [
			token.lemma_.casefold().strip()
			for token in doc
			if not token.is_punct #and token.lemma_.lower() != '-pron-'
			# Remove stop tokens: <https://stackoverflow.com/questions/40288323/what-do-spacys-part-of-speech-and-dependency-tags-mean>
			#if not (token.is_punct or token.pos_ in ['PART','DET','ADP','CONJ','SCONJ'])
		]
	
	def get_concept_dict(self, doc_parser: DocParser, concept_counter_dict=None, similarity_threshold=None, with_numbers=True, size=None, remove_stopwords=True, lemmatized=True, tfidf_importance=None, concept_label_filter=None, concept_id_filter=None, parallel_extraction=False):
		if concept_counter_dict is None:
			concept_counter_dict = {}
		# Extract concept dict list
		concept_dict_list = self.concept_extractor.get_concept_list(doc_parser, remove_source_paragraph=False, remove_idx=True, remove_span=True, parallel_extraction=parallel_extraction)
		get_concept_label = lambda x: x['concept']['lemma' if lemmatized else 'text']
		# Remove unwanted concepts
		filter_empty_fn = lambda x: x
		filter_stopwords_fn = lambda x: x not in stopwords.words('english')
		filter_numbers_fn = lambda x: re.search(r'\d', x) is None
		filter_fn_list = [filter_empty_fn]
		if concept_label_filter:
			filter_fn_list.append(concept_label_filter)
		if with_numbers:
			filter_fn_list.append(filter_numbers_fn)
		if remove_stopwords:
			filter_fn_list.append(filter_stopwords_fn)
		def global_filter_fn(x):
			for fn in filter_fn_list:
				if not fn(x):
					return False
			return True
		concept_dict_list = list(filter(lambda x: global_filter_fn(get_concept_label(x)), concept_dict_list))
		# Extract concept_counter
		concept_iter = map(get_concept_label, concept_dict_list)
		concept_list = tuple(concept_iter)
		concept_counter = Counter(concept_list)
		# Merge input concept_counter_dict with concept_counter
		for concept,count in concept_counter.items():
			if concept not in concept_counter_dict:
				concept_counter_dict[concept] = {
					'count': count, 
					'source_list': [],
					'similar_to': []
				}
			else:
				concept_counter_dict[concept]['count'] += count
		# Add sources
		for concept, cdict in zip(concept_list, concept_dict_list):
			# concept_counter_dict[concept]['span'] = cdict['concept']['span']
			concept_counter_dict[concept]['source_list'].append(cdict['source'])
		# Add similarities
		if not concept_counter_dict:
			return {}
		text_list, cdict_list = zip(*concept_counter_dict.items())
		# formatted_text_list = tuple(map(lambda x: x['span'], cdict_list))
		index_of_most_similar_documents_list = self.get_index_of_most_similar_documents(
			self.get_formatted_query_similarity(text_list, tfidf_importance=tfidf_importance), 
			similarity_threshold= similarity_threshold,
			similarity_type= 'weighted',
		)
		for concept, index_of_most_similar_documents in zip(text_list, index_of_most_similar_documents_list):
			if concept_id_filter is not None:
				index_of_most_similar_documents = filter(lambda x: concept_id_filter(x['id']), index_of_most_similar_documents)
			if concept_label_filter is not None:
				index_of_most_similar_documents = filter(lambda x: concept_label_filter(x['doc']), index_of_most_similar_documents)
			concept_counter_dict[concept]['similar_to'] = tuple(itertools.islice(index_of_most_similar_documents, size) if size else index_of_most_similar_documents)
			concept_counter_dict[concept]['source_list'] = tuple(unique_everseen(concept_counter_dict[concept]['source_list'], key=lambda x:x['sentence_text']))
		return concept_counter_dict

	@staticmethod
	def get_missing_concepts_counter(concept_dict):
		return {
			concept: value['count']
			for concept, value in concept_dict.items()
			if len(value['similar_to'])==0
		}

	def annotate(self, doc_parser: DocParser, similarity_threshold=None, max_concepts_per_alignment=1, tfidf_importance=None, concept_id_filter=None, is_preprocessed_content=False):
		self.logger.info(f'Annotating {"preprocessed" if is_preprocessed_content else "new"} documents..')
		if not is_preprocessed_content:
			concept_dict = self.get_concept_dict(
				doc_parser, 
				similarity_threshold= similarity_threshold, 
				with_numbers= True, 
				lemmatized= False,
				remove_stopwords= True,
				size= max_concepts_per_alignment,
				tfidf_importance= tfidf_importance,
				concept_id_filter=concept_id_filter,
				parallel_extraction=False,
			)
			annotation_iter = (
				{
					'text': concept_label,
					'annotation': concept_uri_dict['id'],
					'similarity': concept_uri_dict['similarity'],
					'syntactic_similarity': concept_uri_dict['syntactic_similarity'],
					'semantic_similarity': concept_uri_dict['semantic_similarity'],
				}
				for concept_label,similarity_dict in concept_dict.items()
				for concept_uri_dict in similarity_dict['similar_to']
			)
			annotation_iter = unique_everseen(annotation_iter, key=lambda x: x['text'])
		else:
			content_txt = ' '.join(unique_everseen(doc_parser.get_content_iter())).casefold()
			annotation_iter = (
				{
					'text': concept_label,
					'annotation': concept_id,
					# 'similarity': concept_uri_dict['similarity'],
					'similarity': 1,
					# 'syntactic_similarity': concept_uri_dict['syntactic_similarity'],
					'syntactic_similarity': 1,
					# 'semantic_similarity': concept_uri_dict['semantic_similarity'],
					'semantic_similarity': 0,
				}
				for concept_id, concept_label in zip(self.ids, self.documents)
				if concept_id_filter(concept_id) 
				and len(concept_label) > 1
				and concept_label.casefold() in content_txt
			)

		annotation_list = list(annotation_iter)
		# print(json.dumps(annotation_list, indent=4))
		return annotation_list
