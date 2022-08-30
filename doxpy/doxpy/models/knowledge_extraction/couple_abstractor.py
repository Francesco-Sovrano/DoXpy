from doxpy.models.knowledge_extraction.couple_extractor import CoupleExtractor
from doxpy.misc.jsonld_lib import *
try:
	from nltk.corpus import framenet as fn
except OSError:
	print('Downloading nltk::framenet\n'
		"(don't worry, this will only happen once)")
	import nltk
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('framenet_v17')
	from nltk.corpus import framenet as fn
try:
	from pywsd import disambiguate
	from pywsd.similarity import max_similarity
	from pywsd.lesk import simple_lesk, adapted_lesk, cosine_lesk
except OSError:
	print("Downloading nltk::wordnet (don't worry, this will only happen once)")
	import nltk
	nltk.download('wordnet')
	from pywsd import disambiguate
	from pywsd.similarity import max_similarity
	from pywsd.lesk import simple_lesk, adapted_lesk, cosine_lesk
import logging

class CoupleAbstractor(CoupleExtractor):
	def abstract_couple_list(self, concept_dict_list):
		assert False, 'Not implemented'

class WordnetAbstractor(CoupleAbstractor):

	# def __init__(self, model_options):
	# 	nltk.download('punkt')
	# 	nltk.download('averaged_perceptron_tagger')
	# 	nltk.download('wordnet')
	# 	super().__init__(model_options)
		
	'''
	Firstly, the OPs sort of confused between relatedness and similarity, the distinction is fine but it's worth noting.

	Semantic relatedness measures how related two concepts are, using any kind of relation; algorithms:
	* Lexical Chains (Hirst and St-Onge, 1998)
	* Adapted/Extended Sense Overlaps algorithm (Banerjee and Pedersen, 2002/2003)
	* Vectorized Sense Overlaps (Patwardhan, 2003)
	
	Semantic similarity only considers the IS-A relation (i.e. hypernymy / hyponymy); algorithms:
	* Wu-Palmer measure (Wu and Palmer 1994)
	* Resnik measure (Resnik 1995)
	* Jiang-Conrath measure (Jiang and Conrath 1997)
	* Leacock-Chodorow measure (Leacock and Chodorow 1998)
	* Lin measure (Lin 1998)	
	Resnik, Jiang-Conrath and Lin measures are based on information content. The information content of a synset is -log the sum of all probabilities (computed from corpus frequencies) of all words in that synset (Resnik, 1995).
	Wu-Palmer and Leacock-Chodorow are based on path length; the similarity between two concepts /synsets is respective of the number of nodes along the shortest path between them.

	The list given above is inexhaustive, but historically, we can see that using similarity measure is sort of outdated since relatedness algorithms considers more relations and should theoretically give more disambiguating power to compare concepts.
	'''
	def abstract_couple_list(self, triplet_list):
		disambiguation_cache = {}
		self.logger.info('Abstracting couples with Wordnet..')
		for triplet in self.tqdm(triplet_list):
			s,p,o = triplet
			sentence_text = p['source']['sentence_text']
			if sentence_text not in disambiguation_cache:
				sentence_disambiguation = disambiguate(
					sentence_text,
					algorithm=cosine_lesk, 
					#similarity_option='wu-palmer',
				)
				disambiguation_cache[sentence_text] = {k.lower():v for k,v in sentence_disambiguation}
			synset_dict = disambiguation_cache[sentence_text]
			for c in (s,o):
				c['concept']['synset'] = synset_dict.get(c['concept']['text'], None)
				for concept_core_dict in c['concept_core']:
					concept_core_dict['synset'] = synset_dict.get(concept_core_dict['text'], None)
		return triplet_list

