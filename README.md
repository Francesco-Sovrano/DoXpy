# DoXpy: An Objective Metric for Explainable AI

**N.B. This documentation will be updated soon**

DoXpy is a pip-installable python library, giving you all that is necessary to objectively estimate the amount of explainability of any English piece of information (i.e. the output of an Explainable AI), scaling pretty well from single paragraphs to whole sets of documents.

DoXpy relies on deep language models for
- Question Answer Retrieval
- Sentence Embedding

This is why DoXpy supports several state-of-the-art deep learning libraries for NLP, making it easy to change deep language models at wish.
The supported libraries are:
- [Spacy](https://spacy.io)
- [Huggingface](https://huggingface.co)
- [TensorFlow Hub](https://www.tensorflow.org/hub/)
- [SentenceTransformers](https://www.sbert.net/index.html)

DoXpy is model-agnostic, this means that it can work with the output of any AI, if in English, regardless its inner characteristics.
To do so, DoXpy exploits a specific theoretical model from Ordinary Language Philosophy called the Achinstein’s Theory of Explanations. 
You may find a thorough discussion of the underlying theory in this paper [An Objective Metric for Explainable AI: How and Why to Estimate the Degree of Explainability](http://arxiv.org/abs/2109.05327).
  
## Installation
This project has been tested on Debian 9 and macOS Mojave 10.14 with Python 3.7. 
The script [setup_virtualenv.sh](setup_virtualenv.sh) can be used to install DoXpy and all its dependencies in a python3.7 virtualenv.

You can also install DoXpy by downloading this repo and running from within it: 
`pip install  --use-deprecated=legacy-resolver -e doxpy --no-cache-dir`

Before being able to run the [demo/setup_virtualenv.sh](demo/setup_virtualenv.sh) script you have to install: virtualenv, python3-dev, python3-pip and make. 

## What to know
DoXpy is a python library allowing you to measure different aspects of explainability as fruitfulness, exactness and similarity to the explanandum.

As shown in [demo/assess_degree_of_explainability_TF.py](demo/assess_degree_of_explainability_TF.py), DoXpy relies on three main components to work:
- a Knowledge Graph Extractor (or OntologyBuilder)
- a Question Answer Retrieval system (or QuestionAnswerer)
- an Estimator of the Degree of Explainability (or ExplainabilityEstimator)

The OntologyBuilder can analyse any set of English documents (allowed formats: '.akn', '.html', '.pdf', '.json'), extracting from them a knowledge graph of information units.

The knowledge graph of information units is then given as input to the QuestionAnswerer. 
Hence, the QuestionAnswerer is used by the ExplainabilityEstimator for computing the DoX and the WeDoX.

In order for the ExplainabilityEstimator to compute the DoX, it is necessary to define what is the explanandum (the set of aspects to be explained). This may be done by selecting all the nodes in the previously extracted knowledge graph (as shown in [demo/assess_degree_of_explainability_TF.py](demo/assess_degree_of_explainability_TF.py)) or by manually selecting the set of important aspects, i.e. ['credit approval', 'dataset', artificial intelligence', etc..].

The DoX is a vector of numbers. Each one of these numbers approximatively describes how well the analysed information can explain, in average, all the explanandum aspects, by answering to a pre-defined set of archetypal questions (e.g. WHY, HOW, WHEN, WHO) deemed to be sufficient to capture most of the explanation goals.

Considering that the DoX is a vector, it does not allow to say whether some explainable information is overall more explainable than another. 
This is why you may need the Weighted DoX (WeDoX). As the name suggests, the WeDoX computes a weighted sum of the numbers in the DoX. The weight of each archetypal question composing the DoX can be manually specified setting the archetype_weight_dict, passed as parameter to ExplainabilityEstimator::get_weighted_degree_of_explainability. The default weight for each archetype is 0.

The archetypes (or archetypal questions) used for the computation of DoX are in QuestionAnswerer:: archetypal_questions_dict and they are the following:
- What is a description of {X}?
- What is {X}?		
- What is {X}?
- Who {X}?
- Whom {X}?
- Why {X}?
- What if {X}?
- What is {X} for?
- How {X}?
- How much {X}?
- Where {X}?
- When {X}?
- Who by {X}?
- Which {X}?
- Whose {X}?
- In what manner {X}?
- What is the reason {X}?
- What is the result of {X}?
- What is an example of {X}?
- After what {X}?
- While what {X}?
- In what case {X}?
- Despite what {X}?
- What is contrasted with {X}?
- Before what {X}?
- Since when {X}?
- What is similar to {X}?
- Until when {X}?
- Instead of what {X}?
- What is an alternative to {X}?
- Except when {X}?
- Unless what {X}?

For more about why we selected all these archetypes, and many more details, please read [An Objective Metric for Explainable AI: How and Why to Estimate the Degree of Explainability](http://arxiv.org/abs/2109.05327). 

## Usage
To use DoXpy on your own project you need to install it first. 
Then you can import it as done in [demo/assess_degree_of_explainability_TF.py](demo/assess_degree_of_explainability_TF.py) or [demo/assess_degree_of_explainability_FB.py](demo/assess_degree_of_explainability_FB.py).
You may also need to configure the OntologyBuilder, QuestionAnswerer and ExplainabilityEstimator.

An example of import extracted from the aforementioned scripts is the following:
```
from doxpy.models.knowledge_extraction.ontology_builder import OntologyBuilder
from doxpy.models.estimation.explainability_estimator import ExplainabilityEstimator
from doxpy.models.reasoning.question_answerer import QuestionAnswerer
from doxpy.misc.doc_reader import load_or_create_cache
from doxpy.misc.graph_builder import get_betweenness_centrality, save_graphml, get_concept_description_dict
from doxpy.misc.jsonld_lib import *
```

An example of configuration options is the following:
```
ARCHETYPE_FITNESS_OPTIONS = {
	'only_overview_exploration': False,
	'answer_pertinence_threshold': 0.15, 
	'answer_to_question_similarity_threshold': 0.9502, 
	'answer_to_answer_similarity_threshold': 0.9502, 
}
OVERVIEW_OPTIONS = {
	'answer_pertinence_threshold': None, # default is None
	'answer_to_question_similarity_threshold': None, # default is 0.9502
	'answer_to_answer_similarity_threshold': None, # default is 0.9502
	'minimise': False,
	'sort_archetypes_by_relevance': False,
	'set_of_archetypes_to_consider': None, # set(['why','how'])
	'answer_horizon': 10,
	'remove_duplicate_answer_sentences': True,

	'top_k': 100,
	'include_super_concepts_graph': False, 
	'include_sub_concepts_graph': True, 
	'add_external_definitions': False, 
	'consider_incoming_relations': True,
	'tfidf_importance': 0,
}

QA_EXTRACTOR_OPTIONS = {
	'models_dir': '/home/toor/Desktop/data/models', 
	'use_cuda': True,

	'sbert_model': {
		'url': 'facebook-dpr-question_encoder-multiset-base', # model for paraphrase identification
		'use_cuda': True,
	},
}

ONTOLOGY_BUILDER_DEFAULT_OPTIONS = {
	'spacy_model': 'en_core_web_trf',
	'n_threads': 1,
	'use_cuda': True,

	'max_syntagma_length': None,
	'lemmatize_label': False,

	'default_similarity_threshold': 0.75,
	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
		'use_cuda': False,
	},
	'with_centered_similarity': True,
}

CONCEPT_CLASSIFIER_DEFAULT_OPTIONS = {
	# 'spacy_model': 'en_core_web_trf',
	# 'n_threads': 1,
	# 'use_cuda': True,

	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
		'use_cuda': False,
	},
	'with_centered_similarity': True,
	'default_similarity_threshold': 0.75,
	# 'default_tfidf_importance': 3/4,
}

SENTENCE_CLASSIFIER_DEFAULT_OPTIONS = {
	# 'spacy_model': 'en_core_web_trf',
	# 'n_threads': 1,
	# 'use_cuda': True,

	'tf_model': {
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder-qa2/3', # English QA
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3', # Multilingual QA # 16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian)
		# 'url': 'https://tfhub.dev/google/LAReQA/mBERT_En_En/1',
		'use_cuda': True,
	}, 
	# 'sbert_model': {
	# 	'url': 'facebook-dpr-question_encoder-multiset-base', # model for paraphrase identification
	# 	'use_cuda': True,
	# },
	'with_centered_similarity': False,
	'with_topic_scaling': False,
	'with_stemmed_tfidf': False,
	# 'default_tfidf_importance': 1/4,
}
```

## Experiments
Part of the experiments discussed in [An Objective Metric for Explainable AI: How and Why to Estimate the Degree of Explainability](http://arxiv.org/abs/2109.05327) can be run executing the script [demo/run_dox_assessment.sh](demo/run_dox_assessment.sh).
This script runs the automatic assessment of the Weighted Degree of Explainability (WeDoX) for both the 2 experiments on all the 3 different versions of the 2 considered AI-based systems.

On the other hand, the code of the applications adopted for the user-study is available at this GitHub repository: [YAI4Hu](https://github.com/Francesco-Sovrano/YAI4Hu).

Anyway, the 2 XAI-based systems are: 
- an Heart Disease Predictor (HD)
- a Credit Approval System (CA)

For each of these systems we have 3 different versions:
- a Normal AI-based Explanation (NAE): showing only the bare output of an AI together with some extra realistic contextual information, therefore making no use of any XAI.
- a Normal XAI-only Explainer (NXE): showing the output of a XAI, as well as the information given by NAE.
- a 2nd-Level Exhaustive Explanatory Closure (2EC): the output of NXE connected to a 2nd (non-expandable) level of information consisting in an exhaustive and verbose set of autonomous static explanatory resources in the form of web-pages.

The adopted AI are:
- [XGBoost](https://arxiv.org/abs/1806.01830) for HD
- a [Neural Network](https://arxiv.org/abs/1806.01830) for CA

The adopted XAI are:
- [TreeSHAP](https://arxiv.org/abs/1806.01830) for HD
- [CEM](https://arxiv.org/abs/1806.01830) for CA

The 2 experiments are the following:
- The 1st one compares the degree of explainability of NAE and NXE, expecting NXE's being the best, because it relied on XAI.
- The 2nd one compares the degree of explainability of NXE and 2EC, expecting 2EC's being the best, because 2EC can explain many more things than NXE.

## Citations
This code is free. So, if you use this code anywhere, please cite us:
```
@article{sovrano2021metric,
  title={An Objective Metric for Explainable AI: How and Why to Estimate the Degree of Explainability},
  author={Sovrano, Francesco and Vitali, Fabio},
  journal={arXiv preprint arXiv:2109.05327},
  url={https://arxiv.org/abs/2109.05327},
  year={2021}
}
```

Thank you!

## Support
For any problem or question please contact me at `cesco.sovrano@gmail.com`
