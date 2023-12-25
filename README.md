# DoXpy: An Objective Metric for Explainable AI

DoXpy is a pip-installable python library giving you all that is necessary to objectively estimate the amount of explainability of any English piece of information (i.e. the output of an Explainable AI), scaling pretty well from single paragraphs to whole sets of documents.

DoXpy relies on deep language models for
- Answer Retrieval
- Sentence Embedding

This is why DoXpy supports several state-of-the-art deep learning libraries for NLP, making it easy to change deep language models at wish.
The supported libraries are:
- [Spacy](https://spacy.io)
- [Huggingface](https://huggingface.co)
- [TensorFlow Hub](https://www.tensorflow.org/hub/)
- [SentenceTransformers](https://www.sbert.net/index.html)

DoXpy is model-agnostic, which means that it can work with the output of any AI, if in English, regardless of its specific characteristics.
To do so, DoXpy exploits a specific theoretical model from Ordinary Language Philosophy called Achinstein's Theory of Explanations. 
You may find a thorough discussion of the underlying theory in this paper [An Objective Metric for Explainable AI: How and Why to Estimate the Degree of Explainability](https://www.sciencedirect.com/science/article/abs/pii/S0950705123006160). 

**Preprint available at: [http://arxiv.org/abs/2109.05327](http://arxiv.org/abs/2109.05327)**
  
## Installation
This project has been tested on Debian 9 and macOS Mojave 10.14 with Python 3.7. 
The script [setup_virtualenv.sh](setup_virtualenv.sh) can install DoXpy and all its dependencies in a python3.7 virtualenv.

You can also install DoXpy by downloading this repo and running from within it: 
`pip install  --use-deprecated=legacy-resolver -e doxpy --no-cache-dir`

Before being able to run the [setup_virtualenv.sh](setup_virtualenv.sh) script, you have to install: virtualenv, python3-dev, python3-pip and make. 

For a simple example of how to use DoXpy, please consider the script [simple_example.py](simple_example.py).

## What to know
DoXpy is a python library allowing you to measure different aspects of explainability, such as fruitfulness, exactness and similarity to the explanandum.

As shown in [demo/assess_degree_of_explainability.py](demo/assess_degree_of_explainability.py), DoXpy relies on three main components to work:
- a [Knowledge Graph Extractor](doxpy/doxpy/models/knowledge_extraction/knowledge_graph_extractor.py)
- an [Answer Retriever](doxpy/doxpy/models/reasoning/answer_retriever.py)
- a [DoX Estimator](doxpy/doxpy/models/estimation/dox_estimator.py)

The Knowledge Graph Extractor can analyse any set of English documents (allowed formats: '.akn', '.html', '.pdf', '.json'), extracting from them a knowledge graph of information units.

The knowledge graph is then given as input to the Answer Retriever. 
Hence, the Answer Retriever is used by the DoX Estimator for computing the (average) DoX.

For the DoX Estimator to compute the DoX, it is necessary to define what is to be explained (i.e., the explanandum, the set of aspects to be explained). This may be done by selecting all the nodes in the previously extracted knowledge graph (as shown in [demo/assess_degree_of_explainability.py](demo/assess_degree_of_explainability.py)).

The DoX is a set of numbers. Each one of these numbers approximatively describes how well the analysed information can explain, on average, all the explanandum aspects by answering a pre-defined set of archetypal questions (e.g. WHY, HOW, WHEN, WHO) deemed to be sufficient to capture most of the explanation goals.

Considering that the DoX is a set, it is not sortable. 
This is why you need to average the values in the set as an average DoX. 

The default archetypes (or archetypal questions) used for the computation of DoX are the values of AnswerRetriever:: archetypal_questions_dict, and they are the following:
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

It is possible to manually specify a different list of archetypal questions, as shown in [demo/assess_degree_of_explainability.py](demo/assess_degree_of_explainability.py).

For more about why we selected all these archetypes and many more details, please read [An Objective Metric for Explainable AI: How and Why to Estimate the Degree of Explainability](http://arxiv.org/abs/2109.05327). 

## How DoX is computed: an example

Let's walk through an example using Definitions 2 through 4 of [An Objective Metric for Explainable AI: How and Why to Estimate the Degree of Explainability](http://arxiv.org/abs/2109.05327). We'll use the following sentence as our piece of information $\Phi$: "Aspirin is used to treat pain and reduce inflammation because it inhibits the production of prostaglandins."

Let the set of relevant aspects $A = \{\textit{aspirin, pain, inflammation, prostaglandins}\}$.

Now, let's go through each definition step by step:

1. **Cumulative Pertinence (Definition 2)**: First, we need to compute the cumulative pertinence of details in $\Phi$ to each archetypal question $q_a$ about each aspect $a \in A$. For simplicity, let's consider just two archetypal questions: \texttt{why} and \texttt{how}. We have the following details $D_a$ for each aspect $a$:

- Aspirin: "Aspirin is used to treat pain and inflammation", "Aspirin is used to treat pain", "Aspirin is used to treat inflammation", "Aspirin inhibits the production of prostaglandins"
- Pain: "Aspirin is used to treat pain"
- Inflammation: "Aspirin is used to reduce inflammation"
- Prostaglandins: "Aspirin inhibits the production of prostaglandins"

Next, we need to calculate the pertinence $p(d, q_a)$ of each detail $d \in D_a$ to each question $q_a$ (e.g., \texttt{why} and \texttt{how} questions about each aspect $a$). Let's assume we obtain the following pertinence values (on a scale from 0 to 1) for each detail and each question archetype:

| Related Aspect | Detail                                         | Why Pertinence | How Pertinence |
|----------------|------------------------------------------------|----------------|---------------|
| Aspirin        | Aspirin is used to treat pain and inflammation | 0.8            | 0.6           |
| Aspirin        | Aspirin is used to treat pain | 0.7            | 0.5           |
| Aspirin        | Aspirin is used to treat inflammation | 0.7            | 0.5           |
| Aspirin        | Aspirin inhibits the production of prostaglandins | 0.6            | 0.8           |
| Pain           | Aspirin is used to treat pain                  | 0.4            | 0.3           |
| Inflammation   | Aspirin is used to reduce inflammation         | 0.4            | 0.3           |
| Prostaglandins | Aspirin inhibits the production of prostaglandins | 0.6            | 0.8           |

To refine the information, we have established a duplication threshold of 0.85. If the similarity between two details is above this threshold, we consider them to be duplicates. In this case, we found that "Aspirin is used to treat pain" and "Aspirin is used to treat inflammation" are highly similar to "Aspirin is used to treat pain and inflammation" and their similarity exceeds the duplication threshold. As a result, we have removed these two details and only retained "Aspirin is used to treat pain and inflammation," which is more comprehensive for all related questions.

The following table shows the similarities between various details about aspirin and the detail "Aspirin is used to treat pain and inflammation":

| Detail | Similarity to "Aspirin is used to treat pain and inflammation" | Above Duplication Threshold (r = 0.85) |
|--------|--------|--------------------------------------------------------------|
| Aspirin is used to treat pain | 0.9 | Yes |
| Aspirin is used to treat inflammation | 0.9 | Yes |
| Aspirin inhibits the production of prostaglandins | 0.6 | No |

Now, we can calculate the cumulative pertinence $P_{D_a, q_a}$ for each aspect $a \in A$ and each question $q_a$. Let's assume a pertinence threshold $t = 0.5$. The cumulative pertinence for each aspect and each question archetype would be:

| Aspect         | Why Cumulative Pertinence | How Cumulative Pertinence |
|----------------|---------------------------|---------------------------|
| Aspirin        | 1.4                       | 1.4                       |
| Pain           | 0                       | 0                       |
| Inflammation   | 0                       | 0                       |
| Prostaglandins | 0.6                       | 0.8                       |

2. **Explanatory Illocution (Definition 3)**: Now, we can calculate the explanatory illocution for each aspect $a \in A$. This is a set of tuples containing each archetypal question and its corresponding cumulative pertinence. For example, the explanatory illocution for the aspect "prostaglandins" would be:

$\{<\texttt{why}, 0.6>, <\texttt{how}, 0.8>\}$

3. **Degree of Explainability (Definition 4)**: Finally, we can calculate the Degree of Explainability (DoX) as the average explanatory illocution per aspect. To do this, we sum the cumulative pertinences for each archetypal question and divide by the number of archetypal questions. In this case, we have two archetypal questions: \texttt{why} and \texttt{how}. The Degree of Explainability for each aspect would be:

Aspect | Degree of Explainability
--- | ---
Aspirin | (1.4 + 1.4) / 2 = 1.4
Pain | (0 + 0) / 2 = 0
Inflammation | (0 + 0) / 2 = 0
Prostaglandins | (0.6 + 0.8) / 2 = 0.7

In this example, the Degree of Explainability for the aspect "prostaglandins" is 0.7 and for "aspirin" is 1.4, which is the highest among all aspects in the set $A$. This indicates that the information $\Phi$ is most explanatory regarding the aspect "aspirin" and its relation to pain, inflammation and prostaglandins.

To compute the total Degree of Explainability (DoX) for the set of aspects $A$, we can sum the DoX values for each aspect and divide by the number of aspects. In our example, we have four aspects: aspirin, pain, inflammation, and prostaglandins. Using the previously calculated DoX values, we can compute the total DoX for $A$ as follows:

Total DoX for A = (DoX_aspirin + DoX_pain + DoX_inflammation + DoX_prostaglandins) / 4

Total DoX for A = (1.4 + 0 + 0 + 0.7) / 4 = 0.525

So, the total Degree of Explainability for the set of aspects $A$ is 0.525. This value represents the average explainability of the given information $\Phi$ across all aspects in the set $A$. In this example, the total DoX is relatively low, indicating that the information is not highly explanatory across all aspects. However, it's important to note that the aspects "aspirin" and "prostaglandins" have high individual DoX values, meaning that the information $\Phi$ is more explanatory regarding those specific aspects.

It's important to note that the values obtained for pertinence and the Degree of Explainability depend on the specific method used to calculate them, as well as the chosen threshold value. Different methods or thresholds might yield different results. This example demonstrates a simplified approach to the concepts of Cumulative Pertinence, Explanatory Illocution, and Degree of Explainability, and serves as a starting point for more complex or tailored approaches.

## How to use DoXpy in your project
To use DoXpy on your project, you need to install it first. 
Then you can import it as in [demo/assess_degree_of_explainability.py](demo/assess_degree_of_explainability.py).
You may also need to configure the KnowledgeGraphExtractor,  the AnswerRetriever, and the DoXEstimator.

An example of import extracted from the script mentioned above is the following:
```
from doxpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor
from doxpy.models.estimation.dox_estimator import DoXEstimator
from doxpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from doxpy.models.reasoning.answer_retriever import AnswerRetriever
```

An example of configuration options is the following:
```
ARCHETYPE_FITNESS_OPTIONS = {
	'one_answer_per_sentence': False,
	'answer_pertinence_threshold': 0.6, 
	'answer_to_question_max_similarity_threshold': None,
	'answer_to_answer_max_similarity_threshold': 0.85,
}

KG_MANAGER_OPTIONS = {
	'with_cache': False,
	'with_tqdm': False,
}

KG_BUILDER_DEFAULT_OPTIONS = {
	'spacy_model': 'en_core_web_trf',
	'n_threads': 1,
	'use_cuda': True,

	'max_syntagma_length': None,
	'lemmatize_label': False,

	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		#'cache_dir': '/Users/root/Documents/Software/DLModels/tf_cache_dir',
		'use_cuda': False,
	},
}

CONCEPT_CLASSIFIER_DEFAULT_OPTIONS = {
	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		#'cache_dir': '/Users/root/Documents/Software/DLModels/tf_cache_dir',
		'use_cuda': False,
	},
	'with_centered_similarity': True,
	'default_similarity_threshold': 0.75,
    'default_tfidf_importance': 0,
}

SENTENCE_CLASSIFIER_DEFAULT_OPTIONS = {
	'sbert_model': {
		'url': 'multi-qa-MiniLM-L6-cos-v1',
		#'cache_dir': '/Users/root/Documents/Software/DLModels/sb_cache_dir',
		'use_cuda': True,
	},
	'default_tfidf_importance': 0,
}
```

## Experiments
Part of the experiments discussed in [An Objective Metric for Explainable AI: How and Why to Estimate the Degree of Explainability](http://arxiv.org/abs/2109.05327) can be run by executing the script [assessment_software_documentation.sh](assessment_software_documentation.sh).
This script runs the automatic assessment of the DoX for the two experiments on all the different versions of the two considered AI-based systems.
The results of the automatic assessment can be found at [demo/logs](demo/logs).

On the other hand, the code of the applications adopted for the user study is available at this GitHub repository: [YAI4Hu](https://github.com/Francesco-Sovrano/YAI4Hu).

The 2 XAI-based systems are: 
- a Heart Disease Predictor (HD)
- a Credit Approval System (CA)

For each of these systems, we have three different versions:
- a Normal AI-based Explanation (NAE): showing only the bare output of an AI together with some extra realistic contextual information, therefore making no use of any XAI.
- a Normal XAI-only Explainer (NXE): showing the output of an XAI and the information given by NAE.
- a 2nd-Level Exhaustive Explanatory Closure (2EC): the output of NXE connected to a 2nd (non-expandable) level of information consisting of an exhaustive and verbose set of autonomous static explanatory resources in the form of web pages.

The adopted AI are:
- [XGBoost](https://dl.acm.org/doi/abs/10.1145/2939672.2939785) for HD
- a Neural Network for CA

The adopted XAI are:
- [TreeSHAP](https://www.nature.com/articles/s42256-019-0138-9) for HD
- [CEM](https://dl.acm.org/doi/abs/10.5555/3326943.3326998) for CA

The two experiments are the following:
- The 1st compares the degree of explainability of NAE and NXE, expecting NXE to be the best because it relied on XAI.
- The 2nd compares the degree of explainability of NXE and 2EC, expecting 2EC to be the best because 2EC can explain many more things than NXE.

## Citations
This code is free. So, if you use this code anywhere, please cite us:
```
@article{sovrano2023objective,
  title={An objective metric for explainable AI: how and why to estimate the degree of explainability},
  author={Sovrano, Francesco and Vitali, Fabio},
  journal={Knowledge-Based Systems},
  volume={278},
  pages={110866},
  year={2023},
  publisher={Elsevier}
}
```

Thank you!

## Support
For any problem or question, please contact me at `cesco.sovrano@gmail.com`
