#!/bin/bash

virtualenv .env -p python3.7
source .env/bin/activate

pip --use-deprecated=legacy-resolver install pip==22.2.2
pip --use-deprecated=legacy-resolver install -U setuptools wheel twine
echo 'Installing DoXpy'
pip --use-deprecated=legacy-resolver install -e doxpy

python -m spacy download en_core_web_trf
python -m spacy download en_core_web_md
python -m nltk.downloader stopwords punkt averaged_perceptron_tagger framenet_v17 wordnet brown omw-1.4
