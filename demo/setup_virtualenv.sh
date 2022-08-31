#!/bin/bash

virtualenv .env -p python3.7
source .env/bin/activate

# cd ../Packages

pip install -U pip
pip install -U setuptools wheel twine
echo 'Install DoXpy'
cd ..
pip install -e doxpy
cd demo
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_md
# python3 -m nltk.downloader -d $MY_DIR/.env/nltk_data stopwords punkt averaged_perceptron_tagger framenet_v17 wordnet brown 
python -m nltk.downloader stopwords punkt averaged_perceptron_tagger framenet_v17 wordnet brown omw-1.4
