#!/bin/bash

MY_DIR="`python -c "import os; print(os.path.realpath('$1'))"`"
cd $MY_DIR


# if [ ! -d ".env" ]; then
# 	sh $MY_DIR/setup_python_env.sh 3.8 6 $MY_DIR .env
# fi
# . .env/bin/activate	

virtualenv .env -p python3.7
source .env/bin/activate

# cd ../Packages

pip install pip==21.1.1 --no-cache-dir
pip install --use-deprecated=legacy-resolver -U setuptools wheel twine --no-cache-dir
echo 'Install DoXpy'
cd ..
pip install  --use-deprecated=legacy-resolver -e doxpy --no-cache-dir
cd demo
python3 -m spacy download en_core_web_trf --no-cache-dir
python3 -m spacy download en_core_web_md --no-cache-dir
# python3 -m nltk.downloader -d $MY_DIR/.env/nltk_data stopwords punkt averaged_perceptron_tagger framenet_v17 wordnet brown 
python3 -m nltk.downloader stopwords punkt averaged_perceptron_tagger framenet_v17 wordnet brown 

cd $MY_DIR
