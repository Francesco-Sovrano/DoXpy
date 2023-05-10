source .env/bin/activate

cd demo

# NLTK_DATA=$MY_DIR/.env/nltk_data
mkdir logs
mkdir cache

#### EXP 1
## TF
mkdir cache/cache_exp3
python assess_degree_of_explainability.py tf 0.15 ./data/experiment_3/explicanda ./data/experiment_3/explainable_information ./cache/cache_exp3 &> ./logs/exp3.tf.log.txt
## FB
mkdir cache/cache_exp3
python assess_degree_of_explainability.py fb 0.15 ./data/experiment_3/explicanda ./data/experiment_3/explainable_information ./cache/cache_exp3 &> ./logs/exp3.fb.log.txt 
