source .env/bin/activate

cd demo

# NLTK_DATA=$MY_DIR/.env/nltk_data
mkdir logs
mkdir cache

#### Credit Approval System - V1
## TF
mkdir cache/cache_exp3_ca_v1
python assess_degree_of_explainability.py tf 0.15 ./data/experiment_3/explicanda ./data/experiment_3/explainable_information/credit_approval_system/v1 ./cache/cache_exp3_ca_v1 &> ./logs/exp3.ca_v1.tf.log.txt
## FB
mkdir cache/cache_exp3_ca_v1
python assess_degree_of_explainability.py fb 0.15 ./data/experiment_3/explicanda ./data/experiment_3/explainable_information/credit_approval_system/v1 ./cache/cache_exp3_ca_v1 &> ./logs/exp3.ca_v1.fb.log.txt 

#### Credit Approval System - V2
## TF
mkdir cache/cache_exp3_ca_v2
python assess_degree_of_explainability.py tf 0.15 ./data/experiment_3/explicanda ./data/experiment_3/explainable_information/credit_approval_system/v2 ./cache/cache_exp3_ca_v2 &> ./logs/exp3.ca_v2.tf.log.txt
## FB
mkdir cache/cache_exp3_ca_v2
python assess_degree_of_explainability.py fb 0.15 ./data/experiment_3/explicanda ./data/experiment_3/explainable_information/credit_approval_system/v2 ./cache/cache_exp3_ca_v2 &> ./logs/exp3.ca_v2.fb.log.txt 
