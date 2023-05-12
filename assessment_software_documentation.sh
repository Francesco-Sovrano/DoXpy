source .env/bin/activate

cd demo

# NLTK_DATA=$MY_DIR/.env/nltk_data
mkdir logs
mkdir cache

ANSWER_PERTINENCE_THRESHOLD=0.15
SYNONYMITY_THRESHOLD=1

#### EXP 1
## TF
mkdir cache/cache_exp1_ca_nae
python assess_degree_of_explainability.py tf $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_1/CrediAppS/explicanda ./data/experiment_1/CrediAppS/explainable_information/NAE ./cache/cache_exp1_ca_nae &> ./logs/exp1_ca_nae.tf.log.txt
mkdir cache/cache_exp1_hd_nae
python assess_degree_of_explainability.py tf $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_1/HeartDisP/explicanda ./data/experiment_1/HeartDisP/explainable_information/NAE ./cache/cache_exp1_hd_nae &> ./logs/exp1_hd_nae.tf.log.txt 
mkdir cache/cache_exp1_ca_nxe
python assess_degree_of_explainability.py tf $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_1/CrediAppS/explicanda ./data/experiment_1/CrediAppS/explainable_information/NXE ./cache/cache_exp1_ca_nxe &> ./logs/exp1_ca_nxe.tf.log.txt
mkdir cache/cache_exp1_hd_nxe
python assess_degree_of_explainability.py tf $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_1/HeartDisP/explicanda ./data/experiment_1/HeartDisP/explainable_information/NXE ./cache/cache_exp1_hd_nxe &> ./logs/exp1_hd_nxe.tf.log.txt 
## FB
mkdir cache/cache_exp1_ca_nae
python assess_degree_of_explainability.py fb $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_1/CrediAppS/explicanda ./data/experiment_1/CrediAppS/explainable_information/NAE ./cache/cache_exp1_ca_nae &> ./logs/exp1_ca_nae.fb.log.txt 
mkdir cache/cache_exp1_hd_nae
python assess_degree_of_explainability.py fb $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_1/HeartDisP/explicanda ./data/experiment_1/HeartDisP/explainable_information/NAE ./cache/cache_exp1_hd_nae &> ./logs/exp1_hd_nae.fb.log.txt 
mkdir cache/cache_exp1_ca_nxe
python assess_degree_of_explainability.py fb $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_1/CrediAppS/explicanda ./data/experiment_1/CrediAppS/explainable_information/NXE ./cache/cache_exp1_ca_nxe &> ./logs/exp1_ca_nxe.fb.log.txt
mkdir cache/cache_exp1_hd_nxe
python assess_degree_of_explainability.py fb $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_1/HeartDisP/explicanda ./data/experiment_1/HeartDisP/explainable_information/NXE ./cache/cache_exp1_hd_nxe &> ./logs/exp1_hd_nxe.fb.log.txt 

#### EXP 2
## TF
mkdir cache/cache_exp2_ca_nxe
python assess_degree_of_explainability.py tf $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_2/CrediAppS/explicanda ./data/experiment_2/CrediAppS/explainable_information/NXE ./cache/cache_exp2_ca_nxe &> ./logs/exp2_ca_nxe.tf.log.txt
mkdir cache/cache_exp2_hd_nxe
python assess_degree_of_explainability.py tf $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_2/HeartDisP/explicanda ./data/experiment_2/HeartDisP/explainable_information/NXE ./cache/cache_exp2_hd_nxe &> ./logs/exp2_hd_nxe.tf.log.txt 
mkdir cache/cache_exp2_ca_2ec
python assess_degree_of_explainability.py tf $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_2/CrediAppS/explicanda ./data/experiment_2/CrediAppS/explainable_information/2EC ./cache/cache_exp2_ca_2ec &> ./logs/exp2_ca_2ec.tf.log.txt
mkdir cache/cache_exp2_hd_2ec
python assess_degree_of_explainability.py tf $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_2/HeartDisP/explicanda ./data/experiment_2/HeartDisP/explainable_information/2EC ./cache/cache_exp2_hd_2ec &> ./logs/exp2_hd_2ec.tf.log.txt 
#### EXP 2
## FB
mkdir cache/cache_exp2_ca_nxe
python assess_degree_of_explainability.py fb $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_2/CrediAppS/explicanda ./data/experiment_2/CrediAppS/explainable_information/NXE ./cache/cache_exp2_ca_nxe &> ./logs/exp2_ca_nxe.fb.log.txt
mkdir cache/cache_exp2_hd_nxe
python assess_degree_of_explainability.py fb $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_2/HeartDisP/explicanda ./data/experiment_2/HeartDisP/explainable_information/NXE ./cache/cache_exp2_hd_nxe &> ./logs/exp2_hd_nxe.fb.log.txt 
mkdir cache/cache_exp2_ca_2ec
python assess_degree_of_explainability.py fb $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_2/CrediAppS/explicanda ./data/experiment_2/CrediAppS/explainable_information/2EC ./cache/cache_exp2_ca_2ec &> ./logs/exp2_ca_2ec.fb.log.txt
mkdir cache/cache_exp2_hd_2ec
python assess_degree_of_explainability.py fb $ANSWER_PERTINENCE_THRESHOLD $SYNONYMITY_THRESHOLD ./data/experiment_2/HeartDisP/explicanda ./data/experiment_2/HeartDisP/explainable_information/2EC ./cache/cache_exp2_hd_2ec &> ./logs/exp2_hd_2ec.fb.log.txt 

