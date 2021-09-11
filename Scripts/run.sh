MY_DIR="`python -c "import os; print(os.path.realpath('$1'))"`"
cd $MY_DIR

source .env/bin/activate

# NLTK_DATA=$MY_DIR/.env/nltk_data

cd FB_model
python3 assess_degree_of_explainability.py ../../Experiments/experiment_1/CrediAppS/explicanda ../../Experiments/experiment_1/CrediAppS/explainable_information/NAE ./cache_exp1_ca_nae &> exp1_ca_nae.fb.log.txt  
python3 assess_degree_of_explainability.py ../../Experiments/experiment_1/HeartDisP/explicanda ../../Experiments/experiment_1/HeartDisP/explainable_information/NAE ./cache_exp1_hd_nae &> exp1_hd_nae.fb.log.txt
python3 assess_degree_of_explainability.py ../../Experiments/experiment_1/CrediAppS/explicanda ../../Experiments/experiment_1/CrediAppS/explainable_information/NXE ./cache_exp1_ca_nxe &> exp1_ca_nxe.fb.log.txt 
python3 assess_degree_of_explainability.py ../../Experiments/experiment_1/HeartDisP/explicanda ../../Experiments/experiment_1/HeartDisP/explainable_information/NXE ./cache_exp1_hd_nxe &> exp1_hd_nxe.fb.log.txt
python3 assess_degree_of_explainability.py ../../Experiments/experiment_2/CrediAppS/explicanda ../../Experiments/experiment_2/CrediAppS/explainable_information/NXE ./cache_exp2_ca_nxe &> exp2_ca_nxe.fb.log.txt 
python3 assess_degree_of_explainability.py ../../Experiments/experiment_2/HeartDisP/explicanda ../../Experiments/experiment_2/HeartDisP/explainable_information/NXE ./cache_exp2_hd_nxe &> exp2_hd_nxe.fb.log.txt
python3 assess_degree_of_explainability.py ../../Experiments/experiment_2/CrediAppS/explicanda ../../Experiments/experiment_2/CrediAppS/explainable_information/2EC ./cache_exp2_ca_2ec &> exp2_ca_2ec.fb.log.txt 
python3 assess_degree_of_explainability.py ../../Experiments/experiment_2/HeartDisP/explicanda ../../Experiments/experiment_2/HeartDisP/explainable_information/2EC ./cache_exp2_hd_2ec &> exp2_hd_2ec.fb.log.txt
cd ..

cd TF_model
python3 assess_degree_of_explainability.py ../../Experiments/experiment_1/CrediAppS/explicanda ../../Experiments/experiment_1/CrediAppS/explainable_information/NAE ./cache_exp1_ca_nae &> exp1_ca_nae.tf.log.txt 
python3 assess_degree_of_explainability.py ../../Experiments/experiment_1/HeartDisP/explicanda ../../Experiments/experiment_1/HeartDisP/explainable_information/NAE ./cache_exp1_hd_nae &> exp1_hd_nae.tf.log.txt
python3 assess_degree_of_explainability.py ../../Experiments/experiment_1/CrediAppS/explicanda ../../Experiments/experiment_1/CrediAppS/explainable_information/NXE ./cache_exp1_ca_nxe &> exp1_ca_nxe.tf.log.txt 
python3 assess_degree_of_explainability.py ../../Experiments/experiment_1/HeartDisP/explicanda ../../Experiments/experiment_1/HeartDisP/explainable_information/NXE ./cache_exp1_hd_nxe &> exp1_hd_nxe.tf.log.txt
python3 assess_degree_of_explainability.py ../../Experiments/experiment_2/CrediAppS/explicanda ../../Experiments/experiment_2/CrediAppS/explainable_information/NXE ./cache_exp2_ca_nxe &> exp2_ca_nxe.tf.log.txt 
python3 assess_degree_of_explainability.py ../../Experiments/experiment_2/HeartDisP/explicanda ../../Experiments/experiment_2/HeartDisP/explainable_information/NXE ./cache_exp2_hd_nxe &> exp2_hd_nxe.tf.log.txt
python3 assess_degree_of_explainability.py ../../Experiments/experiment_2/CrediAppS/explicanda ../../Experiments/experiment_2/CrediAppS/explainable_information/2EC ./cache_exp2_ca_2ec &> exp2_ca_2ec.tf.log.txt 
python3 assess_degree_of_explainability.py ../../Experiments/experiment_2/HeartDisP/explicanda ../../Experiments/experiment_2/HeartDisP/explainable_information/2EC ./cache_exp2_hd_2ec &> exp2_hd_2ec.tf.log.txt
cd ..