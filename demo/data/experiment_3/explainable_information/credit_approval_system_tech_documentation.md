# AI-Powered Credit Approval System: Comprehensive Technical Documentation

This technical documentation provides a comprehensive overview of the AI-based Credit Approval System designed by our company in adherence to the strict requirements of the EU AI Act. The system assesses the creditworthiness of loan applicants, falling under the high-risk category of AI-based systems. Our objective is to ensure that our AI system is transparent, fair, and robust, while providing a high degree of explainability to both our internal teams and external stakeholders.

We have identified three primary user groups for our system: data scientists, loan officers, and bank customers. Each user group has unique needs and the system accordingly provides tailored explanation methods for machine learning predictions. 

Data scientists are tasked with evaluating the machine learning model prior to its deployment. To provide a comprehensive understanding of the model's behavior, two interpretable rule-based models are presented: Boolean Rule Column Generation (BRCG) and Logistic Rule Regression (LogRR) algorithms from AIX360. BRCG generates simple OR-of-ANDs classification rules, while LogRR offers weighted combinations of rules that maintain high interpretability while enhancing accuracy.

The loan officer makes the final decision based on the model's output. We demonstrate a different way of explaining machine learning predictions by showing examples. Specifically, we use the ProtoDash method (class ProtodashExplainer) to find prototypes or representatives in the training data that are similar to a given loan applicant and receive the same class label.

The bank customer wants to understand the reasons for their application result. For this purpose, we consider the Contrastive Explanations Method (CEM) for explaining the predictions of black box models to end-users. CEM identifies features that are present in the input instance responsible for the model's classification. In addition to these, CEM also identifies features that are (minimally) absent in the input instance but whose presence would have altered the classification.

The system is primarily deployed by loan officers to assess credit eligibility and determine the credit quantum. It predicts the likelihood of punctual payments over a period of two years using data derived from credit reports. Our system is built on the IBM AIX360 platform and leverages the robustness and accuracy of the XGBoost machine learning algorithm. It also includes several advanced libraries and tools to guarantee reliable and highly accurate results.

The system relies on the IBM AIX360 platform and the XGBoost machine learning algorithm to predict creditworthiness. It also utilizes a diverse range of libraries such as joblib, scikit-learn, torch, torchvision, cvxpy, tensorflow, keras, matplotlib, numpy, pandas, scipy, xgboost, bleach, docutils, and Pygments. The model underwent an intensive training period of approximately 8 hours on 8x NVIDIA A100 to ensure optimal performance.

The following are the system requirements for the AI-based Credit Approval System:

- Operating System: Compatibility with Windows, Linux, and macOS.
- Processor: A minimum requirement of an Intel Core i5 processor.
- RAM: At least 8 GB for optimal performance.
- Storage: A minimum of 100 GB of free disk space.
- Graphics Card: A dedicated Nvidia graphics card, such as Nvidia A100.
- Internet Connection: A stable connection for seamless API access.
- Additional Software: Python 3.6 or later, CUDA 10.0, and cuDNN 7.6.5 or later.

These requirements may be subject to changes depending on the system configuration and the volume of data involved.

This documentation provides detailed information about the system, from data sourcing and preprocessing to the predictive and explanatory models utilized. It also outlines our commitment to continuous evaluation of the system's performance in the post-market phase for regulatory compliance.

The following sections detail the specifics of the AI-based Credit Approval System:

1. **Data Source, Pre-processing, and Training**
2. **Explainability Features**
3. **Lifecycle for Evaluating AI System Performance in the Post-Market Phase**
4. **Post-Market Monitoring Plan**

Each section includes a detailed account of our procedures, methodologies, and commitments, providing clear insights into our AI system's operations. By the end of this document, readers should have a thorough understanding of how our Credit Approval AI System functions, how it meets the EU AI Act's requirements, and how we plan to maintain and improve its performance post-deployment. 

Let's dive into the specifics of our AI system, beginning with our data source and how we preprocess and train our models.

## Data Source, Pre-processing, and Training

In this section, we provide details about the origin of our data, the techniques we use to preprocess it, and the methodologies followed during model training. It is crucial to note that all these procedures strictly adhere to privacy regulations and ethical guidelines.

### Data Source

The credit approval system uses the FICO HELOC dataset as its primary data source. The dataset consists of anonymized customer data, including credit history, income levels, employment status, and other relevant financial details. The dataset is composed of a diverse range of customers who have requested a credit line in the range of USD 5,000 - 150,000, which is typically offered by US banks as a percentage of home equity. 

We ensure that all data collection and usage comply with GDPR and other relevant privacy laws. Specifically, all personally identifiable information (PII) is removed or obfuscated to ensure customer privacy.

### Data Pre-processing

Before the data is used to train the credit approval AI model, we follow a pre-processing phase that involves several steps to ensure data quality, integrity, fairness, and non-discrimination. We describe these steps below:

**Data Cleaning.** We begin by meticulously removing any inconsistencies, errors, or duplicates in the credit approval dataset to ensure data quality and integrity.

**Feature Selection.** We identify and select the most relevant features that contribute to a customer's creditworthiness using both domain knowledge and feature importance techniques. This process helps to ensure that only the most relevant features are used to train our model.

**Data Normalization.** We normalize the data to ensure that all features have a similar scale. Normalizing the data helps our model to train more effectively and avoid bias towards any specific feature. This step is crucial in preventing bias towards specific features that may have a larger scale than others.

**Data Bias Reduction.** To ensure fairness and non-discrimination in our credit approval system, we use the Reweighing algorithm to reduce bias in the training data. Specifically, we follow the steps below:

1. **Loading and Preparing Data:** 
We load the initial credit approval dataset and split it into training and testing datasets. We use the testing dataset to assess the accuracy, fairness, and other metrics during the development of the machine learning model.

2. **Computing Fairness Metric on Original Training Dataset:** 
We use the AI Fairness 360 (aif360) toolkit to compute the fairness metric on the original training dataset. We identify the protected attribute 'age' and define privileged and unprivileged values for that attribute. We then use the mean_difference method on the BinaryLabelDatasetMetric class to compare the percentage of favorable outcomes for each group. A negative value indicates less favorable outcomes for the unprivileged group.

3. **Mitigating Bias by Transforming the Original Dataset:** 
To mitigate bias in the training dataset, we use the Reweighing algorithm implemented in the Reweighing class in the aif360.algorithms.preprocessing package. This algorithm transforms the dataset to have more equity in positive outcomes on the protected attribute for the privileged and unprivileged groups. We use the fit and transform methods to perform the transformation, producing a newly transformed training dataset.

4. **Computing Fairness Metric on Transformed Dataset:** 
We use the mean_difference method on the BinaryLabelDatasetMetric class again to compute the fairness metric on the transformed training dataset. This metric indicates the percentage of favorable outcomes for each group after the dataset has been transformed. We see that the Reweighing algorithm was highly effective in removing bias, as the difference in mean outcomes is now 0.0. Therefore, there is no advantage for the privileged group.

By following these steps and utilizing the aif360 toolkit, we can ensure that our credit approval system is fair and unbiased, reducing the risk of discrimination and promoting ethical and responsible use of AI-based systems.

### Data Splitting

After pre-processing, the data is split into three distinct sets: a training set, a validation set, and a testing set. This partitioning enables us to train our models effectively and evaluate their performance objectively:

1. **Training Set:** The majority of the data (around 70%) is used to train our models.
2. **Validation Set:** A portion of the data (around 15%) is set aside to tune the model's parameters and prevent overfitting.
3. **Testing Set:** The remaining data (approximately 15%) is used to test the model's performance on unseen data, which gives us a realistic measure of how the model will perform in real-world scenarios.

The model's performance is evaluated using the testing set, which gives us a realistic measure of how the model will perform in real-world scenarios. The model is regularly monitored and updated to detect and address emerging biases promptly. This monitoring process also enables the system to adapt to changing trends and operate in compliance with the EU AI Act.

### Dataset Variables

The HELOC dataset contains 24 predictor variables and one target variable called RiskPerformance. The table below provides information about the predictor variables and the target variable, including their meaning and monotonicity constraint with respect to the probability of bad = 1:

Meaning | Monotonicity Constraint (with respect to probability of bad = 1)
--- | ---
Age | None
Consolidated version of risk markers | Monotonically Decreasing
Months Since Oldest Trade Open | Monotonically Decreasing
Months Since Most Recent Trade Open | Monotonically Decreasing
Average Months in File | Monotonically Decreasing
Number Satisfactory Trades | Monotonically Decreasing
Number Trades 60+ Ever | Monotonically Decreasing
Number Trades 90+ Ever | Monotonically Decreasing
Percent Trades Never Delinquent | Monotonically Decreasing
Months Since Most Recent Delinquency | Monotonically Decreasing
Max Delinquency/Public Records Last 12 Months | Values 0-7 are monotonically decreasing
Max Delinquency Ever | Values 2-8 are monotonically decreasing
Number of Total Trades (total number of credit accounts) | No constraint
Number of Trades Open in Last12 Months | Monotonically Increasing
Percent Installment Trades | No constraint
Months Since Most Recent Inquiry excluding last 7 days | Monotonically Decreasing
Number of Inquiries Last 6 Months | Monotonically Increasing
Number of Inquiries Last 6 Months excluding last 7 days. Excluding the last 7 days removes inquiries that are likely due to price comparison shopping. | Monotonically Increasing
Net Fraction Revolving Burden. This is revolving balance divided by credit limit | Monotonically Increasing
Net Fraction Installment Burden. This is installment balance divided by original loan amount | Monotonically Increasing
Number Revolving Trades with Balance | No constraint
Number Installment Trades with Balance | No constraint
Number Bank/National Trades with High Utilization Ratio | Monotonically Increasing
Percent Trades with Balance | No constraint
Risk Performance | Target

The target variable, Risk Performance, is a binary variable that indicates whether an applicant was 90 days past due or worse at least once over a period of 24 months from when the credit account was opened. The value "Bad" indicates that an applicant is a bad credit risk, while the value "Good" indicates that they have made their payments without ever being more than 90 days overdue.

### Model Training

The training phase involves fitting our models to the training dataset. The primary predictive model used in our AI system is XGBoost, a gradient boosting framework renowned for its efficiency and performance.

XGBoost works by iteratively combining multiple decision trees to create a strong learner that can predict whether a loan applicant is creditworthy. It employs a novel technique called gradient boosting to improve the performance of decision trees by combining their predictions.

We train the XGBoost model using the training set, tuning its hyperparameters using the validation set to avoid overfitting and underfitting. The model's performance is evaluated using the subset accuracy metric, as calculated by the sklearn.metrics.accuracy_score function.

The XGBoost model has several benefits that make it ideal for our credit approval system. It's capable of handling a large number of features, is robust to outliers in the data, and can model complex non-linear relationships. Furthermore, it provides several parameters that can be tuned to optimize the model's performance.

In addition to its efficiency and performance, XGBoost is also considered state-of-the-art for tabular data (such as the FICO HELOC data). It has consistently outperformed other popular machine learning algorithms in various data science competitions and real-world applications. XGBoost's ability to handle large datasets with many features and its flexibility to incorporate a variety of objective functions and evaluation metrics make it a popular choice for many high-risk AI-based systems, including our credit approval system.

The XGBoost model is trained on the pre-processed and normalized data, with its performance evaluated using subset accuracy, as calculated by the sklearn.metrics.accuracy_score function. Our model is trained to minimize the risk of false negatives (i.e., approving a loan for someone who will default) while balancing the risk of false positives (i.e., denying a loan to someone who would have paid it back). This approach ensures that our model is more accurate.

## Explainability Features

A key feature of our AI system is its explainability. We believe it's essential for data scientists, loan officers, and bank customers to understand the reasoning behind the credit approval decisions made by the system. The system incorporates built-in human oversight to ensure that the outputs are correct. The models are designed to be transparent, so their workings can be easily understood. They are also monitored during deployment, and adjustments can be made as necessary.

To ensure transparency and explainability in our credit approval decisions, we use three different explanatory models: BooleanRuleCG, LogisticRuleRegression, and Generalized Linear Rule Models (GLRM). These models help us understand the reasoning behind the decisions made by our XGBoost model.

1. **BooleanRuleCG (BRCG):** The BooleanRuleCG (BRCG) algorithm generates a set of simple boolean rules using a genetic algorithm. These rules can explain the decisions made by the predictive model, even for complex non-linear relationships. BRCG is designed to produce either a disjunctive normal form (DNF) or a conjunctive normal form (CNF) rule to predict whether an applicant will repay the loan on time. A DNF rule corresponds to an individual rule in the rule set, where the AND clauses in the DNF correspond to individual rules in the rule set. The algorithm uses column generation to search the space of possible clauses, which is exponential in size. The training accuracy of the BRCG model is 0.71, and the test accuracy is 0.69. 

2. **LogisticRuleRegression (LRR):** The LogisticRuleRegression (LRR) algorithm fits a logistic regression model using rule-based features and is capable of generating interpretable rules that can explain the decision-making process. It uses column generation to generate promising candidates from the space of all possible rules, including unbinarized ordinal features in addition to rules. The complexity parameters lambda0 and lambda1 penalize the number of rules included in the model and the number of conditions in each rule. The training accuracy of the LRR model is 0.74, and the test accuracy is 0.72. 

3. **Generalized Linear Rule Model (GLRM):** The Generalized Linear Rule Model (GLRM) algorithm extends the rule-based explanatory model by allowing for more general linear relationships. It produces models that are weighted combinations of rules, which give insight into loan repayment predictability. The algorithm also has the option of combining rules with linear terms. The model output predicts the probability of repaying on time (Y=1). The training accuracy of the CEMExplainer algorithm is 0.73, and the test accuracy is 0.72.

The use of multiple explanatory models allows us to cross-verify the explanations and ensure their reliability. This is particularly important in high-stakes contexts like credit approval, where understanding the reasons behind decisions can have significant impacts. 

Our system also supports XAI algorithms for explaining credit approval decisions to loan officers and bank customers. The algorithms we use are:

* **ProtodashExplainer:** This XAI tool is for producing explanations for Loan Officers/employees. The ProtodashExplainer algorithm works with an existing predictive model to show how the customer compares to others who have similar profiles and had similar repayment records to the model's prediction for the current customer. It helps to evaluate and predict the applicant’s risk. The method takes as input a datapoint (or group of datapoints) that we want to explain with respect to instances in a training set belonging to the same feature space. The method then tries to minimize the maximum mean discrepancy (MMD metric) between the datapoints we want to explain and a prespecified number of instances from the training set that it will select.

* **CEMExplainer:** This XAI tool is for Applicants/bank customers. The CEMExplainer algorithm sits on top of an existing predictive model and helps detect both the features that a bank customer could improve and also further detects the features that will increase the likelihood of approval and those that are within reach for the customer. It provides insight into what – if their application was rejected – they can improve in order to increase the likelihood it will be accepted in the future.

In addition to these XAI tools, we use GitHub to track and audit the development of our AI system. Every change is logged and timestamped to ensure transparency and accountability. We also keep third-party tools up to date and ensure their security to maintain the system's high standards and comply with all relevant regulations.

## Lifecycle for Evaluating AI System Performance in the Post-Market Phase

The AI-based credit approval system is designed to continually learn and improve from the new data it encounters post-deployment. To ensure that the system maintains its performance and complies with the EU AI Act requirements, a lifecycle has been established to evaluate its performance in the post-market phase. 

The lifecycle comprises three stages: monitoring, testing, and reporting.

**Monitoring.** During the monitoring phase, we gather data on the system's real-world performance. This includes system inputs and outputs, user feedback, and any errors or anomalies that might occur.

**Testing.** In the testing phase, the AI system is subjected to various scenarios to assess its robustness, accuracy, and performance. This includes testing the system with new, unseen data to ensure it can generalize well beyond the initial training set.

**Reporting.** The reporting phase involves generating comprehensive reports on the system's performance, including any issues identified during monitoring and testing, as well as any corrective actions taken. These reports are then submitted to the relevant regulatory authorities, ensuring we maintain full transparency and compliance with regulatory requirements.

The evaluation of the AI system's post-market performance is guided by ISO/IEC 25012, which provides guidelines for software quality measurement, and ISO/IEC 27001, which provides guidelines for information security management. Adherence to these standards ensures the system maintains regulatory compliance and delivers reliable performance.

## Post-Market Monitoring Plan

Ensuring the ongoing performance and compliance of our AI-based credit approval system post-deployment requires a robust post-market monitoring plan. This plan comprises several components, which are designed to collect and analyze performance data, assess the system's ongoing performance, and facilitate reporting and documentation.

### Data Collection

Data collection is a continuous process that captures the system's inputs and outputs, user feedback, error reports, and other relevant data. This real-world data is invaluable for understanding the system's performance in various scenarios and identifying areas for improvement.

### Data Analysis

Once collected, the data is systematically analyzed to identify any potential issues, such as declining performance, bias in decision-making, or security vulnerabilities. By observing trends and patterns in the data, we can proactively address potential issues before they impact the system's performance.

### Regular Testing

The system is regularly tested using the newly collected data. These tests assess the system's accuracy, fairness, robustness, and security. If there are significant changes in the credit market or regulatory landscape, additional tests are conducted to ensure the system's performance remains consistent.

### Incident Response

In the event of any significant issues or incidents, a response plan is activated. This plan includes immediate measures to mitigate any harmful effects, an investigation into the cause of the incident, and long-term measures to prevent similar incidents in the future.

### Reporting and Documentation

Regular reports are generated detailing the system's performance, any identified issues or incidents, and any corrective actions taken. These reports are shared with relevant regulatory bodies as required. All activities related to post-market monitoring are thoroughly documented, creating a comprehensive record of the system's performance and compliance with regulations.

### Updates and Maintenance

Based on the findings from data analysis and regular testing, the AI system will be periodically updated and maintained. These updates might include changes to the machine learning model, the data processing methods, or the user interface.

### Stakeholder Feedback

Feedback from all stakeholders, including end-users, is continuously collected and considered when making improvements to the system. This feedback provides valuable insights into the system's performance from different perspectives.

By implementing this post-market monitoring plan, we aim to ensure that our AI-based credit approval system continues to deliver reliable performance and meets the high standards set by the EU AI Act. We believe that through constant monitoring, testing, and improvement, we can provide a system that is not only effective but also trustworthy and transparent.