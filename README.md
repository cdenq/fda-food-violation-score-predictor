<!-- Back to top -->
<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="_">
    <img src="repo_assets/header.jpg" alt="Logo" width="250" height="100">
  </a>

  <h3 align="center">FDA Noncompliance Predictor</h3>

  <p align="center">
    Logistic regression model that predicts noncompliance to 0.98 testing precision. Deployed <a href="https://cdenq-fda-food-violation-score-predictor-app-bimc4d.streamlit.app/">here</a>.
    <br>
    <br>
    <a href="https://github.com/cdenq/">GitHub Home</a>
    ·
    <a href="https://github.com/cdenq/fda-food-violation-score-predictor/issues">Report Bug </a>
    ·
    <a href="https://github.com/cdenq/my-directory">Other Projects </a>
  </p>
</div>

<!-- Table of Contents -->
# Table of Contents
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#live-demo">Live Demo</a>
    </li>
    <li>
      <a href="#exe-sum">Executive Summary</a>
    </li>
    <li>
      <a href="#started">Getting Started</a>
      <ul>
        <li><a href="#started-live">Live Link</a></li>
        <li><a href="#started-setup">Setup</a></li>
        <li><a href="#started-directory">Directory</a></li>
      </ul>
    </li>
    <li>
      <a href="#about">About The Project</a>
      <ul>
        <li><a href="#about-ps">Problem Statement</a></li>
        <li><a href="#about-bw">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#process">Process</a>
      <ul>
        <li><a href="#process-setup">Data Sourcing, Cleaning</a></li>
        <li><a href="#process-eda">EDA</a></li>
        <li><a href="#process-work">Modeling</a></li>
        <li><a href="#process-eval">Evaluation</a></li>
      </ul>
    </li>
        <li>
      <a href="#results">Results</a>
      <ul>
        <li><a href="#results-conclusion">Conclusions</a></li>
        <li><a href="#results-future">Future Considerations</a></li>
        <li><a href="#results-note">Ending Note</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DEMO -->
<a name="live-demo"></a>
# Live Demo
<img src="repo_assets/demo.gif">

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Header -->
<a name="exe-sum"></a>
# Executive Summary

In an effort to help allocate government resources more effectively, this project seeks to create a logistic regression to predict which food establishments are at risk of safety noncompliance. Specifically, this means receiving a "serious violation score", which is a inspection penalty score above 10.

This project provides a starting point for predicting noncompliance, with mentions of future improvements to help refine performance. **The target optimization metric was precision.** A high precision score indicates that the model reduces the amount of false positives it makes; in context, this means that the model does not falsely penalize compliant restaurants as noncompliant.

In its current state, this model achieves a **noncompliance prediction to a 98% precision** and an overall ability to **differentiate between noncompliance and compliance 82% of the time** (AUC score of 0.82). These scores are important in regulatory contexts where false positives would be extremely costly. The model also suggested that the **`average previous penalty scores` and `whether the inspection took place in Winter` had the highest impact on noncompliance predictions**. 

Depending on business needs, this model can be easily retrained to optimize for other metrics, like the F1-score for example, if stakeholders prefer stronger overall performance.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ABOUT THE PROJECT -->
<a name="started"></a>
# Getting Started

<a name="started-live"></a>
## Live Link
You can visit the deployed Streamlit app [here](https://cdenq-fda-food-violation-score-predictor-app-bimc4d.streamlit.app/).

<a name="started-setup"></a>
## Setup

Clone repo
```sh
git clone https://github.com/cdenq/fda-food-violation-score-predictor
```

Setup virtual environment
```sh
conda create --name fda_noncompliance_predictor
source activate fda_noncompliance_predictor
pip install -r requirements.txt
pip install --upgrade dataframe_image # to fix Chrome update issues
```

Navigate to streamlit folder and run app
```sh
cd fda-food-violation-score-predictor
streamlit run app.py
```

NOTE: The raw data is not pushed with this repo. While you are able to deploy this app locally, you will not be able to re-run `eda.ipynb` or `model.ipynb` due to missing data sources. Please <a href="#contact">contact me</a> for more information if you wish to rerun these files.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="started-directory"></a>
## Directory

```bash
fda-food-violation-score-predictor
├── data
│   ├── prepped             # contains the prepped data
│   │   └── *.csv
│   └── raw                 # contains the raw data
│       │                   # note: the raw data is not uploaded in this repo
│       └── *.csv
├── dev
│   ├── eda.ipynb           # code used for EDA
│   └── model.ipynb         # code used for modeling
├── modules
│   ├── __init__.py         # used for backend imports
│   ├── eda.py              # helper code for EDA
│   ├── evaluate.py         # helper code for modeling
│   ├── helper.py           # helper code for general programming
│   └── imports.py          # code for imports & global+default variables 
├── outputs
│   ├── eda_outputs         # contains EDA visualizations and files 
│   │   └── *.csv
│   ├── model_outputs       # contains modeling visualizations and files
│   │   └── *.csv
│   └── saved_models        # contains saved models
│       └── *.pkl
├── repo_assets
│   ├── demo.gif            # the README's gif
│   └── header.jpg          # the README's header image
├── .gitignore
├── app.py                  # the deployed app's homepage
├── LICENSE
├── README.md
└── requirements.txt        # the required pip installations
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ABOUT THE PROJECT -->
<a name="about"></a>
# About

<a name="about-ps"></a>
## Problem Statement

The FDA is interested in efficiently allocating resources to food inspection cycles. Certain restaurants are more likely to violate health regulations than others, and thus, identifying such restaurants could help save time and money in the form of remediary or preemptive measures. Given raw data, this project aims to create a model to accurately predict whether a restaurant will receive a "serious violation score" (score of 10 or above), or **noncompliance**.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="about-bw"></a>
## Built With
| **Backend** | **Database** |  **Frontend** | **Deployment**
| - | - | - | - |
| ![Python](https://img.shields.io/badge/Python-blue?logo=python&logoColor=white) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange?logo=scikitlearn&logoColor=white) ![NLTK](https://img.shields.io/badge/NLTK-blue?logo=NLTK&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-black?logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-blue?logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-black?logo=matplotlib&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-blue?logo=seaborn&logoColor=white) | ![Local](https://img.shields.io/badge/Local-white?logo=microsoftexcel&logoColor=black) | ![Streamlit](https://img.shields.io/badge/Streamlit-crimson?logo=streamlit&logoColor=white) | ![Streamlit Community](https://img.shields.io/badge/Streamlit--Community-crimson?logo=streamlit&logoColor=white)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ABOUT THE PROJECT -->
<a name="process"></a>
# Process

<a name="process-setup"></a>
## Data Collection and Cleaning

Data is provided by the client. The files were checked for any formatting or value errors. One major task was type-parsing, as the data types for certain fields were presented as strings but had to be parsed as lists and date-times.

Feature engineering was heavily used in this process to augment the dataset. Here are the list of features that were extracted from the raw data:
- inspection duration in days
- inspection season
- penalty score trend
- percentage of non-positive reviews
- number of cuisine labels
- whether the cuisine was ethnic (client paramaters)
- whether the cuisine was asian (client paramaters)

Specific details are found in the `eda.ipynb` development file, located in `dev/`. See <a href="#started-directory">Directory</a> for how to locate this file.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="process-eda"></a>
## EDA

At a high level, EDA was performed to check the data quality, understand data shape, and properly inform potential feature engineering.

Outliers and multicollinearity were also checked, and in cases where violated, prompted tune-ups in the form of scaling, dropping, and combining data. 

For example, the graph below shows the presence of outliers in the `inspection_duration` variable. This type of EDA prompted the use of robust scaling, which is effective at reducing outlier impact without directly removing them from the dataset.

<img src="outputs/eda_outputs/boxplot_inspection_duration.png">

Specific details are found in the `eda.ipynb` development file, located in `dev/`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="process-work"></a>
## Modeling (& Evaluation on Validation Data)

Data preprocessing involved splitting the dataset into train-val-test segments, following a 75-10-15 split. Then, the training data copied into 3 instances, with each formated a different way: MinMax scalar, Standard scalar, and Robust scalar. This effectively produced three variants of the logistic regression model for which to compare. Only the numerical data was scaled; the categorical was left untouched.

One benefit of the logistic regression model was its compute speed, and that leveraged in the training phase. Specifically, the model was trained using a grid-search technique with a 5-fold cross-validation to reduce overfitting. Coefficient tables, confusion matrices, classification reports, and ROC curves were generated for each of the three variants to visualize performance. From modeling, the average previous penalty scores and whether the inspection took place in the Winter season tended to have the highest impact on prediction.

The performance is summarized below.

<img src="outputs/model_outputs/bar_comparisons_model_comparisons.png">
<img src="outputs/model_outputs/total_model_performance.png">

From validation evaluation, the MinMax scalar-variant of the logistic model ended up performing the best on precision but slighty worse in all other metrics. Because precision was our target metric, the MinMax model was selected for testing evaluation. Worth noting, there was little difference in the effect of Standard scaling versus Robust scaling in terms of model performance.

Specific details—including insights from the coefficient tables, confusion matrices, etc.—are found in the `model.ipynb` development file, located in `dev/`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="process-eval"></a>
## Evaluation (on Testing Data)

Deploying the MinMax model onto our testing data, leads to the following results. Note: the blue and orange bars below represent the same model, just evaluated with the validation and testing data respectively.

<img src="outputs/model_outputs/bar_comparisons_val_vs._test_comparison.png">
<img src="outputs/model_outputs/final_model_performance.png">

As shown in the graph above, the MinMax model scored extremely well in precision. The model also scored slightly lower in all the other metrics, which suggests that the model was overfit during the training phase.

In general, this MinMax model is a great overall starting point for predicting noncompliance, as it correctly predicts noncompliance 98% of the time (precision score) and overall can differentiate between noncompliant and compliant cases 82% of the time (AUC score). 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONCLUSIONS -->
<a name="results"></a>
# Results

<a name="results-conclusion"></a>
### Conclusion

The logistic model serves as a signficiant improvement over the baseline format. When differentiating between noncompliance and complaince, a naive guesser would get this right 50% of the time, while the model would get this right 82% of the time (AUC score). When it comes to predicting noncompliance, a naive guesser that bases its guess on the most common class would get this right 67% of the time, while the model will correctly predict this 98% of the time (precision score).

For this reason, this model would be especially effective in informing policy or noncompliance actions. 

One method of deployment would be to use this model in an **ensemble system, specifically as a double-checker for a another, more well-balanced model** (SVM, random forest, DL NN, etc.). Given the regulatory context of this project, a false positive prediction that erronously penalizes a compliant restaurant for noncompliance would be disastrous. By having this logistic regression model quality-checking another model's prediction, we can drastically reduce the number of false positives.

Another method of deployment would be to use this model "as-is". While not perfect, this model significantly outperforms a naive guesser, which could be sufficient for certain business contexts. Depending on stakeholders' needs, this model can also be quickly retrained to prioritize other metrics, like the F1-score, if a more holistic/well-rounded model is prefered over a precision-optimized model.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="results-future"></a>
### Future Considerations

Below are suggested ways for future improvements to the model. 

#### Augmenting Dataset:
Including more granular data...
- on past violations (e.g., types of violations, frequency, severity)
- restaurant size, location
- food suppliers, region where food was grown
- staff training records (e.g. how much food handling training does each staff member need)
- number of customers on daily basis
- weight of garbage generated on daily basis

This information might be obtained from health inspection records, business registration databases, or directly from establishments through surveys or compliance reporting

#### Augmenting Feature Engineering & Model Training:
- Focusing on regularization (higher penalty coefficients) would help reduce the slight overfitting problem with the model.
- Training the model to optimize for the F1-score would ensure an overall increase in performance (but at the cost of the high precision score) 

#### Augmenting Modeling Approaches:
- Directly predicting a restaurant’s violation score, rather than a binary outcome, could provide a more nuanced view of risk, allowing for more tailored inspection approaches. For example, this project used a Log Reg model, but if predicting a numerical result, we would use a multi-linear regression (MLR) model. This approach would, in turn, could better allocate inspection resources by identifying not just whether a violation might occur, but also the likely severity of the violation. A "11" would need less of a response than a "100", despite both receiving the same violation.

- Embed this model into an ensemble system that uses the results of multiple models to generate a final prediction.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="results-note"></a>
### Ending Note about AI in Social Contexts

It is important to not over-rely on ML predictions, as there are potentially overlooked limitations and biases in the datasets used to train them. For example, datasets may not capture all relevant data, may be collected under atypical conditions, or contain inherent biases, leading to skewed model predictions. This can result in unjust and costly outcomes, especially in the regulatory space.

- Example: A model might wrongly associate a particular ethnicity of restaurants with poor food quality due to an external factor like a negligent local supplier (which is not captured in the dataset). This could lead to unfair predictions about these establishments.

- Example: Data collected during unusual circumstances, like pest infestations in neighboring establishments, can inaccurately influence the model's predictions for a restaurant.

- Example: Historical biases in data, such as inspectors disproportionately downrating minority-owned restaurants, can perpetuate systemic bias in model predictions, creating a cycle of unjustified scrutiny and violations for these businesses.

**In general, while models may provide stronger predictive powers that lead to better efficiency, we must always ensure that the models are developed transparently and ethically to ensure that the results do not come at the cost of others. Careful consideration of the results, continued iterative improvements to modeling, and diligent scrutiny of the provided dataset are critical.**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
<a name="contact"></a>
# Contact

To directly message me or setup a time to chat virtually, see my [LinkedIn](https://www.linkedin.com/in/christopherdenq/) and [Calendly](https://calendly.com/christopherkd/coffee-chats) links.

If you're curious about more projects, check out my [website](https://cdenq.github.io/) or [GitHub](https://github.com/cdenq).

<p align="right">(<a href="#readme-top">back to top</a>)</p>