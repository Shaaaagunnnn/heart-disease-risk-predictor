# Heart Disease Risk Predictor

### A Probability and Statistics Approach

A student project combining statistical analysis with a Logistic
Regression model to estimate the probability of heart disease from
selected clinical attributes. A Gradio web interface lets users enter
patient information and view the model's estimated probability and a
project-defined category.

> **Disclaimer:** This application is for educational and academic
> demonstration only. It is not a medical device, does not provide a
> diagnosis, and must not be used to make healthcare decisions. Consult
> a qualified healthcare professional for medical concerns.

## Project Overview

The project has two main parts: 1. **Statistical analysis:** summarize
the data and investigate associations between variables and disease
status. 2. **Predictive application:** use a trained Logistic Regression
model in a Gradio interface.

## Objectives

-   Clean and summarize the dataset.
-   Examine normality of selected numerical variables.
-   Investigate numerical associations with disease status.
-   Compare numerical variables between disease and no-disease groups.
-   Test associations between categorical variables and disease status.
-   Train and evaluate a binary Logistic Regression classifier.
-   Build and deploy an interactive academic demonstration.

## Dataset

**Dataset:** UCI Machine Learning Repository --- Heart Disease,
Cleveland subset\
**Source file:** `processed.cleveland.data`\
**Records:** 303\
**Original attributes:** 14

The original target contains values `0` through `4`. It was converted to
a binary outcome: - `0` → no disease (`disease = 0`) - `1`, `2`, `3`, or
`4` → disease present (`disease = 1`)

Class counts after conversion: - No disease: **164** - Disease present:
**139**

### Attributes

  Attribute    Meaning
  ------------ -------------------------------------------------------
  `age`        Age in years
  `sex`        0 = female, 1 = male
  `cp`         Chest-pain type; Cleveland codes 1--4
  `trestbps`   Resting blood pressure (mm Hg)
  `chol`       Serum cholesterol (mg/dl)
  `fbs`        Fasting blood sugar indicator
  `restecg`    Resting ECG category
  `thalach`    Maximum heart rate achieved
  `exang`      Exercise-induced angina indicator
  `oldpeak`    ST depression induced by exercise relative to rest
  `slope`      Peak exercise ST-segment slope; codes 1--3
  `ca`         Number of major vessels (0--3)
  `thal`       Thalassemia-related category; Cleveland codes 3, 6, 7
  `target`     Original UCI outcome, converted into `disease`

The application uses the same numeric encodings as the training data. In
particular, `cp` is coded 1--4, `slope` 1--3, and `thal` uses 3, 6, and
7.

## Data Preparation

Six missing values were identified: - `ca`: 4 missing values - `thal`: 2
missing values

Missing `ca` values were replaced with the **median**. Missing `thal`
values were replaced with the **mode**. A binary `disease` column was
created from the original `target`. The 13 input attributes were used as
model features; `target` and `disease` were excluded from the feature
matrix.

## Statistical Methods

### 1. Descriptive statistics

Mean, standard deviation, median, minimum, and maximum were calculated
for `age`, `trestbps`, `chol`, `thalach`, and `oldpeak`. Group means
were also compared between disease and no-disease groups.

### 2. Shapiro--Wilk normality test

The test assessed normality for the five numerical variables. All five
reported p-values were below 0.05, so the normality null hypothesis was
rejected for each variable.

### 3. Spearman rank correlation

Spearman correlation examined associations between numerical variables
and the binary disease outcome.

  Variable                   Spearman rho      p-value
  ------------------------ -------------- ------------
  Age                             +0.2367     0.000032
  Resting blood pressure          +0.1282     0.025684
  Cholesterol                     +0.1211     0.035082
  Maximum heart rate              -0.4235   \<0.000001
  Oldpeak                         +0.4134   \<0.000001

Maximum heart rate had the strongest negative association among these
variables, while oldpeak had the strongest positive association. These
are associations, not proof of causation.

### 4. Mann--Whitney U test

This non-parametric test compared numerical-variable distributions
between the two independent disease groups. All five tested variables
had p-values below 0.05.

### 5. Chi-square test of independence

This test examined associations between disease status and categorical
variables. Seven of the eight tested variables were statistically
significant at the 0.05 level. Fasting blood sugar (`fbs`) was not
significant in this analysis (p = 0.781273).

Statistical significance does not by itself imply causation or clinical
importance.

## Machine Learning Model

**Algorithm:** Logistic Regression\
**Task:** Binary classification\
**Target:** `disease` (0 = no disease, 1 = disease)

### Training procedure

1.  Used the 13 clinical/demographic attributes as features.
2.  Split the data into **80% training and 20% testing**.
3.  Used `random_state=42` for reproducibility.
4.  Used `stratify=y` to maintain approximately the same class
    proportions in both splits.
5.  Fitted `StandardScaler` on the training data only.
6.  Used the fitted scaler to transform training and test features.
7.  Trained Logistic Regression with `max_iter=1000`.

The scaler was fitted only on the training set to avoid information
leakage from the test set.

### Why Logistic Regression?

The outcome is binary. Logistic Regression estimates the probability of
the positive class and is appropriate for this binary prediction task.

## Reported Results

The reported evaluation from the project notebook was:

  Metric                          Result
  ------------------------- ------------
  Test accuracy               **86.89%**
  ROC-AUC                     **0.9513**
  Disease-class precision           0.81
  Disease-class recall              0.93
  Disease-class F1-score            0.87

Reported confusion matrix:

``` text
                 Predicted
                 No disease   Disease
Actual no disease     27          6
Actual disease         2         26
```

This corresponds to 27 true negatives, 6 false positives, 2 false
negatives, and 26 true positives.

These metrics are from one stratified 80/20 split of a relatively small
dataset. They should not be interpreted as proof of clinical performance
or expected performance on a different population.

## Application Workflow

The application is built with **Gradio**. Users enter 13 attributes
using numeric fields and dropdowns. At prediction time, the app:

1.  Collects the entered values.
2.  Places them in a Pandas DataFrame with the expected feature columns.
3.  Applies the saved `StandardScaler`.
4.  Passes the scaled values to the saved Logistic Regression model.
5.  Uses `predict_proba()` to obtain the estimated probability of the
    disease class.
6.  Converts the probability to a percentage and displays a
    project-defined category.

The category bands are: - **Low:** below 30% - **Moderate:** 30% to
below 60% - **High:** 60% or above

These boundaries are project-defined only and are **not clinically
validated risk thresholds**. The output is a model estimate, not a
medical diagnosis.

## Project Files

The GitHub repository is expected to contain:

``` text
heart-disease-risk-predictor/
├── app.py              # Gradio interface and prediction function
├── model.pkl           # Serialized trained Logistic Regression model
├── scaler.pkl          # Serialized fitted StandardScaler
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

The app does not need the CSV at runtime because the model has already
been trained.

## Run Locally

Keep `app.py`, `model.pkl`, `scaler.pkl`, and `requirements.txt` in the
same directory.

### 1. Install dependencies

``` bash
pip install -r requirements.txt
```

### 2. Start the application

``` bash
python app.py
```

Gradio will print a local URL in the terminal. Open it in a browser.

> Only load pickle/joblib model files from sources you trust. Serialized
> Python objects can execute code when loaded.

## Deployment

The project was deployed as a **Render Web Service** connected to the
GitHub repository.

-   **Runtime:** Python 3
-   **Build command:** `pip install -r requirements.txt`
-   **Start command:** `python app.py`
-   **Port binding:** the app reads Render's `PORT` environment variable
    and binds to `0.0.0.0`

Render installs the dependencies, runs `app.py`, and serves the Gradio
interface through the service URL. Free-tier services may sleep when
idle and can take time to wake up.

## Limitations

-   The dataset contains only 303 records.
-   Evaluation used one train/test split; cross-validation and external
    validation were not reported.
-   The dataset is relatively old and may not represent current or
    diverse patient populations.
-   Associations do not establish causation.
-   Model probabilities may not be calibrated and should not be
    interpreted as a person's true clinical risk without further
    validation.
-   The Low/Moderate/High categories are project-defined, not medically
    validated.
-   This is an educational prototype, not a tool for diagnosis,
    treatment, or emergency decisions.

## Acknowledgements

-   **Dataset:** UCI Machine Learning Repository, Heart Disease dataset
    (Cleveland subset).
-   **Libraries:** Pandas, NumPy, scikit-learn, Joblib, and Gradio.
-   **Hosting:** GitHub and Render.

------------------------------------------------------------------------

**Project title:** *Predicting Heart Disease Risk: A Probability and
Statistics Approach*
