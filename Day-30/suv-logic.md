# Day 30 – Logic and Explanations (Part A + Part B)

## Dataset
The dataset used is `suv_data-2.csv`.

Columns available in the dataset:
- User ID
- Gender
- Age
- EstimatedSalary
- Purchased

The assignment asks for an end-to-end Logistic Regression pipeline using Pandas, preprocessing, model training, and evaluation.

---

## Part A – Concept Application

### 1. Data Loading and Exploration
The dataset is loaded using Pandas with `read_csv()`.

Then we display:
- first 5 rows using `head()`
- shape using `df.shape`
- column names using `df.columns`
- data types using `df.dtypes`
- missing values using `df.isnull().sum()`

This verifies that the file is loaded properly and helps identify preprocessing needs.

### 2. Data Preprocessing
The preprocessing steps are kept simple:

- Remove `User ID` because it does not help prediction.
- Encode `Gender` into numeric form:
  - Male → 1
  - Female → 0
- Fill missing values:
  - numeric columns → median
  - categorical columns → mode

Relevant features selected:
- `Age`
- `EstimatedSalary`

Target selected:
- `Purchased`

So:
- `X = [Age, EstimatedSalary]`
- `y = Purchased`

### 3. Train-Test Split
The assignment explicitly asks for train-test split, starting with 80/20.[file:526]

The code uses:
- 80/20
- 75/25
- 70/30

This helps compare performance across splits.

### 4. Feature Scaling
`Age` and `EstimatedSalary` are on different scales.

For example:
- Age is roughly in tens
- EstimatedSalary is in thousands

So `StandardScaler` is applied before Logistic Regression.  
This helps the model train more effectively and keeps both features comparable.

### 5. Model Training
A Logistic Regression model is created using `LogisticRegression()` and fitted on the scaled training set.

This is the core classifier for predicting:
- 0 → Not Purchased
- 1 → Purchased

---

## Part B – Stretch Problem

### 1. Model Evaluation
After training, predictions are made on the test set.

Two required metrics are computed:
- Accuracy
- Confusion Matrix

Accuracy tells us the percentage of correct predictions.  
Confusion matrix gives class-wise prediction details.

### 2. Visualization
Since there are only two numeric features, a decision boundary plot is generated.

Axes:
- X-axis → Age
- Y-axis → Estimated Salary

This plot helps visualize how the classifier separates buyers and non-buyers.

### 3. Improvement by Test Size Comparison
The assignment asks to compare:
- 80/20
- 75/25
- 70/30.[file:526]

The script stores all results and selects the best one based on accuracy.

This is a simple improvement step and keeps the solution effective without making it unnecessarily complex.

---

## Expected Outputs
Running the script should produce:
- Data preview
- Shape and columns
- Missing value summary
- Accuracy for 3 train-test splits
- Confusion matrix for each split
- One decision boundary plot saved as `decision_boundary.png`
- Final comparison table