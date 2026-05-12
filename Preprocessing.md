# What You Can Reuse From Last Year’s Notebook

Your old CPC152 notebook structure is still useful, but many parts must be adapted because the old notebook was for **wine quality**, while this project uses the **dermatology** dataset.

## Reuse Directly

- **Library imports**
  - `pandas`
  - `numpy`
  - `matplotlib.pyplot`
  - `seaborn`
  - `train_test_split`
  - `StandardScaler`
  - `SelectKBest`, `f_classif`
  - model/evaluation imports if Part 2 continues into modeling

- **Basic EDA structure**
  - `df.head()`
  - `df.shape`
  - `df.columns`
  - `df.info()`
  - `df.describe()`
  - class distribution plot
  - missing value check
  - duplicate row check

- **Train/validation/test splitting idea**
  - CPC251 template specifically says split into **training, validation, and test sets**, not only 80/20 train-test.
  - So reuse the idea, but modify it to something like:
    - 70% train
    - 15% validation
    - 15% test

- **Feature scaling idea**
  - Reuse `StandardScaler`, especially because later models like **SVM** are sensitive to scale.

- **Feature selection idea**
  - Reuse `SelectKBest(score_func=f_classif)`.
  - This is suitable because dermatology is a **classification** dataset.

# What You Should Change

## 1. Dataset Loading

Your current cell:

```python
df = pd.read_csv("dermatology.csv")
```

should be changed because the dermatology CSV has **no header row** and missing age values are marked with `?`.

You should load it with:

- `header=None`
- custom column names from the README
- `na_values='?'`

The dataset has:

- **366 rows**
- **35 columns**
  - 34 features
  - 1 target class column

Important: the README says 34 attributes, but the CSV includes the class label as the final column, so the actual dataframe should have 35 columns.

## 2. Replace Wine-Specific Columns

Remove or rewrite anything using:

- `quality`
- `good wine`
- wine feature names like:
  - `alcohol`
  - `sulphates`
  - `volatile acidity`
  - `density`
  - etc.

For dermatology, the target column should be something like:

```python
class
```

or

```python
disease_class
```

The class labels are:

- **1**: psoriasis
- **2**: seboreic dermatitis
- **3**: lichen planus
- **4**: pityriasis rosea
- **5**: cronic dermatitis
- **6**: pityriasis rubra pilaris

This should stay as a **multi-class classification** problem. Do **not** convert it into binary classification like the wine notebook did.

# What You Should Add for Data Preprocessing

## 1. Column Naming

Add clear column names based on the README.

This is important because the raw CSV has no header, and without column names your notebook will show columns as `0, 1, 2, ...`.

You should include names such as:

- `erythema`
- `scaling`
- `definite_borders`
- `itching`
- `koebner_phenomenon`
- `family_history`
- `age`
- `class`

This makes the notebook much more readable and rubric-friendly.

## 2. Missing Value Handling

The README says:

- **8 missing values**
- all in the **age** attribute
- represented by `?`

After loading with `na_values='?'`, check:

```python
df.isnull().sum()
```

Then handle age missing values.

Recommended approach:

- Use **median imputation** for `age`.

Reason:

- `age` is numeric.
- Median is robust and simple.
- Only 8 out of 366 rows are missing, so this is reasonable.

You can explain:

> The missing values were found only in the age column. Since age is numeric and only a small number of values are missing, the median age was used to impute the missing values.

## 3. Duplicate Checking

Keep your duplicate-checking section.

For this dataset, I confirmed:

- **0 duplicate rows**

So you can keep the code but the result should say no duplicates were removed.

## 4. Data Type Conversion

Because `?` appears in `age`, `age` may load as `float64` after converting missing values.

That is okay. But after imputation, you can either:

- keep `age` as float, or
- convert to integer if desired.

Better to keep it numeric and avoid unnecessary conversion.

## 5. Outlier Handling

Be careful here.

Your old notebook removes outliers using IQR across all numeric predictors. For dermatology, **do not blindly reuse that removal method**.

Reason:

- Most dermatology features are ordinal clinical scores from **0 to 3**.
- Values like 3 are not “outliers”; they are valid medical severity levels.
- Removing rows based on IQR could delete valid disease cases and distort the class distribution.
- The dataset is small, only 366 rows, so aggressive removal is risky.

Recommended:

- Do **not remove outliers** for the 0-3 dermatology features.
- Only inspect `age` for unusual values.
- If age values are reasonable, keep them.

You can still include an outlier section, but write that no outlier removal was performed because the feature values are clinically valid ordinal scores.

## 6. Label Encoding

For this dataset, label encoding is mostly **not needed**.

Reason:

- All features are already numeric.
- The target class is already encoded as integers `1` to `6`.
- No text categories are present.

But you should still mention this in the preprocessing description:

> Label encoding was not required because all attributes and class labels are already numerically encoded in the dataset.

Optional improvement:

- Create a dictionary to map class numbers to disease names for interpretation.
- But for modeling, keep the numeric class labels.

## 7. Feature Scaling / Standardization

This should be added after the split.

Important: fit the scaler on the **training set only**, then transform validation and test sets.

Recommended:

- Use `StandardScaler`
- Scale all feature columns before SVM
- Decision Tree does not require scaling, but using a scaled copy for models like SVM is useful

Best practice:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

Do not fit scaler before splitting, because that causes data leakage.

## 8. Train / Validation / Test Split

The CPC251 template asks for:

> Split the dataset into training, validation and test sets.

So instead of the old 80/20 split, use a 3-way split.

Recommended:

- `X`: all columns except `class`
- `y`: `class`
- first split:
  - 70% training
  - 30% temporary
- second split:
  - split temporary into 15% validation and 15% test

Use `stratify=y` because class distribution is imbalanced, especially class 6 has only 20 samples.

## 9. Class Distribution

Reuse your class distribution visualization, but update it from wine quality to dermatology class.

For example:

- show `df['class'].value_counts().sort_index()`
- plot `sns.countplot(x='class', data=df)`

Mention that the dataset is somewhat imbalanced:

- class 1: 112
- class 2: 61
- class 3: 72
- class 4: 49
- class 5: 52
- class 6: 20

This is useful for later evaluation because accuracy alone may not be enough.

# Suggested Part 1 Notebook Flow

Use this structure after `Load the Dataset`.

## 1. Exploratory Data Analysis

Keep/adapt:

- `df.head()`
- `df.shape`
- `df.columns`
- `df.info()`
- `df.describe()`
- class distribution

## 2. Data Cleaning

Include:

- check missing values
- replace `?` with `NaN` during loading
- impute missing `age` using median
- check duplicates
- explain why outlier removal is not applied aggressively

## 3. Split the Dataset

Add:

- `X = df.drop(columns=['class'])`
- `y = df['class']`
- train/validation/test split
- stratified sampling

## 4. Data Preprocessing

Add:

- standardization using `StandardScaler`
- no label encoding needed explanation
- optional: normalized/scaled dataframe preview

## 5. Feature Selection

Reuse:

- `SelectKBest`
- ANOVA F-test
- feature score table
- bar chart of feature scores

But update target/feature variables to use dermatology names.

# Main Things to Remove From Old Notebook

Remove or rewrite:

- **Binary wine classification**
  - `good wine`
  - `quality >= 7`
- **Wine target**
  - `quality`
- **Wine-specific feature names**
- **Outlier row removal using IQR across all features**
- **Repeated train-test splits for each model**
  - For CPC251 Part 1, just prepare clean `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`.
  - You can create model-specific copies later in modeling.

# Summary

You can reuse the **overall EDA, cleaning, splitting, scaling, and feature-selection structure**, but you must adapt it for a **multi-class dermatology classification dataset**.

The most important additions are:

- **load CSV with custom column names**
- **handle `?` missing values in `age`**
- **do train/validation/test split**
- **standardize after splitting**
- **avoid inappropriate outlier removal**
- **keep the target as 6 disease classes instead of binary classification**

Status: I reviewed your notebook, the CPC251 template, the dermatology CSV, and README, and mapped exactly what should be reused and changed for Part 1 preprocessing.