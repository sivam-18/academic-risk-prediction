#!/usr/bin/env python3
# Import the built-in operating system module to interact with system files and file paths
import os

# Import the NumPy library for numerical operations and handling missing values (np.nan)
import numpy as np

# Import the Pandas library for data manipulation, cleaning, and tabular data analysis
import pandas as pd

# Define a tuple of standard file locations where Google Colab or local scripts usually save/store uploaded CSV files
path_options = ("student_dataset.csv", "/content/student_dataset.csv", "/student_dataset.csv")

# Use a generator expression with next() to scan the paths in sequence and pick the first file path that physically exists on disk
DATA_PATH = next((path for path in path_options if os.path.exists(path)), None)

# Verify whether a valid file path was located during the search step
if DATA_PATH is None:
    # Stop execution and raise an explicit error message prompting the user to upload the required file if not found
    raise FileNotFoundError("Upload student_dataset.csv using the folder icon on the left, then run again.")

# Read the CSV dataset from the verified path into a Pandas DataFrame object named 'df'
df = pd.read_csv(DATA_PATH)


# ==========================================
# STEP 1: LOOK AT THE DATA (INSPECTION PHASE)
# ==========================================

# Display the total dimensions of the loaded dataset as a tuple (number of rows, number of columns)
print("Shape (rows, columns):", df.shape)

# Identify missing entries (NaN/nulls) in each column, sum them up, and display the per-column missing count
print("\nMissing values per column:\n", df.isnull().sum())

# Identify duplicate rows across all columns, sum the boolean flags, and print the total number of duplicate rows found
print("\nDuplicate rows:", df.duplicated().sum())

# Define a list containing the names of all continuous numerical score and percentage columns to be audited
score_cols = [
    "Attendance (%)",
    "Midterm_Score",
    "Final_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Participation_Score",
    "Projects_Score",
    "Total_Score",
    "math_score",
    "reading_score",
    "writing_score",
    "science_score"
]

# Before aggregating or checking values, ensure all score columns are converted to numeric type to avoid comparison errors with hidden strings
for col in score_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Print a header text announcing the summary statistics for numerical score validation
print("\nSmallest and largest value in each score column (to spot impossible values):")

# Compute the minimum ('min') and maximum ('max') values for each score column, transpose (.T) the result table for readability, and display it
print(df[score_cols].agg(["min", "max"]).T)


# ==========================================
# STEP 2: CLEAN THE DATA (PREPROCESSING PHASE)
# ==========================================

# Remove any identical duplicate rows from the DataFrame and assign the filtered dataset back to 'df'
df = df.drop_duplicates()

# Remove rows where the critical target metric 'Total_Score' is missing (NaN), keeping only rows with valid performance scores
df = df.dropna(subset=["Total_Score"])

# Iterate through each score column name defined in the 'score_cols' list to detect out-of-range anomalies
for col in score_cols:
    # Create a boolean Series evaluating to True where values are strictly less than 0 OR strictly greater than 100
    impossible = (df[col] < 0) | (df[col] > 100)

    # Check if there are one or more True flags (out-of-range values) in the boolean mask
    if impossible.sum() > 0:
        # Print a status message indicating how many invalid entries were discovered in the current column
        print(f"{col}: {impossible.sum()} impossible value(s) set to missing")

        # Replace the invalid/out-of-range entries in that column with NaN (missing value) using location-based indexers (.loc)
        df.loc[impossible, col] = np.nan

# Calculate the median value of 'Attendance (%)' (excluding NaNs) and fill any missing entries in 'Attendance (%)' with this median
df["Attendance (%)"] = df["Attendance (%)"].fillna(df["Attendance (%)"].median())

# Print the final row and column count of the DataFrame after completing all deduplication, filtering, and cleaning steps
print("\nCleaned shape:", df.shape)
# Import the Pyplot module from Matplotlib for creating figures, subplots, and customizing plot layouts
import matplotlib.pyplot as plt

# Import the Seaborn data visualization library for drawing attractive statistical graphics
import seaborn as sns

# Set the global aesthetic theme of all Seaborn visualizations to 'whitegrid' (white background with gray gridlines)
sns.set_theme(style="whitegrid")


# ==============================================================================
# SECTION 1: VISUALIZE DISTRIBUTIONS OF TOTAL SCORES AND ATTENDANCE
# ==============================================================================

# Create a figure object containing 1 row and 2 subplots (side-by-side) with a figure dimension of 12 inches wide by 4 inches high
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Draw a histogram with a Kernel Density Estimate (KDE) curve overlay for 'Total_Score' using a skyblue color on the left subplot (axes[0])
sns.histplot(df["Total_Score"], kde=True, color="skyblue", ax=axes[0])

# Set the title text for the left subplot to "Distribution of Total Scores"
axes[0].set_title("Distribution of Total Scores")

# Draw a histogram with a Kernel Density Estimate (KDE) curve overlay for 'Attendance (%)' using an orange color on the right subplot (axes[1])
sns.histplot(df["Attendance (%)"], kde=True, color="orange", ax=axes[1])

# Set the title text for the right subplot to "Distribution of Attendance (%)"
axes[1].set_title("Distribution of Attendance (%)")

# Automatically adjust subplot padding and spacing so that plot elements, titles, and labels do not overlap
plt.tight_layout()

# Render and display the side-by-side distribution plots figure on the screen
plt.show()


# ==============================================================================
# SECTION 2: ATTENDANCE VS TOTAL SCORE SCATTER PLOT WITH REGRESSION LINE
# ==============================================================================

# Create a new standalone figure for the scatter plot with dimensions of 8 inches wide by 5 inches high
plt.figure(figsize=(8, 5))

# Plot a scatter plot with a linear regression trend line; uses a random sample of up to 3000 rows (fixed random_state=42 for reproducibility) to avoid overplotting points
sns.regplot(data=df.sample(min(3000, len(df)), random_state=42), x="Attendance (%)", y="Total_Score",
            scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})

# Set the main chart title to "Attendance vs Total Score"
plt.title("Attendance vs Total Score")

# Render and display the regression plot figure on the screen
plt.show()


# ==============================================================================
# SECTION 3: CALCULATE AND INTERPRET CORRELATION COEFFICIENT
# ==============================================================================

# Compute the Pearson correlation coefficient (r) between 'Attendance (%)' and 'Total_Score' columns
r = df["Attendance (%)"].corr(df["Total_Score"])

# Print the computed correlation value formatted to 2 decimal places
print(f"Correlation (r) between Attendance and Total Score: {r:.2f}")

# Check if the absolute strength of the correlation coefficient is strictly less than 0.1
if abs(r) < 0.1:
    # Print an interpretation stating that no meaningful linear relationship exists in the dataset
    print("=> No meaningful relationship: students with high attendance do not score higher in this data.")

# Otherwise, check if the absolute strength of the correlation coefficient is strictly less than 0.3
elif abs(r) < 0.3:
    # Print an interpretation stating that the linear relationship is weak
    print("=> Weak relationship.")

# For any absolute correlation value of 0.3 or higher, execute this fallback block
else:
    # Print an interpretation stating that a moderate or strong linear relationship exists
    print("=> Moderate/strong relationship.")


# ==============================================================================
# SECTION 4: GROUP ATTENDANCE INTO BANDS AND BAR PLOT AVERAGE SCORES
# ==============================================================================

# Bin continuous 'Attendance (%)' values into categorical intervals ([0-60], (60-70], (70-80], (80-90], (90-100]) labeled with explicit group names
bands = pd.cut(df["Attendance (%)"], bins=[0, 60, 70, 80, 90, 100],
               labels=["<60", "60-70", "70-80", "80-90", "90-100"], include_lowest=True)

# Group the DataFrame by the defined attendance bands and calculate the mean 'Total_Score' for each bin (observed=False retains all categorical bins)
band_avg = df.groupby(bands, observed=False)["Total_Score"].mean()

# Print the grouped average total scores rounded to 1 decimal place in text format
print("\nAverage marks by attendance band:\n", band_avg.round(1))

# Plot a bar chart from the aggregated Series using steelblue bars, set figure size to 7x4 inches, and keep x-axis labels horizontal (rot=0)
band_avg.plot.bar(color="steelblue", figsize=(7, 4), rot=0)

# Set the chart title for the bar plot to "Average Total Score by attendance band"
plt.title("Average Total Score by attendance band")

# Label the y-axis of the plot as "Average Total Score"
plt.ylabel("Average Total Score")

# Render and display the bar chart figure on the screen
plt.show()


# ==============================================================================
# SECTION 5: HEATMAP OF CORRELATIONS ACROSS ALL NUMERIC COLUMNS
# ==============================================================================

# Create a new standalone figure for the correlation heatmap with dimensions of 11 inches wide by 8 inches high
plt.figure(figsize=(11, 8))

# Draw an annotated heatmap representing the pairwise correlation matrix of all numeric features; show values to 2 decimal places, centered at 0 with coolwarm palette
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)

# Set the main chart title to "Correlation between numeric columns"
plt.title("Correlation between numeric columns")

# Render and display the correlation matrix heatmap figure on the screen
plt.show()
# Define a custom python function named 'rule_risk' that takes two numeric arguments: 'marks' and 'attendance'
def rule_risk(marks, attendance):
    # Calculate a composite weighted risk score where lower marks contribute 60% and lower attendance contributes 40%
    score = 0.6 * (100 - marks) + 0.4 * (100 - attendance)

    # Check if the calculated risk score is greater than/equal to 40 OR if the student's marks are strictly below 50
    if score >= 40 or marks < 50:
        # Return the string classification "High" for high-risk students meeting either condition
        return "High"

    # Check if the risk score is greater than or equal to 25 for remaining students
    elif score >= 25:
        # Return the string classification "Medium" for moderate-risk students
        return "Medium"

    # Fallback return statement for students with risk scores below 25
    return "Low"

# Apply 'rule_risk' row-by-row (axis=1) using a lambda function passing 'Total_Score' and 'Attendance (%)' to populate a new 'Risk_Level' column
df["Risk_Level"] = df.apply(lambda row: rule_risk(row["Total_Score"], row["Attendance (%)"]), axis=1)

# Pass test inputs (marks = 55, attendance = 62) directly into 'rule_risk' and print the returned risk level alongside verification text
print("Check with the PDF example (attendance 62, marks 55):", rule_risk(55, 62))

# Print a section header text indicating that value counts per risk level category will follow
print("\nHow many students in each risk level:")

# Count the frequency of occurrence for each distinct risk category in 'Risk_Level' and print the results
print(df["Risk_Level"].value_counts())

# Draw a categorical count bar plot of 'Risk_Level' ordering bars as ["Low", "Medium", "High"], mapped to custom colors ("seagreen", "gold", "crimson")
sns.countplot(data=df, x="Risk_Level", hue="Risk_Level", order=["Low", "Medium", "High"], legend=False,
              palette={"Low": "seagreen", "Medium": "gold", "High": "crimson"})

# Set the title text of the current Seaborn count plot figure to "Number of students in each risk level"
plt.title("Number of students in each risk level")

# Render and display the count plot figure on the screen
plt.show()
# Import the function to split datasets into random train and test subsets
from sklearn.model_selection import train_test_split

# Import modules to scale numerical features and one-hot encode categorical features
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import ColumnTransformer to apply different preprocessing pipelines to different columns
from sklearn.compose import ColumnTransformer

# Import Pipeline to sequentially apply a list of transforms and a final estimator
from sklearn.pipeline import Pipeline

# Import SimpleImputer to fill in missing values (NaNs) in the dataset
from sklearn.impute import SimpleImputer


# ==============================================================================
# SECTION 1: FEATURE SELECTION & DROP LEAKAGE / IDENTIFIER COLUMNS
# ==============================================================================

# Define a list of column names that should not be used as input features for training
drop_cols = [
    "Student_ID", "First_Name", "Last_Name", "Email",   # Identifiers (non-predictive unique identifiers)
    "Gender",                                            # Sensitive attribute excluded for algorithmic fairness
    "Total_Score", "Grade", "Final_Score",              # Outcome columns (causes data leakage since risk target is derived from these)
    "Risk_Level"                                         # The target variable label itself
]

# Use a list comprehension to select all DataFrame columns except those listed in drop_cols
features = [c for c in df.columns if c not in drop_cols]

# Create the feature matrix X containing only the allowed predictor columns from the DataFrame
X = df[features]

# Create the target vector y containing the Risk_Level string categories to predict
y = df["Risk_Level"]


# ==============================================================================
# SECTION 2: STRATIFIED TRAIN / TEST SPLIT
# ==============================================================================

# Split feature matrix X and target y into 80% training and 20% testing sets, maintaining class proportions (stratify=y) and fixing random seed (random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ==============================================================================
# SECTION 3: COLUMN TRANSFORMER & PREPROCESSING PIPELINE DESIGN
# ==============================================================================

# Force 'test_preparation_course' to string type to prevent any automated pandas detection issues
X_train = X_train.copy()
X_test = X_test.copy()
X_train["test_preparation_course"] = X_train["test_preparation_course"].astype(str)

# Extract numerical column names, ensuring we exclude text categories explicitly
num_cols = X_train.select_dtypes(include="number").columns.tolist()
if "test_preparation_course" in num_cols:
    num_cols.remove("test_preparation_course")

# Extract all categorical/non-numerical column names
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
if "test_preparation_course" not in cat_cols:
    cat_cols.append("test_preparation_course")

# Define a ColumnTransformer to apply distinct preprocessing pipelines to numerical and categorical features independently
preprocessor = ColumnTransformer(transformers=[
    # Numerical pipeline: imputer replaces NaNs with column median, then StandardScaler normalizes features to zero mean and unit variance
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_cols),

    # Categorical pipeline: imputer replaces NaNs with most frequent value, then OneHotEncoder creates binary dummy variables (ignoring unseen categories in test)
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

# Print the final list of feature names used for model inputs
print("Features used:", features)

# Print the exact row counts for both the training set and testing set
print(f"Training students: {len(X_train)}   Testing students: {len(X_test)}")
# Import the DummyClassifier model from scikit-learn to serve as a simple benchmark/baseline
from sklearn.dummy import DummyClassifier

# Import the LogisticRegression algorithm for linear classification modeling
from sklearn.linear_model import LogisticRegression

# Import the DecisionTreeClassifier for tree-based non-linear classification modeling
from sklearn.tree import DecisionTreeClassifier

# Import the RandomForestClassifier ensemble algorithm combining multiple decision trees
from sklearn.ensemble import RandomForestClassifier

# Import evaluation metrics: overall accuracy, macro-averaged F1 score, and class recall
from sklearn.metrics import accuracy_score, f1_score, recall_score

# Import classification_report to generate comprehensive per-class precision, recall, and F1 metrics
from sklearn.metrics import classification_report

# Import confusion_matrix to compute the table of actual versus predicted class distributions
from sklearn.metrics import confusion_matrix

# Import ConfusionMatrixDisplay to graphically render the confusion matrix table
from sklearn.metrics import ConfusionMatrixDisplay


# ==============================================================================
# SECTION 1: MODEL INITIALIZATION & HYPERPARAMETER DEFINITION
# ==============================================================================

# Define a constant list establishing the specific ordered class labels for risk prediction
LEVELS = ["Low", "Medium", "High"]

# Define a dictionary containing candidate classification models paired with their configuration hyperparameter settings
models = {
    # Baseline dummy model that always predicts the single most frequent class label in y_train
    "Baseline": DummyClassifier(strategy="most_frequent"),

    # Logistic Regression with increased max iterations to reach convergence and class weighting to address imbalance
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),

    # Decision Tree restricted to depth 5 and minimum leaf size 20 to prevent overfitting on training data
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=42),

    # Random Forest ensemble with 200 trees, parallel execution across all CPU cores (n_jobs=-1), and balanced class weights
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=20, class_weight="balanced",
                                            random_state=42, n_jobs=-1),
}


# ==============================================================================
# SECTION 2: MODEL TRAINING, TESTING, & METRIC AGGREGATION LOOP
# ==============================================================================

# Initialize an empty dictionary to hold trained pipeline objects and their test predictions
trained = {}

# Initialize an empty list to store metric dictionary records for each evaluated model
rows = []

# Iterate over each model name and model instance key-value pair in the models dictionary
for name, model in models.items():
    # Construct an end-to-end Machine Learning Pipeline chaining the preprocessing step to the current classifier
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    # Fit the complete pipeline on training features (X_train) and training labels (y_train)
    clf.fit(X_train, y_train)

    # Generate predictions on unseen test set features (X_test) using the fitted pipeline
    y_pred = clf.predict(X_test)

    # Store the fitted pipeline and test set predictions as a tuple in the 'trained' dictionary under the model name key
    trained[name] = (clf, y_pred)

    # Append a dictionary record containing the calculated evaluation metrics for this model to the 'rows' list
    rows.append({
        # Store the current model name string
        "Model": name,

        # Calculate overall accuracy score comparing actual test labels (y_test) with predicted labels (y_pred)
        "Accuracy": accuracy_score(y_test, y_pred),

        # Calculate macro-averaged F1 score (unweighted mean of per-class F1 scores, setting zero_division=0 to prevent division errors)
        "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),

        # Extract the specific recall score for the "High" risk class (proportion of actual High-risk students correctly identified)
        "High-risk recall": recall_score(y_test, y_pred, labels=["High"], average=None, zero_division=0)[0],
    })


# ==============================================================================
# SECTION 3: MODEL COMPARISON & SELECTION
# ==============================================================================

# Convert the list of metric dictionaries into a Pandas DataFrame, set 'Model' as index, round metrics to 2 decimal places, and save to 'comparison'
comparison = pd.DataFrame(rows).set_index("Model").round(2)

# Display the metric comparison summary table on the console
print(comparison)

# Filter out the baseline model, locate the index name of the model achieving the highest Macro F1 score, and store it in 'best_name'
best_name = comparison.drop(index="Baseline")["Macro F1"].idxmax()

# Unpack the best performing pipeline object and its corresponding test predictions from the 'trained' dictionary
best_model, best_pred = trained[best_name]

# Print the name of the winning selected model surrounded by formatting text
print(f"\n---> Best model: {best_name}\n")


# ==============================================================================
# SECTION 4: DETAILED EVALUATION & CONFUSION MATRIX VISUALIZATION
# ==============================================================================

# Print a detailed classification report showing Precision, Recall, and F1-score for each class level (Low, Medium, High)
print(classification_report(y_test, best_pred, labels=LEVELS, zero_division=0))

# Compute the 3x3 confusion matrix array comparing actual test labels with best model predictions ordered by LEVELS
cm = confusion_matrix(y_test, best_pred, labels=LEVELS)

# Create a figure and axis subplot object for rendering the confusion matrix plot with 6x5 inch dimensions
fig, ax = plt.subplots(figsize=(6, 5))

# Create a ConfusionMatrixDisplay object from the matrix array, format as raw integer counts (values_format="d"), and plot on the axis using a blue color map
ConfusionMatrixDisplay(cm, display_labels=LEVELS).plot(ax=ax, cmap="Blues", values_format="d")

# Set the title text of the confusion matrix plot to reflect the winning model's name
ax.set_title(f"Confusion Matrix - {best_name}")

# Render and display the final confusion matrix graphic plot on the screen
plt.show()
import os
def make_recommendation(*args, **kwargs):
	return "Review academic progress"

# Safety check: this cell needs the earlier cells to have been run first
for needed in ["df", "features", "best_model", "make_recommendation"]:
    if needed not in globals():
        raise NameError(f"'{needed}' is missing - run all the cells above first (Runtime -> Run all).")

os.makedirs("outputs", exist_ok=True)

# Predict risk for every student
all_pred = best_model.predict(df[features])

# One personalised recommendation per student
recs = [make_recommendation(row["Attendance (%)"], row, risk)
        for (_, row), risk in zip(df.iterrows(), all_pred)]

# Build the report table (astype(str) avoids errors if a name is empty or not text)
report = pd.DataFrame({
    "Student": df["Student_ID"].astype(str) + " - " + df["First_Name"].astype(str) + " " + df["Last_Name"].astype(str),
    "Attendance (%)": df["Attendance (%)"].round(1),
    "Risk": all_pred,
    "Recommendation": recs,
})

# Most urgent first: High -> Medium -> Low, and lowest attendance first inside each level
order = {"High": 0, "Medium": 1, "Low": 2}
report = (report.assign(sort_key=report["Risk"].map(order))
                .sort_values(["sort_key", "Attendance (%)"])
                .drop(columns="sort_key")
                .reset_index(drop=True))

# Save the report (the CSV always works; Excel needs the 'openpyxl' package, which Colab has)
report.to_csv("outputs/student_risk_report.csv", index=False)
try:
    report.to_excel("outputs/student_risk_report.xlsx", index=False)
    print("Report saved in the 'outputs/' folder (CSV and Excel).")
except ImportError:
    print("Report saved as CSV. Excel skipped - run  !pip install openpyxl  and re-run this cell for the .xlsx file.")

print("\nPredicted risk levels:\n", report["Risk"].value_counts())
report.head(10)
# Create a directory named 'outputs' on disk if it does not already exist, preventing errors using exist_ok=True
os.makedirs("outputs", exist_ok=True)

# Generate risk level predictions for every student in the entire dataset using the trained best model pipeline
all_pred = best_model.predict(df[features])

# Use a list comprehension to generate a personalized recommendation string for each student by pairing row data with predicted risk
recs = [make_recommendation(row["Attendance (%)"], row, risk)
        for (_, row), risk in zip(df.iterrows(), all_pred)]

# Construct a consolidated summary DataFrame containing student names, attendance, predicted risk levels, and generated recommendations
report = pd.DataFrame({
    # Combine Student_ID, First_Name, and Last_Name into a single formatted string identifier column
    "Student": df["Student_ID"].astype(str) + " - " + df["First_Name"] + " " + df["Last_Name"],

    # Extract attendance percentages rounded to 1 decimal place
    "Attendance (%)": df["Attendance (%)"].round(1),

    # Store the array of predicted risk categories ("High", "Medium", "Low")
    "Risk": all_pred,

    # Store the generated custom recommendation text strings
    "Recommendation": recs,
})

# Define a dictionary mapping risk categories to numeric ranks to prioritize high-risk students at the top of the report
order = {"High": 0, "Medium": 1, "Low": 2}

# Sort the report by risk severity first (High -> Medium -> Low) and attendance ascending second, then clean up the temporary sort key and reset row indices
report = (report.assign(sort_key=report["Risk"].map(order))
                .sort_values(["sort_key", "Attendance (%)"])
                .drop(columns="sort_key")
                .reset_index(drop=True))

# Export the ordered risk summary report as a CSV file inside the 'outputs' directory without row indices
report.to_csv("outputs/student_risk_report.csv", index=False)

# Export the ordered risk summary report as an Excel spreadsheet inside the 'outputs' directory without row indices
report.to_excel("outputs/student_risk_report.xlsx", index=False)

# Print a confirmation message indicating that both output files have been successfully written to disk
print("Report saved in the 'outputs/' folder (CSV and Excel).")

# Print a section header for the predicted risk class distribution table
print("\nPredicted risk levels:\n", report["Risk"].value_counts())

# Render and display the top 10 most urgent student records from the sorted report DataFrame
report.head(10)

