# Academic Risk Prediction

A machine-learning pipeline that identifies students needing academic intervention, categorizes their risk level (`Low`, `Medium`, `High`), and auto-generates personalized action steps. Built for the Developer Community SASTRA x GDG on Campus AI/ML recruitment task.

* **Colab Notebook:** [Open Interactive Notebook](https://colab.research.google.com/drive/1lqelgmVHPImMjumi-qTE-zskU6PF0FVn?usp=sharing)
* **Dataset:** Kaggle Student Performance Dataset

---

## What the Project Does
1. **Data Exploration:** Analyzes distribution of marks, attendance, and feature correlations.
2. **Data Cleaning:** Cleans invalid entries (negative scores/out-of-bound values) and fills missing values using pipeline medians.
3. **Risk Definition:** Defines target risk categories using a domain-specific formula.
4. **Model Training & Evaluation:** Trains and evaluates Baseline, Logistic Regression, Decision Tree, and Random Forest models using Macro F1 and High-Risk Recall.
5. **Report Generation:** Generates personalized action advice and exports a sorted Excel report (`outputs/student_risk_report.xlsx`).

---

## How Risk is Defined

$$\text{Risk Score} = 0.6 \times (100 - \text{Marks}) + 0.4 \times (100 - \text{Attendance})$$

* **High Risk:** $\text{Risk Score} \ge 40$ **OR** $\text{Marks} < 50$
* **Medium Risk:** $\text{Risk Score} \ge 25$
* **Low Risk:** Everything else

### Rationale
* **Marks Weight (60%):** Direct reflection of academic performance. Any score below 50 immediately flags a student as High Risk regardless of attendance.
* **Attendance Weight (40%):** Serves as an early indicator of student disengagement before final exams.

### Worked Example (From Brief)
For a student with **62% Attendance** and **55 Marks**:
$$\text{Risk Score} = 0.6 \times (100 - 55) + 0.4 \times (100 - 62) = 0.6(45) + 0.4(38) = 27 + 15.2 = 42.2$$
Since $42.2 \ge 40$, the student is categorized as **High Risk**.

---

## Features & Engineering Choices
* **Removed Features:** Dropped `Student_ID`, names, email, and `Gender` to prevent non-predictive noise and algorithmic bias.
* **Prevented Data Leakage:** Removed `Total_Score` and `Grade` from model training features since they directly derive the target label.
* **Median Imputation:** Imputed missing values using median statistics inside the pipeline to avoid skew from extreme outliers.
* **Primary Evaluation Metric:** Focused on **High-Risk Recall** and **Macro F1** because failing to identify an at-risk student (False Negative) has severe academic consequences.

---

## Model Performance & Results

| Model | Accuracy | Macro F1 | High-Risk Recall |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.89** | **0.88** | **0.92** |
| Random Forest | 0.85 | 0.82 | 0.84 |
| Decision Tree | 0.82 | 0.79 | 0.81 |
| Baseline | 0.50 | 0.33 | 0.35 |

**Best Model:** **Logistic Regression** delivered the highest recall for high-risk students and generalized better than tree-based models without overfitting to training noise.

---

## Example Output

```text
Student ID: Student B
Attendance: 62%
Marks: 55
Risk Level: HIGH
Recommendation: Focus on upcoming assessments, revise weak topics, and maintain attendance above 75%.
