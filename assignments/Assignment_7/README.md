## Graduate Admission Prediction

### Overview
This project involves building a machine learning classifier using a Decision Tree to predict whether a student will get admission to a foreign university based on their GRE score and academic performance. The dataset contains information such as GRE scores, TOEFL scores, university ratings, statement of purpose strength, letter of recommendation strength, undergraduate GPA, research experience, and admission status.

### Steps:
1. **Data Loading:** The dataset is loaded using pandas from the provided CSV file.

2. **Exploratory Data Analysis (EDA):** Initial exploration of the dataset is performed to understand its structure and contents. This includes displaying the first few rows of the dataset and generating descriptive statistics.

3. **Data Visualization:** Various visualizations such as histograms and bar charts are created to gain insights into the distribution of data and relationships between different variables.

4. **Data Preprocessing:** Binary values are assigned to the 'Chance of Admit' variable based on a specific condition.

5. **Model Training:** The dataset is split into training and testing sets. A Decision Tree Classifier is trained on the training data.

6. **Model Evaluation:** The trained model is evaluated using various metrics such as confusion matrix, accuracy, precision, recall, and F1 score.

### Files:
- `Admission_Predict.csv`: Dataset containing information about GRE scores, academic performance, and admission status.


### Requirements:
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, plotly, sklearn


