# Sklearn-pipeline-using-titanic


🛳Titanic Survival Prediction using Scikit-learn Pipelines



This repository contains a machine learning project that predicts the survival of passengers aboard the Titanic using a structured Scikit-learn pipeline. It includes both model training and production-ready prediction code using a `DecisionTreeClassifier`.



 🧠 Project Overview

Objective:
Predict whether a given passenger survived the Titanic disaster using relevant features.

Target variable:
Survived (0 = No, 1 = Yes)

⚠ Note on Accuracy

The primary objective of this project was not to achieve the highest possible accuracy, but rather to gain a solid understanding of how to build, structure, and deploy a machine learning pipeline using Scikit-learn. 
While the model (Decision Tree Classifier) may not deliver top-tier performance compared to more advanced techniques or ensembles, 
it serves as a practical and educational example of preprocessing, encoding, feature selection, and serialization — all handled cleanly through Scikit-learn’s Pipeline and ColumnTransformer utilities.

⚙MLPipeline Details


1. Preprocessing Steps:

Dropped columns: `Name`, `Ticket`, `Cabin`, `PassengerId`


Missing Value Imputation:

Age: SimpleImputer (default strategy: mean)
Embarked:SimpleImputer(strategy="most_frequent")

Categorical Encoding:
Used OneHotEncoder(sparse_output=False,handle_unknown="ignore") for Sex and Embarked

Feature Scaling:
Applied `MinMaxScaler()` on numerical features

Feature Selection:
Used SelectKBest(score_func=chi2, k=8) to retain top 8 features

2. Model:  DecisionTreeClassifier() from Scikit-learn

 🧪 Model Training

Training and pipeline setup are done inside 3-titanic-using-pipeline.ipynb.

Train the pipeline and save the model
Run the notebook: 3-titanic-using-pipeline.ipynb


🚀 Production Inference

Load the trained model and use it for predictions on new data:

Run inference using test input
Run the notebook: 4-production-using-pipeline.ipynb

Sample test input:
test_input = np.array([2, "male", 31.0, 0, 0, 10.5, 'S'], dtype=object).reshape(1, 7)


📊 Evaluation

Model performance is evaluated using:

- train_test_split()
- accuracy_score from sklearn.metrics

📦 Technologies Used
- Python 
- Pandas, NumPy
- Scikit-learn
- Jupyter Notebook
- Pickle (for model serialization)

📌 Future Enhancements
- Replace DecisionTreeClassifier with models like `RandomForest` or XGBoost- Use cross-validation (`GridSearchCV`)
- Deploy as an API using Flask or Streamlit

