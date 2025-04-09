import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import  metrics
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

#Load the dataset
dataset = pd.read_csv(r"C:\Users\vish8\OneDrive\Desktop\Cursos\MLopsBootcamp\MLOpsBootcamp\MlFlow\train.csv")
numerical_cols = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Loan_Status')
categorical_cols.remove('Loan_ID')

#Filling categorical columns with mode
for col in categorical_cols:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

#Filling numerical columns with mean
for col in numerical_cols:
    dataset[col].fillna(dataset[col].mean(), inplace=True)

#Take care of outliers
dataset[numerical_cols] = dataset[numerical_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))

#Log transformation  & Domain Processing
dataset['LoanAmount'] = np.log(dataset['LoanAmount']).copy()
dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome'] = np.log(dataset['TotalIncome']).copy()

#Dropping ApplicantIncome and CoapplicantIncome
dataset = dataset.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])

#Label encoding categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])

#Encode the target columns
dataset['Loan_Status'] = le.fit_transform(dataset['Loan_Status'])

#Train test split
X = dataset.drop(columns=['Loan_Status', 'Loan_ID'])
y = dataset['Loan_Status']
RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

#RandomForest
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    'n_estimators': [200,400, 700],
    'max_depth': [10,20,30],
    'criterion' : ["gini", "entropy"],
    'max_leaf_nodes': [50, 100]
}

grid_forest = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_forest, 
        cv=5, 
        n_jobs=1, 
        scoring='accuracy',
        verbose=0
    )
model_forest = grid_forest.fit(X_train, y_train)

#Logistic Regression
lr = LogisticRegression(random_state=RANDOM_SEED)
param_grid_logistic = {
    'C': [100, 10, 1.0, 0.1, 0.01],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
grid_log = GridSearchCV(
    estimator=lr,
    param_grid=param_grid_logistic,
    cv=5,
    n_jobs=1,
    scoring='accuracy',
    verbose=0
)
model_log = grid_log.fit(X_train, y_train)

#Decision Tree
dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
param_grid_tree = {
    'max_depth' : [ 3, 5, 7, 9, 11, 13],
    'criterion' : [ 'gini', 'entropy'],
}
grid_tree = GridSearchCV(
    estimator = dt,
    param_grid = param_grid_tree,
    cv = 5,
    n_jobs = 1,
    scoring = 'accuracy',
    verbose = 0
)
model_tree = grid_tree.fit(X_train, y_train)


mlflow.set_experiment("Predição de Empréstimos")


#Model metrics
def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f'%auc)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    plt.close()
    return(accuracy, f1, auc)

# Armazena os resultados dos modelos
model_results = {}

def mlflow_logging(model, X, y, name):
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        run_id = run.info.run_id
        pred = model.predict(X)
        (accuracy, f1, auc) = eval_metrics(y, pred)

        # Loga parâmetros e métricas
        mlflow.log_params(model.best_params_)
        mlflow.log_metric("Mean CV score", model.best_score_)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("AUC", auc)

        # Artefatos
        mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.sklearn.log_model(model, name)

        # Armazena resultados para comparação posterior
        model_results[run_id] = {
            "name": name,
            "f1-score": f1,
            "accuracy": accuracy,
            "AUC": auc
        }

# Roda os experimentos
mlflow_logging(model_tree, X_test, y_test, "DecisionTreeClassifier")
mlflow_logging(model_log, X_test, y_test, "LogisticRegression")
mlflow_logging(model_forest, X_test, y_test, "RandomForestClassifier")

# Define critério de melhor modelo
melhor_run_id = max(model_results, key=lambda x: model_results[x]["f1-score"])

# Marca o melhor modelo com uma tag extra
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.start_run(run_id=melhor_run_id)
mlflow.set_tag("melhor_modelo", "True")
mlflow.end_run()


mlflow_logging(model_tree, X_test, y_test, "DecisionTreeClassifier")
mlflow_logging(model_log, X_test, y_test, "LogisticRegression")
mlflow_logging(model_forest, X_test, y_test, "RandomForestClassifier")
# mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001



