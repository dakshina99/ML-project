# %% [markdown]
# # **Layer 12**

# %% [markdown]
# ## **Prepare Environment**

# %% [markdown]
# ### Import libraries and modules
# 
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# %% [markdown]
# ### Load the dataset
# *   Training Data
# *   Validation Data
# *   Test Data

# %%
train_path = 'D:\\ACADEMIC\\SEMESTER 07\\ML - 3\\project\\Layer12\\train.csv'
valid_path = 'D:\\ACADEMIC\\SEMESTER 07\\ML - 3\\project\\Layer12\\valid.csv'
test_path = 'D:\\ACADEMIC\\SEMESTER 07\\ML - 3\\project\\Layer12\\test.csv'

train_data = pd.read_csv(train_path)

valid_data = pd.read_csv(valid_path)

test_data = pd.read_csv(test_path)

# %% [markdown]
# ### Dataset Info and split dataset according to the labels

# %%
print(train_data.shape)
print(valid_data.shape)
print(test_data.shape)

# %%
train_data.head()

# %%
valid_data.head()

# %%
test_data.head()

# %% [markdown]
# Drop the ID column from the test dataset

# %%
test_IDs = test_data['ID'].to_numpy()
test_data = test_data.drop(columns=['ID'])

# %% [markdown]
# Prepare training and validation data for each label

# %%
train_data_label1 = train_data.drop(columns=['label_2', 'label_3', 'label_4'])
train_data_label2 = train_data.drop(columns=['label_1', 'label_3', 'label_4'])
train_data_label3 = train_data.drop(columns=['label_1', 'label_2', 'label_4'])
train_data_label4 = train_data.drop(columns=['label_1', 'label_2', 'label_3'])

valid_data_label1 = valid_data.drop(columns=['label_2', 'label_3', 'label_4'])
valid_data_label2 = valid_data.drop(columns=['label_1', 'label_3', 'label_4'])
valid_data_label3 = valid_data.drop(columns=['label_1', 'label_2', 'label_4'])
valid_data_label4 = valid_data.drop(columns=['label_1', 'label_2', 'label_3'])

# %% [markdown]
# # Define Functions

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# > *Train a model to predict the label 01 after appling some feature engineering techniques and methods to the training data.
# Features are selected based on the correlation matrix and the PCA used to extract the features*

# %% [markdown]
# ### Data Cleaning

# %% [markdown]
# 
# 
# > Remove null values for labels and determine missing values in features
# 

# %% [markdown]
# **Drop** the rows where there are null values for the lables in the training dataset

# %%
def clean_null_labels(train_data_label1, label):
    print("Train set shape before: {}".format(train_data_label1.shape))

    train_features_null_counts = train_data_label1.drop(columns=[f'label_{label}']).isnull().sum()
    train_label_null_count = train_data_label1[f'label_{label}'].isnull().sum()
    print("Null value counts of the features\n{}".format(train_features_null_counts))
    print("Null value count: {}".format(train_label_null_count))

    cleaned = train_data_label1.dropna(subset=train_data_label1.columns[-1:], how='any')
    print("Train set shape after: {}".format(cleaned.shape))
    return cleaned

# %% [markdown]
# Fill the null values in the features with their **means** in the datasets.

# %%
def fill_null_features(train_data_label1, valid_data_label1, test_data):
    train_data_label1 = train_data_label1.fillna(train_data_label1.mean())
    valid_data_label1 = valid_data_label1.fillna(valid_data_label1.mean())
    test_data = test_data.fillna(test_data.mean())
    return train_data_label1, valid_data_label1, test_data

# %% [markdown]
# Split the Features and Labels in the dataset

# %%
def split_features_labels(train_data_label1, valid_data_label1, test_data, label):
    train_features_label1 = train_data_label1.iloc[:, :-1]
    train_label1 = train_data_label1[f'label_{label}']

    valid_features_label1 = valid_data_label1.iloc[:, :-1]
    valid_label1 = valid_data_label1[f'label_{label}']

    test_features_label1 = test_data
    return train_features_label1, train_label1, valid_features_label1, valid_label1, test_features_label1

# %% [markdown]
# Label 01 distribution after cleaning

# %%
def plot_label(train_label1, label):    
    labels, counts = np.unique(train_label1, return_counts=True)

    plt.figure(figsize=(18, 3))
    plt.xticks(labels)
    plt.bar(labels, counts)
    plt.xlabel(f'label_{label}')
    plt.ylabel('Frequency')
    plt.title('Distribution of the Label')
    plt.show()

# %% [markdown]
# ### Feature Standardization

# %% [markdown]
# > Standardize the features of the dataset using **Robust scaler**

# %%
def standardize_data(train_features_label1, valid_features_label1, test_features_label1, scaler=StandardScaler()): 
    standardized_train_features_label1 = scaler.fit_transform(train_features_label1)
    standardized_valid_features_label1 = scaler.transform(valid_features_label1)
    standardized_test_features_label1 = scaler.transform(test_features_label1)
    return standardized_train_features_label1, standardized_valid_features_label1, standardized_test_features_label1

# %% [markdown]
# ### Feature Extraction

# %% [markdown]
# > Principal Componenet Analysis(PCA) used to extract the features that can explain the variance of the label to 95% and display the resulting explained variances of each PC

# %%
def apply_PCA(standardized_train_features_label1, standardized_valid_features_label1, standardized_test_features_label1, variance_threshold):

    pca = PCA(n_components=variance_threshold, svd_solver='full')

    pca_train_features_label1 = pca.fit_transform(standardized_train_features_label1)
    pca_valid_features_label1 = pca.transform(standardized_valid_features_label1)
    pca_test_features_label1 = pca.transform(standardized_test_features_label1)

    explained_variance_ratio_reduced = pca.explained_variance_ratio_

    plt.figure(figsize=(18, 10))
    plt.bar(range(1, pca_train_features_label1.shape[1] + 1), explained_variance_ratio_reduced)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio per Principal Component (Reduced)')
    plt.show()

    print("\nReduced Train feature matrix shape: {}".format(pca_train_features_label1.shape))
    print("Reduced valid feature matrix shape: {}".format(pca_valid_features_label1.shape))
    print("Reduced test feature matrix shape: {}".format(pca_test_features_label1.shape))
    
    return pca_train_features_label1, pca_valid_features_label1, pca_test_features_label1

# %% [markdown]
# ## Hyperparameter tuning

# %% [markdown]
# Define parameters for random search

# %%
def get_hyper_params():
    svm_grid_params = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf','linear']
    }

    knn_grid_params = {
        'n_neighbors' : [3, 5, 7, 9, 11, 13],
        'weights' : ['uniform', 'distance'],
        'metric' : ['minkowski', 'euclidean', 'manhattan', 'hamming']
    }

    random_forest_grid_params = {
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    }
    return svm_grid_params, knn_grid_params, random_forest_grid_params

# %% [markdown]
# Tune hyperparameters with the best method by testing severel methods and for several models

# %%
def tune_hyper_params(svm_grid_params, knn_grid_params, rf_grid_params, pca_train_features_label1, train_label1, rand=True):
    classification_models_params = [
        ('SVM', SVC(), svm_grid_params),
    #     ('K Neighbors', KNeighborsClassifier(), knn_grid_params),
    #     ('Random Forest', RandomForestClassifier(), rf_grid_params)
    ]

    for model_name, model, grid_params in classification_models_params:
        if rand:
            search = RandomizedSearchCV(
                estimator = model,
                param_distributions = grid_params,
                n_iter = 40, cv = 3, verbose=4, random_state=42, n_jobs = -1
            )
        else:
            search = HalvingGridSearchCV(
                estimator=model,
                param_grid=grid_params,
                cv=3,
                n_jobs=-1,
                factor=2,
                verbose=2
            )
        result = search.fit(pca_train_features_label1, train_label1)

        print(f"best score for {model_name} : {result.best_score_}")
        print(f"best hyper parameters for {model_name} : {result.best_params_}")

# %% [markdown]
# ## Train the best performing model

# %% [markdown]
# Select the model that best predicts the valid and test datasets based on accuracy, precision and recall and train

# %%
def train_model(pca_train_features_label1, train_label1, pca_valid_features_label1, valid_label1, svm=None, rf=None, knn=None):
    classification_models = [
        # ('K Neighbors', knn),
        # ('Random Forest', rf),
        ('SVM', svm)
        
    ]
    
    models = []

    for model_name, model in classification_models:
        num_features = pca_train_features_label1.shape[1]
        print(f"{model_name} is training for {num_features} number of features\n")
        
        models.append(model)
        kf = KFold(n_splits=3, random_state=42, shuffle=True)
        cross_val_scores = cross_val_score(model, pca_train_features_label1, train_label1, cv=kf, verbose=4)

        print("CV Accuracy: %0.4f accuracy with a standard deviation of %0.2f" % (cross_val_scores.mean(), cross_val_scores.std()))
        print("\n")
    
    return models

# %%
def get_test_result(pca_train_features_label1, train_label1, pca_test_features_label1, model): 
    model.fit(pca_train_features_label1, train_label1)
    return model.predict(pca_test_features_label1)

# %%
def create_csv(ID, pred_label1, destination):
  df = pd.DataFrame()

  df.insert(loc=0, column='ID', value=ID)
  df.insert(loc=1, column='label_1', value=pred_label1)

  df.to_csv(destination, index=False)

# %% [markdown]
# # Label 01

# %%
train_data_label1 = clean_null_labels(train_data_label1, 1)
train_data_label1, valid_data_label1, test_data = fill_null_features(train_data_label1, valid_data_label1, test_data)
train_features_label1, train_label1, valid_features_label1, valid_label1, test_features_label1 = split_features_labels(train_data_label1, valid_data_label1, test_data, 1)
plot_label(train_label1, 1)

# %%
standardized_train_features_label1, standardized_valid_features_label1, standardized_test_features_label1 = standardize_data(train_features_label1, valid_features_label1, test_features_label1, StandardScaler())

# %%
pca_train_features_label1, pca_valid_features_label1, pca_test_features_label1 = apply_PCA(standardized_train_features_label1, standardized_valid_features_label1, standardized_test_features_label1, 0.99)

# %%
# svm_grid_params, knn_grid_params, random_forest_grid_params = get_hyper_params()
# tune_hyper_params(svm_grid_params, knn_grid_params, random_forest_grid_params, pca_train_features_label1, train_label1, rand=True)

# %%
model1 = train_model(standardized_train_features_label1, train_label1, standardized_valid_features_label1, valid_label1, SVC(C=100, gamma=0.001, kernel='rbf', class_weight='balanced'))

# %%
y_pred_test_label1 = get_test_result(standardized_train_features_label1, train_label1, standardized_test_features_label1, model1[0])

# %%
destination = 'D:\\ACADEMIC\\SEMESTER 07\\ML - 3\\project\\Layer12\\results\\01.csv'

create_csv(test_IDs, y_pred_test_label1, destination)

# %% [markdown]
# # Label 02

# %%
train_data_label2 = clean_null_labels(train_data_label2, 2)
train_data_label2, valid_data_label2, test_data = fill_null_features(train_data_label2, valid_data_label2, test_data)
train_features_label2, train_label2, valid_features_label2, valid_label2, test_features_label2 = split_features_labels(train_data_label2, valid_data_label2, test_data, 2)
plot_label(train_label2, 2)

# %%
standardized_train_features_label2, standardized_valid_features_label2, standardized_test_features_label2 = standardize_data(train_features_label2, valid_features_label2, test_features_label2, StandardScaler())

# %%
pca_train_features_label2, pca_valid_features_label2, pca_test_features_label2 = apply_PCA(standardized_train_features_label2, standardized_valid_features_label2, standardized_test_features_label2, 0.99)

# %%
# svm_grid_params, knn_grid_params, random_forest_grid_params = get_hyper_params()
# tune_hyper_params(svm_grid_params, knn_grid_params, random_forest_grid_params, pca_train_features_label2, train_label2, rand=True)

# %%
model2 = train_model(standardized_train_features_label2, train_label2, standardized_valid_features_label2, valid_label2, SVC(C=100, gamma=0.001, kernel='rbf', class_weight='balanced'))

# %%
y_pred_test_label2 = get_test_result(standardized_train_features_label2, train_label2, standardized_test_features_label2, model2[0])

# %%
destination = 'D:\\ACADEMIC\\SEMESTER 07\\ML - 3\\project\\Layer12\\results\\02.csv'

create_csv(test_IDs, y_pred_test_label2, destination)

# %% [markdown]
# # Label 03

# %%
train_data_label3 = clean_null_labels(train_data_label3, 3)
train_data_label3, valid_data_label3, test_data = fill_null_features(train_data_label3, valid_data_label3, test_data)
train_features_label3, train_label3, valid_features_label3, valid_label3, test_features_label3 = split_features_labels(train_data_label3, valid_data_label3, test_data, 3)
plot_label(train_label3, 3)

# %%
standardized_train_features_label3, standardized_valid_features_label3, standardized_test_features_label3 = standardize_data(train_features_label3, valid_features_label3, test_features_label3, StandardScaler())

# %%
pca_train_features_label3, pca_valid_features_label3, pca_test_features_label3 = apply_PCA(standardized_train_features_label3, standardized_valid_features_label3, standardized_test_features_label3, 0.99)

# %%
# svm_grid_params, knn_grid_params, random_forest_grid_params = get_hyper_params()
# tune_hyper_params(svm_grid_params, knn_grid_params, random_forest_grid_params, pca_train_features_label3, train_label3, rand=True)

# %%
model3 = train_model(standardized_train_features_label3, train_label3, standardized_valid_features_label3, valid_label3, SVC(C=100, gamma=0.001, kernel='rbf', class_weight='balanced'))

# %%
y_pred_test_label3 = get_test_result(standardized_train_features_label3, train_label3, standardized_test_features_label3, model3[0])

# %%
destination = 'D:\\ACADEMIC\\SEMESTER 07\\ML - 3\\project\\Layer12\\results\\03.csv'

create_csv(test_IDs, y_pred_test_label3, destination)

# %% [markdown]
# # Label 04

# %%
train_data_label4 = clean_null_labels(train_data_label4, 4)
train_data_label4, valid_data_label4, test_data = fill_null_features(train_data_label4, valid_data_label4, test_data)
train_features_label4, train_label4, valid_features_label4, valid_label4, test_features_label4 = split_features_labels(train_data_label4, valid_data_label4, test_data, 4)
plot_label(train_label4, 4)

# %%
standardized_train_features_label4, standardized_valid_features_label4, standardized_test_features_label4 = standardize_data(train_features_label4, valid_features_label4, test_features_label4, StandardScaler())

# %%
pca_train_features_label4, pca_valid_features_label4, pca_test_features_label4 = apply_PCA(standardized_train_features_label4, standardized_valid_features_label4, standardized_test_features_label4, 0.99)

# %%
svm_grid_params, knn_grid_params, random_forest_grid_params = get_hyper_params()
tune_hyper_params(svm_grid_params, knn_grid_params, random_forest_grid_params, pca_train_features_label4, train_label4, rand=True)

# %%
model4 = train_model(standardized_train_features_label4, train_label4, standardized_valid_features_label4, valid_label4, SVC(C=100, gamma=0.001, kernel='rbf', class_weight='balanced'))

# %%
y_pred_test_label4 = get_test_result(standardized_train_features_label4, train_label4, standardized_test_features_label4, model4[0])

# %%
destination = 'D:\\ACADEMIC\\SEMESTER 07\\ML - 3\\project\\Layer12\\results\\04.csv'

create_csv(test_IDs, y_pred_test_label4, destination)

# %% [markdown]
# ## Create CSV for all results

# %%
destination = 'D:\\ACADEMIC\\SEMESTER 07\\ML - 3\\project\\Layer12\\results\\190507U.csv'

# create the csv output file
create_csv(test_IDs, y_pred_test_label1, y_pred_test_label2, y_pred_test_label3, y_pred_test_label4, destination)


