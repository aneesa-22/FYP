# The LightGBM (LGBMClassifier) model is used for training and prediction in the provided code. Various oversampling and 
# undersampling methods are applied to handle class imbalance, including techniques like SMOTE and AllKNN. In the final 
# output, AllKNN was selected as the undersampling method, which significantly improved model performance, as shown by the 
# higher metrics in comparison to the original dataset.


import warnings, logging
warnings.filterwarnings('ignore')                # still ignore sklearn warnings
logging.getLogger('lightgbm').setLevel(logging.ERROR)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, RandomOverSampler, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, RandomUnderSampler, TomekLinks
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.table import Table
from prettytable import PrettyTable
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.regularizers import l1, l2
from keras.layers import LSTM, Dense, Dropout  # Add Dropout import
from keras.regularizers import l1  # Add l1 regularization import
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, RandomOverSampler, BorderlineSMOTE
# from imblearn.under_sampling import EditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, RandomUnderSampler, TomekLinks
from imblearn.under_sampling import AllKNN
from collections import Counter
import pandas as pd
from lightgbm import LGBMClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.regularizers import l1
from keras.callbacks import History
from tensorflow.keras.layers import GRU
from imblearn.over_sampling import KMeansSMOTE
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import make_pipeline
from prettytable import PrettyTable
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1
from keras.callbacks import History
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


# Load Data
csv_file_path = "diabetes.csv"
data = pd.read_csv(csv_file_path)

# Basic statistical information of the data set
tanitim = data.describe()

# Extract features and target variable
X = data.drop(['Outcome'], axis=1).values
y = data['Outcome'].values

print(tanitim)


# Oversampling methods
oversamplers = [SMOTE(random_state=42), KMeansSMOTE(random_state=42), ADASYN(random_state=42), SVMSMOTE(random_state=42),
                RandomOverSampler(random_state=42), BorderlineSMOTE(random_state=42)]

# Undersampling methods
undersamplers = [EditedNearestNeighbours(), AllKNN(), InstanceHardnessThreshold(random_state=42),
                 NearMiss(), NeighbourhoodCleaningRule(), OneSidedSelection(random_state=42),
                 RandomUnderSampler(random_state=42), TomekLinks()]

#lightgbm

def plot_metrics_comparison_lightgbm(X, y, samplers):
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = {metric: [] for metric in metrics_names}
    sampler_names = ['Original']  # will store names like 'Original', 'AllKNN', etc.

    # Original dataset
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)
    lgbm_model_orig = LGBMClassifier(random_state=42, force_col_wise=True, num_leaves=42, min_child_samples=60)
    lgbm_model_orig.fit(X_train_orig, y_train_orig)
    y_pred_orig = lgbm_model_orig.predict(X_test_orig)

    accuracy_orig = accuracy_score(y_test_orig, y_pred_orig)
    precision_orig = precision_score(y_test_orig, y_pred_orig)
    recall_orig = recall_score(y_test_orig, y_pred_orig)
    f1_orig = f1_score(y_test_orig, y_pred_orig)

    metrics_values['Accuracy'].append(accuracy_orig)
    metrics_values['Precision'].append(precision_orig)
    metrics_values['Recall'].append(recall_orig)
    metrics_values['F1 Score'].append(f1_orig)

    # Train & collect metrics for each sampler
    for sampler in samplers:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # lgbm_model = LGBMClassifier(random_state=42, force_col_wise=True)
        lgbm_model = LGBMClassifier(random_state=42, force_col_wise=True, verbosity=-1)

        lgbm_model.fit(X_train, y_train)
        y_pred = lgbm_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics_values['Accuracy'].append(accuracy)
        metrics_values['Precision'].append(precision)
        metrics_values['Recall'].append(recall)
        metrics_values['F1 Score'].append(f1)

        sampler_names.append(sampler.__class__.__name__)  # Add sampler name

    # Create a PrettyTable
    table = PrettyTable()
    table.field_names = ['Sampler', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

    # ❗ Now only add "Original" and "AllKNN" rows
    for name, acc, prec, rec, f1 in zip(sampler_names,
                                        metrics_values['Accuracy'],
                                        metrics_values['Precision'],
                                        metrics_values['Recall'],
                                        metrics_values['F1 Score']):
        if name == 'Original' or name == 'AllKNN':   # Only print Original and AllKNN
            table.add_row([name, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

    print(table)

plot_metrics_comparison_lightgbm(X, y, oversamplers + undersamplers)

import joblib

# Train model as you already did
sampler = AllKNN()
X_resampled, y_resampled = sampler.fit_resample(X, y)

model = LGBMClassifier(random_state=42, force_col_wise=True, verbosity=-1)
model.fit(X_resampled, y_resampled)

# Save the trained model
joblib.dump(model, './lgbm_model.pkl')
print("✅ Model saved successfully!")
