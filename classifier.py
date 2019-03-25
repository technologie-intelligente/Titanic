import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion


try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer



# Import all the data.
test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')

# Delete missing values present in our data
train_data = train_data.dropna(subset=['Embarked'])

# Modification of ticket

def modify_ticket (X) :
    Ticket = []
    for i in list(X["Ticket"]):
        if not i.isdigit():
            Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])
        else:
            Ticket.append("X")
    return Ticket

# train_data["Ticket"] = modify_ticket(train_data)
# test_data["Ticket"] = modify_ticket(test_data)


# A class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)



# All Attributes : 		Name 		Age


# Complete : SibSp, Parch, Fare
num_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(["SibSp", "Parch", "Fare"])),
    ("imputer", SimpleImputer(strategy="median")),
])

# Complete : Pclass, Sex, Ticket
# Complete Beginning : Embarked

cat_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])), # Add Ticket
    ("imputer", MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse=False)),
])



# Union all the pipeline
preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])



X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data["Survived"]



from sklearn.svm import SVC
from sklearn import tree


decision_tree = tree.DecisionTreeClassifier()
decision_tree .fit(X_train, y_train)


X_test = preprocess_pipeline.transform(test_data)
y_pred = decision_tree.predict(X_test)




