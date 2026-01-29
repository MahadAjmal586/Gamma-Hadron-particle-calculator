import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler as smot

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

df = pd.read_csv("telescope_data.csv")
df.head()

df["class"].unique()

df["class"] = (df["class"] == "g").astype(int)

#df.head()

df.dtypes
df.describe()
df.info()

#df.hist(bins=40, figsize=(12, 10))

"""# **Data Visualizing**"""

for label in df.columns[:-1]:
  plt.hist(df[df['class']==1][label],color = "blue",label = "gamma", alpha = 0.7 , density= True)
  plt.hist(df[df['class']==0][label],color = "green",label = "hadron", alpha = 0.7 , density= True)
  plt.title(label)
  plt.ylabel("probality")
  plt.xlabel(label)
  plt.legend()
  plt.show()

"""# **Data Scalling**"""

def scale_dataset (dataframe, oversample = False):
  x = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scalar = StandardScaler()
  x = scalar.fit_transform(x)

  if oversample:
    ros = smot()
    x, y = ros.fit_resample(x, y)

  data = np.hstack((x,np.reshape(y, (-1, 1))))

  return data, x , y

"""# **Data Splitting**"""

train_df, validate_df, test_df = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])

train, x_train, y_train = scale_dataset(train_df , oversample= True )
validate, x_validate, y_validate = scale_dataset(validate_df , oversample= False )
test, x_test, y_test = scale_dataset(test_df , oversample= False)

"""# **KNN Model**"""

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_test)

print (classification_report(y_test, y_pred))

"""# **Navie Base**"""

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model ==nb_model.fit(x_train, y_train)

y_pred2 = nb_model.predict(x_test)
print(classification_report(y_test, y_pred2))

print