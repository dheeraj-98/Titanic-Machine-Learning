import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("train.csv")
x = dataset.iloc[:, [5, 9]].values
y = dataset.iloc[:, 1].values

 

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25,  random_state = 0)
dataset_test = pd.read_csv("test.csv")
x_test = dataset_test.iloc[:, [4, 8]].values
out = dataset_test.iloc[:, [0]]                        
#finding
avg = (np.average(x[0]) + np.average(x_test[0])) / 2  

x[np.isnan(x)] = avg
x_test[np.isnan(x_test)] = avg

 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x)
x_test = sc_x.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y)

y_pred = classifier.predict(x_test)
 

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

out[1] = y_pred
#np.savetxt("out.csv", out, delimiter="
out.to_csv("out.csv")
