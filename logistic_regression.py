    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    dataset = pd.read_csv("train.csv")
    x = dataset.iloc[:, [2, 4, 5, 6, 7, 9]].values
    y = dataset.iloc[:, 1].values
    
     
    
    #from sklearn.model_selection import train_test_split
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25,  random_state = 0)
    dataset_test = pd.read_csv("test.csv")
    x_test = dataset_test.iloc[:, [1, 3, 4, 5, 6, 8]].values
    out = dataset_test.iloc[:, [0]]                      
    #finding
    #avg = (np.average(x[0]) + np.average(x_test[0])) / 2 

    
    from sklearn.preprocessing import LabelEncoder
    labelencoder_x = LabelEncoder()
    x[:, 1] = labelencoder_x.fit_transform(x[:, 1])
    labelencoder_x_test = LabelEncoder()
    x_test[:, 1] = labelencoder_x_test.fit_transform(x_test[:, 1])
    
    avg_train = 29.6991
    avg_test = 27
    x[pd.isnull(x)] = avg_train
    x_test[pd.isnull(x_test)] = avg_test

#    from sklearn.preprocessing import Imputer
#    imputer = Imputer(missing_values = "NaN", strategy = "median", axis = 0)
#    
#    imputer = imputer.fit(x[:, [1]])
#    x[:, [1]] = imputer.transform(x[:, [1]])
#    
#    imputer = Imputer(missing_values = "NaN", strategy = "median", axis = 0)
#    
#    imputer = imputer.fit(x_test[:, [1]])
#    x_test[:, [1]] = imputer.transform(x_test[:, [1]])
     
#    from sklearn.preprocessing import StandardScaler
#    sc_x = StandardScaler()
#    x_train = sc_x.fit_transform(x)
#    x_test = sc_x.transform(x_test)
    
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(x, y)
    
    y_pred = classifier.predict(x_test)
     
    
    #from sklearn.metrics import confusion_matrix
    #cm = confusion_matrix(y_test, y_pred)
    
    out["Survived"] = y_pred
    out.to_csv("out.csv")
