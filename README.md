#Predict Cultivator Wine
import numpy as np,pandas as pd
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)

wine_df=pd.read_csv(r'C:\Users\user\Documents\Python\dataset\wine_data.csv',header=None,index_col=None)

wine_df.columns["Cultivator","Alchol","Malic_Acid","Ash","Alcalinity_of_Ash","Magnesium","Total_phenols","Falvanoids","Nonflavanoid_phenols", "Proanthocyanins","Color_intensity", "Hue", "OD280", "Proline"]

wine_df.head()

wine_df["Cultivator"].value_counts()

X=wine_df.iloc[:,1:]
Y=wine_df.iloc[:,0]

Y=Y.astype(int)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
#Fit only the trining Data
scaler.fit(X)


#Now apply the Tranformations to the Data:
X=scaler.transform(X)


from sklearn.model_selection import train_test_split
#Split the data into test an train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)

from sklearn.neural_network import MLPClassifier #Multi Layer perceptrons
mlp=MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=100,early_stopping=True,random_state=10,activation="relu")
mlp.fit(X_train,Y_train)
Y_pred=mlp.predict(X_test)
print(list(zip(Y_test,Y_pred)))

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)    #Accuracy

print("Classification report:")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the Model:",acc*100)
