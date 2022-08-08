import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import warnings

# PART A

#Reading excel and arranging data
fall_data = pd.read_csv('C:/Users/ccane/Desktop/ge461/PROJECT 4/falldetection_dataset.csv', header=None)
labels = fall_data.iloc[:,1]
feature = fall_data.loc[:, fall_data.columns != 0]
features = feature.loc[:, feature.columns != 1]

#Applying PCA
x_train,y_train=features.values,labels.values
pca = PCA(n_components=2)
xt = pca.fit_transform(x_train)

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=list(labels)) 
plt.ylabel('percentange of explained variance')
plt.xlabel('principal component')
plt.title('Explained Variance vs PC')
plt.show()
print(per_var) # Results are: 75.3 and 8.5, with 2 principal components total 83.8 variance has been obtained.

# Visualize data seperately after PCA
plt.scatter(xt[y_train == 'F'], xt[y_train == 'F'], label="Fall")
plt.title('Fall Data - PCA with 2 components')
plt.show()

plt.scatter(xt[y_train == 'NF'], xt[y_train == 'NF'], label="Non-Fall")
plt.title('Non-Fall Data - PCA with 2 components')
plt.show()

# Visualize all data after PCA
plt.scatter(xt[y_train == 'NF'], xt[y_train == 'NF'], label="Non-fall")
plt.scatter(xt[y_train == 'F'], xt[y_train == 'F'], label="Fall")
plt.title('PCA with 2 components')
plt.legend()
plt.show()

N = [2, 3, 4, 5, 6] #Trying different 5 clusters 

for number in N:
    for i in range(number):
        k_means = KMeans(n_clusters=number)
        k_means.fit(xt)
        prediction = k_means.predict(xt)
        plt.scatter(xt[prediction == i], xt[prediction == i], label=("Class number:" + str(i+1)))
        plt.legend()
    plt.title(str(number) + " Means Clustering Predictions\nwith First 2 PCA")
    plt.show()

# N=2 -> check the degree of percentage overlap/consistency between the cluster memberships and the action labels originally provided.
kmeans = KMeans(2)
kmeans.fit(xt)
prediction = kmeans.predict(xt)

label=[]
for i in y_train:
    if i == "NF":
        label.append(1)
    else:
        label.append(0)
count=0
for i in range(len(prediction)):
    if label[i]==prediction[i]:
        count+=1
print("Percentage Overlap with data:",count/len(prediction)) #0.551

# Lets remove one of the outlier to see the change the overlap percentage.
outlier_1 = min(xt[:, 1])
index_1 = np.where((xt[:, 1] == outlier_1) == True)[0][0]
x_train_transformed_out = np.delete(xt, [index_1], axis=0)
train_Y_transformed_out = np.delete(y_train, [index_1], axis=0)
kmeans.fit(x_train_transformed_out)
prediction_2 = kmeans.predict(x_train_transformed_out)

label_2=[]
for i in train_Y_transformed_out:
    if i == "F":
        label_2.append(1)
    else:
        label_2.append(0)
count_2=0
for i in range(len(prediction_2)):
    if label_2[i]==prediction_2[i]:
        count_2+=1

print("Percentage Overlap with data:",(count_2/len(prediction_2))) # 0.784. WE OBSERVE AN IMPROVEMENT.


# PART B

x_train, x_test_valid, y_train, y_test_valid = train_test_split(x_train, y_train, test_size=0.3, shuffle=True,random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_test_valid, y_test_valid, test_size=0.5, shuffle=True,random_state=42)

# Support-Vector-Machine (SVM) Classifer:

def get_key(val,dictt):
    listem=[]
    for key, value in dictt.items():
         if val == value:
             listem.append(key)
    return listem

# the parameter selection that is performed based on the validation set

kernel_opt=["linear", "poly","rbf", "sigmoid"]
c_opt=[0.01,0.1,1,10,100]
gamma_opt=["scale","auto"]

d = {} 
for a in kernel_opt:
    for b in gamma_opt:
        for c in c_opt:
            classifier = SVC(kernel=a,C=c,gamma=b, max_iter=10000, random_state=10)
            classifier.fit(x_train,y_train)

            # perform prediction on x_valid data
            y_pred = classifier.predict(x_valid)

            # creating confusion matrix and accuracy calculation
            cm = confusion_matrix(y_valid,y_pred)
            accuracy = float(cm.diagonal().sum())/len(y_valid)
            print('mlp accuracy is:',accuracy*100,'%')
            p = str(a)+" "+str(b)+" "+str(c)
            d[p]= accuracy
            for key, value in d.items():
                print(key, ':', value)


print("best",get_key(max(d.values()),d)[0])

# Now lets test with test data

classifier = SVC(kernel="poly",C=0.01,gamma="auto", max_iter=10000, random_state=10)
classifier.fit(x_train,y_train)
# perform prediction on x_valid data
y_pred = classifier.predict(x_test)
# creating confusion matrix and accuracy calculation
cm = confusion_matrix(y_test,y_pred)
accuracy = float(cm.diagonal().sum())/len(y_test)
print('mlp accuracy is:',accuracy*100,'%')


df = pd.DataFrame(data=d ,index=[0])
df = (df.T)
print (df)
df.to_excel('svm.xlsx')

# Multi-Layer Perceptron (MLP) Classifer:

hidden_layer_size= [(15,),(15,15),(30,30)] 
activation= ["identity","logistic", "tanh", "relu"] 
solver=["lbfgs", "sgd",]
learning_rate=["constant", "invscaling", "adaptive"]
learning_rate_init=[0.001,0.01,0.1]

dic = {} 
count=0
for d in hidden_layer_size:
    for e in activation:
        for f in solver:
            for g in learning_rate:
                for h in learning_rate_init:
                    warnings.filterwarnings("ignore")
                    mlp = MLPClassifier(hidden_layer_sizes=d, activation=e, solver=f, learning_rate=g,learning_rate_init=h,max_iter=5000,random_state=10)
                    mlp.fit(x_train, y_train)
                    prediction_mlp = mlp.predict(x_valid)
                    cm = confusion_matrix(y_valid,prediction_mlp)
                    accuracy = float(cm.diagonal().sum())/len(y_valid)
                    p = str(d)+" "+str(e)+" "+str(f)+" "+str(g)+" "+str(h)
                    dic[p]= accuracy
                    for key, value in dic.items():
                        print(key, ':', value)

print("best",get_key(max(dic.values()),dic)[0])


df2 = pd.DataFrame(data=dic ,index=[0])
df2 = (df2.T)
df2.to_excel('mlp.xlsx')


# Now lets test with test data

mlp = MLPClassifier(hidden_layer_sizes=(15,), activation="identity", solver="sgd", learning_rate="constant",learning_rate_init=0.001,max_iter=5000,random_state=10)
mlp.fit(x_train, y_train)
prediction_mlp = mlp.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy = float(cm.diagonal().sum())/len(y_test)
print('mlp accuracy is:',accuracy*100,'%') # 100% accuracy
