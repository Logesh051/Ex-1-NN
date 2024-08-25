<H3>EX.NO: 1</H3>  <H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>
<H3>ENTER YOUR NAME       : Logesh.N.A</H3>
<H3>ENTER YOUR REGISTER NO: 212223240078</H3>
<H3>DATE : 23.08.2024</H3>
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("Churn_Modelling.csv")

# Extract features and labels
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Drop unnecessary columns
data = data.drop(['Surname', 'Geography', 'Gender'], axis=1)

# Scale the data
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(data))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Print outputs
print("Original Dataset:")
print(data.head())  # Display the first few rows of the dataset after dropping columns
print("\n")

print("Training Features (X_train):")
print(X_train[:5])  # Display the first 5 rows of the training features
print("\n")

print("Test Features (X_test):")
print(X_test[:5])  # Display the first 5 rows of the test features
print("\n")

print("Training Labels (y_train):")
print(y_train[:5])  # Display the first 5 training labels
print("\n")

print("Test Labels (y_test):")
print(y_test[:5])  # Display the first 5 test labels
print("\n")

print("Length of X_test:", len(X_test))  # Display the length of X_test

```
## OUTPUT:
### Dataset:
![Screenshot 2024-08-23 094705](https://github.com/user-attachments/assets/7aeceb69-ed89-4ba0-b535-8d1885c08abc)
### Training Features (x_train):
![Screenshot 2024-08-23 094713](https://github.com/user-attachments/assets/7e615dda-5bfc-4be3-8855-934e12f6e553)
### Test Features (x_test):
![Screenshot 2024-08-23 094719](https://github.com/user-attachments/assets/aee1b82d-e94a-41fe-b225-a13b3ac33978)
### Training Labels (y_train) and Test Labels (y_test):
![Screenshot 2024-08-23 094723](https://github.com/user-attachments/assets/884fc406-dfa5-499c-bc43-13439fbd6f93)
### Length of X_test:
![Screenshot 2024-08-23 094727](https://github.com/user-attachments/assets/bf5f4753-6776-4f34-a0b9-b66ecf029036)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


