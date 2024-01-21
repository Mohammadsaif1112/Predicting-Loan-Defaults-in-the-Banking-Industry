# Problem: Predicting Loan Defaults in the Banking Industry
People borrow money from financial institutions all the time, either for educational costs, vehicle financing, emergency expenses, or starting a business. However, it is often associated with risk as the borrowers may default on the loan. This, therefore, necessitates a need for predicting defaults. 
Predicting loan defaults is a critical task that helps financial institutions assess the risk of lending to potential borrowers. Accurate default prediction models enable banks to make informed decisions, mitigate financial losses, and maintain a healthy loan portfolio.
For this reason, financial institutions must avoid giving loans to people who are highly likely to default. As a result, they usually invest a lot of time and resources in background checks on people to avoid losses. To help solve this problem, I will develop a machine learning model that will be able to predict how likely an individual is to default based on his annual salary, bank balance, and employment status.  

About the Dataset
To develop this loan default predictor, I will use a dataset with a total of 10,000 clients and four different attributes:
Employed: 1 for employed and 0 for unemployed;
Bank Balance: The amount of money that client had available in their account at the moment the data was obtained;
Annual Salary: The annual salary of each client;
Defaulted?: This is my target variable and it is filled of 0 for each client who did not default and 1 for each client who defaulted their loans.

1. Importing Libraries
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/b7c9965d-c140-42c1-9cc3-2887eecda379)
2. Exploratory Data Analysis
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/0a4bb519-8b61-4c76-ad1f-50322bbf170e)

I will now use some EDA to demonstrate how balanced the target variable is and how it interacts with other attributes
# Checking null values 
df.isnull().sum()
Output
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/bcb1d847-bbce-4d00-80a6-fe4990983b17)
# Renaming target variable
df.rename({'Defaulted?':'Defaulted'}, axis = 1, inplace = True)
df
output
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/75c74633-336c-4223-8633-a9a2cab84bd2)
# Checking class balance
fig = px.pie(df, values = df['Defaulted'].value_counts(), names = ['Did not default','Defaulted'], title = 'Distribution of Clients Who Have Defaulted')
fig.update_traces(rotation=90, pull = [0.2,0.06,0.06,0.06,0.06], textinfo = "percent+label")
fig.show()
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/25bd9f0f-f21c-4ad3-aac5-bfc779590c0a)

From the figure above, it is clear that most clients did not default their loans, indicating a class imbalance with the data.
I will later on use different methods to deal with this imbalance while building my predictor model with AutoML.

I will now separate the clients among those who are employed and those who are unemployed and see how the target variable is distributed among each separate group.
# Separating df into two different groups: Employed clients and unemployed clients
employed = df.query("Employed == 1")
unemployed = df.query("Employed == 0")
# Checking class balance among those who are employed
fig = px.pie(employed, values = employed['Defaulted'].value_counts(), names = ['Did not default','Defaulted'], title = 'Distribution of Clients Who Are Employed and Defaulted')
fig.update_traces(rotation=90, pull = [0.2,0.06,0.06,0.06,0.06], textinfo = "percent+label")
fig.show()
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/4da3866a-07fa-4bef-9f60-067149e2df9d)

# Checking class balance among those who are unemployed
fig = px.pie(unemployed, values = unemployed['Defaulted'].value_counts(), names = ['Did not default','Defaulted'], title = 'Distribution of Clients Who Are Unemployed and Defaulted')
fig.update_traces(rotation=90, pull = [0.2,0.06,0.06,0.06,0.06], textinfo = "percent+label")
fig.show()
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/ffffa6ed-c269-4883-bebc-741c936f913e)
The proportion of those who have defaulted is bigger among unmeployed clients than employed clients.

I will then check how the 'Defaulted' class is distributed according to bank balance and annual salary.
# Default distribution according to bank balance values
fig = plt.figure(figsize = (20, 9))
sns.set_style("dark")
sns.kdeplot(df[df['Defaulted']==1]['Bank Balance'])
sns.kdeplot(df[df['Defaulted']==0]['Bank Balance'])
plt.title('Default x Bank Balance')
plt.legend(labels=['Defaulted', 'Did Not Default'])
plt.show()
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/3a5cbd1c-8c79-424e-9854-7813b0d970a9)
It is evident that on average, clients who defaulted their loans have a higher bank balance than those who did not.
# Default distribution according to annual salaries
fig = plt.figure(figsize = (20, 9))
sns.set_style("dark")
sns.kdeplot(df[df['Defaulted']==1]['Annual Salary'])
sns.kdeplot(df[df['Defaulted']==0]['Annual Salary'])
plt.title('Default x Annual Salaries')
plt.legend(labels=['Defaulted', 'Did Not Default'])
plt.show()
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/b5c8f8e4-8a7c-44da-9345-87c5a601f849)
As seen on the graph, clients who defaulted have lower annual income than those who did not.
# Checking correlations
corr = df.corr()
plt.figure(figsize = (16, 12))
g = sns.heatmap(df.corr(), annot = True)
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/2042c13c-ffa8-45b7-9518-7df5b4c22063)
The strongest correlation is among Annual Salary and Employed, but there is no strong correlation between the target variable and any other attribute. 
There is only a 35% correlation among Defaulted and Bank Balance, but that is not a strong correlation. 

3. Building a Model to Predict Loan Default
Considering there is a big class imbalance in the target variable, the main metric that I am going to use here is the recall score to show me how good the model predicts
the target class, that is, customers who are more likely to default.

I will have my false negatives as low as possible because giving loans to someone who is highly likely to default would lead to the institution's financial loss,
while false positives would not be a problem since I can afterwards check the profile of non-defaulters falsely tagged as defaulters. 

Before I start, I will divide my current dataset into a Training Set and Testing Set.

I will use the Training Set to train and to validate the model on a hold-out sample.
test = df.tail(2000) # 20% of dataset will be used for testing
test
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/89c2f226-1d5c-4d4d-8cf6-6adfa8ad5bab)
# Removing testing data from dataframe and setting up 80% of data left for training and validation
train = df.drop(test.index)
train
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/ab5cbb84-f2c9-4330-b73f-26c955f43b62)
# Importing PyCaret's classification lib
from pycaret.classification import *
setup(# Defining training data
     data = train, 
     # defining target variable
     target = 'Defaulted',
     # 75% of training set will be used for training, 25% will be used for validation while on hold-out
     train_size = 0.75,
     # Ignore index, it won't be any relevant for model building
     ignore_features = ['Index'],
     # This param fixes class imablance with SMOTE technique, increasing the minority class
     fix_imbalance = True,
     # Normalizing features to have them all on a the same scale,
     normalize = True,
     # Transforming features into a Gaussian-like distribution)
     transformation = True)
     # Let's run a bunch of different classification algorithms and rank them by their recall score
compare_models(sort = 'Recall')
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/48a1d1fa-8b25-4baf-a1ae-831ce0f26a33)

I will create three different models with the top 3 best ranked algorithms, tune them, blend them and see how this affects performance
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/c7f95cf8-cfd8-4da2-8760-267e81f8469a)

ridge = create_model('ridge')
lda = create_model('lda')
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/20d1c1a6-1631-4e4f-81ef-9be934c03720)
qda = create_model('qda')
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/b0d98176-a017-481c-ac3a-bcb1176398f1)
# Blending models 
blended_model = blend_models(estimator_list = [ridge, lda, qda],
                            fold = 10,
                            optimize = 'Recall',
                            choose_better = True)
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/b1ac48f3-b28c-498c-bf66-82c872378dcf)
No improvement made. I will now tune each one of the models.
tuned_ridge = tune_model(ridge,
                        n_iter = 1000,
                        optimize = 'Recall',
                        choose_better = True)
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/4093cdae-6418-408d-a711-31e1d06a8c12)
tuned_lda = tune_model(lda,
                        n_iter = 1000,
                        optimize = 'Recall',
                        choose_better = True)
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/f346f00c-ea13-4549-aa45-4949f9dc1115)
tuned_qda = tune_model(qda,
                        n_iter = 1000,
                        optimize = 'Recall',
                        choose_better = True)
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/87a21ee7-6b6a-4493-9e0b-0c6aa6a2d556)
All three models have the same recall score of 94.29%.
Tune_qda has the best Accuracy score, 82.68%.

plot_model(tuned_qda, plot = 'confusion_matrix')
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/7f977bfe-2c12-474c-aaf0-eb516103b82f)
plot_model(tuned_ridge, plot = 'confusion_matrix')
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/b9793c9e-5891-45b0-b85b-4b8674390a6f)
plot_model(tuned_lda, plot = 'confusion_matrix')
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/55a85d3d-810b-451c-aef1-03a5e8da7d9b)
From the graphs, it is clear that all three models have the same value for false negatives, with five defaulters predicted as non-defaulters.
Tuned_qda had the lowest number of false positives, 357.

plot_model(tuned_ridge, plot = 'feature')
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/f993d426-e8b8-4ecc-b1db-c2489d2d1b0d)
plot_model(tuned_lda, plot = 'feature')
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/51c4dada-d358-4a11-a02e-295b77be575a)
Bank Balance seems to be the most important feature to predict default, which is the feature with the highest positive correlation with the target variable

Finally, I will use my model on the hold-out sample to see how well it performs
predict_model(tuned_qda)
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/646fef68-cc49-4792-9fac-b43e188c4aa2)
I have achieved a recall score og 91.07% in my hold-out sample.
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/be4c6d8b-8e98-4659-9a43-5bdf433b2662)
Let me now see how well my model performs with unseen data using the test set that I have created before containing the last 20% of data from the original dataset
test # testing set
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/7b84a30c-1d77-4af4-b296-a174f8414069)
predictions = predict_model(model, data = test)
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/ad3a6fa7-cfdd-4d58-9646-1f19f3c2f709)
Recall Score on Testing Set ==> 91.04%

I will now print a confusion matrix
y_test = predictions.Defaulted
pred = predictions.Label
y_test.value_counts()
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/4514cfa4-186e-4ef7-a3c8-790dfd0d19ea)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(12,9))
ax = plt.subplot()
sns.heatmap(cm,annot = True, fmt ='g', ax = ax)
ax.set_xlabel('Predicted Class')
ax.set_ylabel('True Class')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Non-Defaulter','Defaulter'])
ax.yaxis.set_ticklabels(['Non-Defaulter','Defaulter'])
plt.show()
![image](https://github.com/mk2287/Predicting-Loan-Defaults-in-the-Banking-Industry/assets/152664423/5ec52bbc-8c5a-4ad6-9b58-bc4415fa3429)
There is only 6 false negatives in a dataset with 67 defaulters in total, where 61 of them were correctly predicted as defaulters.

4. Conclusion
Considering that labeling a defaulter as a non-defaulter represents a high chance of financial loss to the institution,
the goal of this model was to have the highest recall possible and correctly predict the highest amount of defaulters as possible.
Through an AutoML library referred to as PyCaret, I was able to successfully run 16 different classification algorithms and find the one with the best recall performance.
After tuning it and testing it, I was able to achieve a 91.04% recall score on unseen data and correctly predicted 61 defaulters among 67 of them.

