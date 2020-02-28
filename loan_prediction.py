import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#About the data
#Loan Id -> Unique Id for each loan
#Gender -> M/F
#Married -> Y/N
#Dependents -> No. of dependents on the person asking for loan
#Educationc ->G/NG If educated chances are higher
#Self_employed ->Y/N
#ApplicantIncome -> Amt (High income more chances)
#CoapplicantIncome ->Amt (High income more chances)
#LoanAmt -> Amt (High amt, chances of repaying will be less)
#loan_amt_term -> months required to repay the loan
#Credit history ->1/0
#Property area ->R/U
#loan status ->Y/N

#EDA
# 1.More salary has more chances of loan approval
# 2.Graduated person has more chances of loan approval
# 3.Married person has more chances of loan approval
# 4.The applicant who has less number of dependents have a high probability for loan approval.
# 5.lesser the loan amount the higher the chance for getting loan.


train['Gender'].value_counts(normalize = True).plot.bar(title = 'Gender')
plt.show()
#80% of the people are male

train['Dependents'].value_counts(normalize = True).plot.bar(title = 'No.of dependents')
plt.show()
#60% of the people have no dependents

train['Education'].value_counts(normalize = True).plot.bar(title = 'Education')
plt.show()
#77% of the people have are educated

train['Property_Area'].value_counts(normalize = True).plot.bar(title = 'Area')
plt.show()
#Semi Urban applicants are slightly more than Urban

train['Married'].value_counts(normalize = True).plot.bar(title = 'Married')
plt.show()
#70% of the people are married

train['Credit_History'].value_counts(normalize = True).plot.bar(title = 'Credit history')
plt.show()
#80% of the people meets Credit history

train['Loan_Status'].value_counts(normalize = True).plot.bar(title = 'Status of loan')
plt.show()
#68% of the people meets Credit history

#let us try to find if Applicant income can exactly separate the Loan_Status. 
sns.FacetGrid(train,hue = 'Loan_Status').map(sns.distplot,'ApplicantIncome').add_legend()
plt.show()
#We can see that we cannot decide only based on ApplicantIncome

sns.FacetGrid(train,hue = 'Loan_Status').map(plt.scatter,"Credit_History","ApplicantIncome").add_legend()
plt.show()
#This also seems not convincing

tab = pd.crosstab(train['Married'],train['Loan_Status'])
tab.plot.bar()
plt.show()
#married people have got loan approved when compared to non- married people.

tab = pd.crosstab(train['Dependents'],train['Loan_Status'])
tab.plot.bar()
plt.show()
#applicants with 0 dependents have got their loan approved.

tab = pd.crosstab(train['Education'],train['Loan_Status'])
tab.plot.bar()
plt.show()
#Graduated applicants have got their loan approved.

tab = pd.crosstab(train['Self_Employed'],train['Loan_Status'])
tab.plot.bar()
plt.show()
#No conclusion could be made from this.

print(train.isnull().sum())
#Except the Loan Amount and Loan_Amount_Term everything else which is missing is of type categorical. 
#Hence we can replace the missing values by mode of that particular column.

train['Gender'].fillna(train['Gender'].mode()[0],inplace = True)
train['Married'].fillna(train['Married'].mode()[0],inplace = True)
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace = True)
train['Education'].fillna(train['Education'].mode()[0],inplace = True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace = True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace = True)

train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace = True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].median(),inplace = True)

train = train.drop('Loan_ID',axis = 1)

X = train.drop('Loan_Status',axis = 1)
y = train['Loan_Status']


# Converting categorical values to Numerical values
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.45,random_state=1)
'''from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()'''

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

'''from sklearn import svm
clf = svm.SVC(kernel='linear',probability=True)
'''

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))

test['Gender'].fillna(test['Gender'].mode()[0],inplace = True)
test['Married'].fillna(test['Married'].mode()[0],inplace = True)
test['Dependents'].fillna(test['Dependents'].mode()[0],inplace = True)
test['Education'].fillna(test['Education'].mode()[0],inplace = True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0],inplace = True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0],inplace = True)

test['LoanAmount'].fillna(test['LoanAmount'].median(),inplace = True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].median(),inplace = True)

Loan_Id = test['Loan_ID']
test = test.drop('Loan_ID',axis = 1)
test = pd.get_dummies(test)

final_predictions = clf.predict(test)

op=list()
loanid = list(Loan_Id)
for i in range(len(loanid)):
        op.append([loanid[i],final_predictions[i]])
csvdata=op
import csv
    
with open('submission_NB.csv', 'w',newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(["Loan_ID", "Loan_Status"])
    writer.writerows(csvdata)
csvFile.close()