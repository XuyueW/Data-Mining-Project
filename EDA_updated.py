#%%
# EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from imblearn.over_sampling import SMOTE
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.stats.multicomp as mc
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_val_score

# %%
df = pd.read_csv('train.csv')
df.head()
df.info()
print("Shape of dataset:", df.shape)
print("Data types:\n", df.dtypes)
# check missing data
print("Number of missing values:\n", df.isna().sum())
# for small number of missing data, we can try fill it or drop those observations.
# for large number of categorical missing data, we can consider grouping the categories with missing values into a new category.
# for large number of numeric missing data, we might try to delete that variable.

# Employer_Category2, Var1, Approved are supposed to be categorical

# numeric variable summary
print(df.describe()) #Descriptive Statistics
print(df.corr()) #Correlation Analysis
print(df['Approved'].value_counts()) # Imbalance Target variable 
# df['Employer_Code'].unique()
# df['Customer_Existing_Primary_Bank_Code'].unique()
# df['Employer_Code'].unique()


#%%
# DATA CLEANING and some EDA
# missing value and convert data

df = df.drop_duplicates() # Remove duplicates
# drop ID after check duplicates 
df = df.drop('ID', axis=1)

df = df[(df['Monthly_Income'] >= 1500) & (df['Monthly_Income'] <= 15000)] 
# Handle outliers and display
plt.hist(df['Monthly_Income'], bins=50)
plt.xlabel('Monthly Income')
plt.ylabel('Frequency')
plt.title('Histogram of Monthly Income')
plt.show()
# By Approved Rate
df.groupby(['Monthly_Income'])['Approved'].mean().plot()
plt.title('Mean Approval Rate by Monthly Income')
plt.show()
# The approval rate is not monotonely increasing

# for 15 missing value in DOB, we can just drop those observations
df = df.dropna(subset=['DOB'])
# convert DOB to age in years

now = datetime.now()
df['age'] = now.year - pd.to_datetime(df['DOB'], format='%d/%m/%y').dt.year
# drop DOB column
df = df.drop('DOB', axis=1)

# We find age have unreal data including birth in the future, drop them
print(f"Number of positive values: {(df['age'] >= 0).sum()}")
print(f"Number of negative values: {(df['age'] < 0).sum()}")
df = df[df['age'] >= 16] # age lower than 16 is also considered unreal
plt.hist(df['age'], bins=50)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age')
plt.show()

# also convert and check Lead_Creation_Date 
df['lead_years'] = now.year - pd.to_datetime(df['Lead_Creation_Date'], format='%y/%m/%d').dt.year
df = df.drop('Lead_Creation_Date', axis=1)
print(df['lead_years'].unique())
print(f"Number of positive values: {(df['lead_years'] >= 0).sum()}")
print(f"Number of negative values: {(df['lead_years'] < 0).sum()}")
# drop negative values
df = df[df['lead_years'] >= 0] 
# now age and lead_years should be numeric

# Find the most frequent category in City_Code
most_frequent = df['City_Code'].mode()[0]
num_occurrences = (df['City_Code'] == most_frequent).sum()
print(f"The most frequent category in City_Code is {most_frequent}")
print(f"It occurs {num_occurrences} times")
# City_Code might be redundant
print( pd.crosstab(df['City_Code'], df['City_Category']))
print(df['City_Category'].value_counts())
# all City_Code are assigned to City_Category, drop City_Code
df = df.drop('City_Code', axis=1)

# Find the most frequent category in Employer_Code 
most_frequent = df['Employer_Code'].mode()[0]
num_occurrences = (df['Employer_Code'] == most_frequent).sum()
print(f"The most frequent category in Employer_Code is {most_frequent}")
print(f"It occurs {num_occurrences} times")
# if we fill missing value with mode, the distribution will change, better to drop them 
# df['Employer_Code'] = df['Employer_Code'].fillna(most_frequent)
# contingency table
print(pd.crosstab(df['Employer_Code'], df['Employer_Category1']))
print(df['Employer_Category1'].value_counts())
print(pd.crosstab(df['Employer_Code'], df['Employer_Category2']))
print(df['Employer_Category2'].value_counts())
# all Employer_Code are assigned to either category, drop Employer_Code
df = df.drop('Employer_Code', axis=1)
# # missing value in Employer_Category1 was dropped simultaneously
# contingency table for Employer_Category1 and Employer_Category2
print(pd.crosstab(df['Employer_Category1'], df['Employer_Category2']))
# convert Employer_Category2 to categorical
df['Employer_Category2'] = df['Employer_Category2'].map({1.0: 'one', 2.0: 'two',3.0:'three',4.0:'four'})

# Find the most frequent category in Customer_Existing_Primary_Bank_Code
most_frequent = df['Customer_Existing_Primary_Bank_Code'].mode()[0]
num_occurrences = (df['Customer_Existing_Primary_Bank_Code'] == most_frequent).sum()
print(f"The most frequent category in Customer_Existing_Primary_Bank_Code is {most_frequent}")
print(f"It occurs {num_occurrences} times")
print(pd.crosstab(df['Customer_Existing_Primary_Bank_Code'], df['Primary_Bank_Type']))
print(df['Primary_Bank_Type'].value_counts())
# all Customer Code are assigned to a binary Bank Type, drop customer Code
df = df.drop('Customer_Existing_Primary_Bank_Code', axis=1)

# Find the most frequent category in Loan_Amount
most_frequent = df['Loan_Amount'].mode()[0]
num_occurrences = (df['Loan_Amount'] == most_frequent).sum()
print(f"The most frequent category in Loan_Amount is {most_frequent}")
print(f"It occurs {num_occurrences} times")
# Handle outliers and display
df = df[df['Loan_Amount'] <= 150000] 
plt.hist(df['Loan_Amount'], bins=50)
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Loan Amount')
plt.show()
df.groupby(['Loan_Amount'])['Approved'].mean().plot()
plt.title('Mean Approval Rate by Loan_Amount')
plt.show()
# There is a trend of increasing in approval rate

# Find the most frequent category in Interest_Rate 
most_frequent = df['Interest_Rate'].mode()[0]
num_occurrences = (df['Interest_Rate'] == most_frequent).sum()
print(f"The most frequent category in Interest_Rate is {most_frequent}")
print(f"It occurs {num_occurrences} times")
# Handle outliers and display
df.Interest_Rate.describe()
plt.hist(df['Interest_Rate'], bins=50)
plt.xlabel('Interest Rate')
plt.ylabel('Frequency')
plt.title('Histogram of Interest Rate')
plt.show()
df.groupby(['Interest_Rate'])['Approved'].mean().plot()
plt.title('Mean Approval Rate by Interest Rate')
plt.show()
# Approval rate drop when reach about 20% (high risk applicant)

# drop missing values
df = df.dropna(subset=['Interest_Rate'])
# missing value in EMI was dropped simultaneously

# Also Handle outliers and display for EMI
df.EMI.describe()
df = df[df['EMI'] <= 3000] 
plt.hist(df['EMI'], bins=50)
plt.xlabel('EMI')
plt.ylabel('Frequency')
plt.title('Histogram of EMI')
plt.show()
# EMI distribution is approximately normal after cut off right tail
df.groupby(['EMI'])['Approved'].mean().plot()
plt.title('Mean Approval Rate by EMI')
plt.show()
# no obivious trend

# Equation of EMI(equated monthly installment) 
# EMI = P x R x (1+R)^N / [(1+R)^N-1]
# P: Principal loan amount (Loan_Amount)
# N: Loan tenure in months (Loan_Period)
# R: Interest rate per month (Interest_Rate)
# this indicate multicollinearity
# can affect the stability and interpretability of the model

# Check other variables, no missing value but might redundant
print(f"Number of Contacted Verified: {(df['Contacted'] == 'Y').sum()}")
# Contacted only left Y,drop it
df = df.drop('Contacted', axis=1)

# contingency table for Source
print(pd.crosstab(df['Source'], df['Source_Category']))
print(df['Source_Category'].value_counts())
# all Source_Category are assigned, drop Source
df = df.drop('Source', axis=1)

# scatter plot for Existing_EMI and EMI
plt.scatter(df['Existing_EMI'], df['EMI'])
plt.xlabel('Existing_EMI')
plt.ylabel('EMI')
plt.title('Scatter Plot of Existing_EMI vs. EMI')
plt.show()
# better to keep both

# contingency table for Var1
print(pd.crosstab(df['Var1'], df['Approved']))
print(df['Var1'].value_counts())
# unknown but meaningful variable

# check again if any missing value
print("Number of missing values:\n", df.isna().sum())
# drop missing values if any
df = df.dropna()
print("Shape of dataset:", df.shape)
# Shape of dataset: (13033, 16)

#%% 
# This part is dropped but keep code
# Dimensional Reduction 
# (City_Code , City_Category) 
# (Employer_Code,Employer_Category1,Employer_Category2)
# (Customer_Existing_Primary_Bank_Code,Primary_Bank_Type)
# (Source,Source_Category)

# Perform chi-square test of independence between Employer_Category1,Employer_Category2

observed = pd.crosstab(df['Employer_Category1'], df['Employer_Category2'])
chi2, p, dof, expected = chi2_contingency(observed)
print(f"Chi-square statistic: {chi2:.2f}")
print(f"P-value: {p:.2f}")

# Combine two highly correlated categorical variables using PCA

# (City_Code , City_Category) 
# X = pd.get_dummies(df[['City_Code', 'City_Category']])
# pca = PCA(n_components=1)
# df['City_Combined'] = pca.fit_transform(X)
# drop original variables
# df = df.drop(columns=['City_Code', 'City_Category'])

# %%
# extreme imbalance data (12708 vs 325), will affect the performance of machine learning
print(df['Approved'].value_counts())
# Solution 1: train this dataset however may have a poor prediction performance on the minority class
# Solution 2: Generate synthetic samples: You can use techniques like Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic samples of the minority class.

# correlation matrix for numeric variables
numerical_vars = ['Monthly_Income', 'Existing_EMI', 'Loan_Amount', 'Loan_Period','Interest_Rate','EMI']
corr_matrix = df[numerical_vars].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Variables')
plt.show()
# Note that high correlation coefficients (e.g., > 0.7) may indicate multicollinearity.
# Loan Amount vs EMI : 0.91
# Monthly Income vs EMI : 0.46
# Monthly Income vs Loan Amount : 0.41
# Loan Amount vs Loan Period : 0.37
# Monthly Income vs Interest Rate : -0.32
# Monthly Income vs Existing EMI : 0.29
# Interest Rate vs Loan Amount: -0.27

#%%
# SMOTE(generate synthetic samples)
# Here will create a clone of dataset but balanced in response variable for maching learning prediction.

# note SMOTE requires all variable involved are numeric
# categorical_cols = df.iloc[:, [0, 7, 8]]
# df_dummy = pd.get_dummies(df, columns=categorical_cols)

# create a copy of original dataset for resampling
df_dummy = df.copy()
df_dummy = pd.get_dummies(df,drop_first=True)
df_dummy.head()

#df_dummy['Gender'] = df_dummy['Gender'].map({'Female': 0, 'Male': 1})
#df_dummy['Approved'] = df_dummy['Approved'].astype(int)
#df_dummy['Var1'] = df_dummy['Var1'].astype(int)

# Apply SMOTE to balance binary response variable
smote = SMOTE()
X = df_dummy.drop('Approved', axis=1)
y = df_dummy['Approved']
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print the first few rows of the resampled data
resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
resampled_data.head()
print("Shape of balanced dataset:", resampled_data.shape)
# Shape of balanced dummy dataset: (25416, 23)
# We have a balanced dataset for machine learning
print(resampled_data['Approved'].value_counts())

# Comparison display

plt.scatter(df['age'], df['Monthly_Income'], c=df['Approved'], cmap='bwr',s=10, alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Monthly_Income')
plt.show()

plt.scatter(resampled_data['age'], resampled_data['Monthly_Income'], c=resampled_data['Approved'], cmap='bwr',s=10, alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Monthly_Income')
plt.show()

#%%
# EDA and Visualization 

# (We only need few of them in presenatation slides)

# Age and Approved density
sns.kdeplot(data=df, x='age', hue='Approved', fill=True)
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Density plot of Age and Approved')
plt.legend(title='Approved', loc='upper right', labels=['Yes', 'No'])
plt.show()

# approved_counts
approved_counts = df.groupby(['age', 'Approved']).size().reset_index(name='count')
print("Approved Counts by Age:\n", approved_counts)

# Age distribution group by approval
sns.violinplot(x="Approved", y="age", data=df)
plt.title('Distribution of Age by Approval Status')
plt.show()

# pie chart by gender
gender_counts = df["Gender"].value_counts()
labels = gender_counts.index
sizes = gender_counts.values
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Gender Distribution")
plt.show()

# barplot comparison
gender_approved = df.groupby(['Gender', 'Approved'])['Approved'].count().unstack()
ax = gender_approved.plot(kind='bar')
# add count labels to each bar
for i in ax.containers:
    ax.bar_label(i, label_type='edge', fontsize=10, padding=4)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Loan Approval by Gender')
plt.legend(title='Approved', labels=['No', 'Yes'])
plt.show()

# Use Seaborn to plot the correlation matrix between numerical variables
# sns.heatmap(df[['Monthly_Income', 'Loan_Amount', 'Loan_Period', 'Interest_Rate']].corr(), annot=True, cmap='coolwarm')
# plt.show()

# Use Seaborn to plot scatterplots between numerical variables
# sns.pairplot(df[['Monthly_Income', 'Loan_Amount', 'Loan_Period', 'Interest_Rate']])
# plt.show()

# Use Seaborn to plot a scatterplot matrix with hue by gender
# sns.pairplot(df[['Monthly_Income', 'Loan_Amount', 'Loan_Period', 'Interest_Rate', 'Gender']], hue='Gender')
# plt.show()

# Scatterplot for loan period and interest rate
sns.scatterplot(data=df, x='Interest_Rate', y='Loan_Period')
plt.xlabel('Loan Period')
plt.ylabel('Interest Rate')
plt.title('Scatterplot of Loan Period and Interest Rate')
plt.show()

# pie chart of Loan Period by year
df["Loan_Period"].value_counts()
plt.pie(df["Loan_Period"].value_counts().values,labels=df["Loan_Period"].value_counts().index,autopct="%1.1f%%")
plt.axis("equal")
plt.title("Loan_Period")
plt.show()

# status by loan period
sns.barplot(x="Loan_Period",y="Loan_Amount",hue = 'Approved',data=df)
plt.show() 
# Loan_Period 4 and 5 are two category years of having the higher Loan_Amount in the group of being approved

# the Gender and Loan_Amount
sns.barplot(x="Gender",y="Loan_Amount",hue = 'Approved',data=df)
plt.show() 
# Female is likely to get higher Loan_Amount compared with Male

# the Age_group and Loan_Amount
sns.barplot(x="age_group",y="Loan_Amount",hue = 'Approved',data=df)
plt.show() 
# 40-50 Age group is likely to get higher Loan_Amount

#%%
# Analysis of Differences

# Run a ANOVA test to evaluate whether different Loan_Period categories in the group of approval have diiferent Loan_Amount 

df_approved = df[df['Approved'] == 1]

# create a dictionary to store the 'Loan_Amount' values for each 'Loan_Period' category
loan_period_dict = {}
for period in df_approved['Loan_Period'].unique():
    loan_period_dict[period] = df_approved[df_approved['Loan_Period'] == period]['Loan_Amount']

# perform ANOVA test
f, p = stats.f_oneway(*loan_period_dict.values())
print('F-statistic:', f)
print('p-value:', p)

# The ANOVA test result shows an F-statistic of 17.36 and a very low p-value of 5.42e-13. 
# This suggests that there is a significant difference in the mean 'Loan_Amount' between at least two 'Loan_Period' categories in the group of approved loans. 

# Perform  post-hoc tests (such as Tukey's HSD test) to determine which pairs of 'Loan_Period' categories have significantly different mean 'Loan_Amount' values.

# perform Tukey's HSD post-hoc test
tukey_results = mc.MultiComparison(df_approved['Loan_Amount'], df_approved['Loan_Period']).tukeyhsd()
print(tukey_results)

# Group 1.0 is different with Group 3, 4, 5
# Group 2 is different with Group 4 ,5
# Group 3 is different with Group 5

#  Run a  T-test to evaluate whether different Gender categories in the group of approval have diiferent Loan_Amount 

# separate approved loans by gender
df_approved_male = df_approved[df_approved['Gender']==1]
df_approved_female = df_approved[df_approved['Gender']==0]

# perform independent t-test
t, p = ttest_ind(df_approved_male['Loan_Amount'], df_approved_female['Loan_Amount'], equal_var=False)
print(f"T-statistic: {t:.2f}") # -1.33
print(f"P-value: {p:.2f}") # 0.19

# If the t-statistic returned by the independent t-test is negative and the p-value is greater than the significance level (e.g., 0.05), 
# it suggests that there is no significant difference in the mean 'Loan_Amount' between the two 'Gender' categories. 
# In other words, there is no evidence to suggest that 'Gender' is a significant predictor of 'Loan_Amount' in this dataset.

#%%
# A linear regression for approved group: 
# Predict how much Loan_amount an applicant will be approved
# Independet variable: Loan_Period(Categorical), Monthly_income, Interest_Rate,Age
# dependent vaiable: Loan_Amount

Loan_Period_dummies = pd.get_dummies(df_approved['Loan_Period'], prefix='Loan_Period')

# Define the independent variables
X = pd.concat([df_approved[['Monthly_Income', 'Interest_Rate', 'age']], Loan_Period_dummies], axis=1)

# Define the dependent variable
y = df_approved['Loan_Amount']

# Add a constant to the independent variables
X = sm.add_constant(X)

# Fit the multiple linear regression model
lmod = sm.OLS(y, X).fit()

# Print the model summary
print(lmod.summary())

# perform 5-fold cross-validation
ols = LinearRegression()
scores = cross_val_score(ols,  X, y, cv=5, scoring='neg_mean_squared_error')

# print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", np.mean(scores))

# Convert the scores to mean squared error (MSE) by taking the negative
mse_scores = -scores
# Compute the root mean squared error (RMSE) for each fold
rmse_scores = np.sqrt(mse_scores)
# Print the mean and standard deviation of the RMSE scores
print("RMSE: %0.2f (+/- %0.2f)" % (np.mean(rmse_scores), np.std(rmse_scores) * 2))

#%%
# Logistic model
# build a full model without feature selection
# logistic regression for the whole dataframe (use original data)

# df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
# df['Approved'] = df['Approved'].astype(int)
# df['Var1'] = df['Var1'].astype(int)

X_0 = df_dummy.drop('Approved', axis=1)
y_0 = df_dummy['Approved']

model_0 = sm.Logit(y_0, sm.add_constant(X_0)).fit()
print(model_0.summary())
# Pseudo R-squared 0.1200
# insignificant variable: 
# Monthly_Income,Loan_Amount,Loan_Period,EMI,age,lead_years,Employer_Category1,Primary_Bank_Type,

# logistic regression for the whole dataframe (use balanced data)
X = resampled_data.drop('Approved', axis=1)
y = resampled_data['Approved']
model_balanced = sm.Logit(y, sm.add_constant(X)).fit()
print(model_balanced.summary())
# Pseudo R-squared 0.4059
# insignificant variable: lead_years, Gender

# Pseudo R-squared and log-likelihood are commonly used metrics for evaluating the performance of logistic regression models,
# but they have their limitations and should be used in conjunction with other evaluation methods.
# The model uses balanced data is better in the goodness of fit for prediction
# usually drop insignificant coeffients (p-value > 0.05), but we may decide later
  
# Check variance inflation factor (VIF) for multicollinearity
vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
vif_df = pd.DataFrame({'Variable': X.columns, 'VIF': vif})
print(vif_df.sort_values('VIF', ascending=False))
# Note that a high VIF value (e.g., > 10) may indicate multicollinearity.

#%%
# Logistic model with feature selection

# 86.4% male and 13.6% female before resampling
print(df['Gender'].value_counts())
# 85% male and 15% female after resampling
print(resampled_data['Gender_Male'].value_counts())
 
grouped = df.groupby(['Gender', 'Approved'])
counts = grouped.size()
table = counts.unstack()
print(table)
# 0.015 in female and 0.026 in male

grouped = resampled_data.groupby(['Gender_Male', 'Approved'])
counts = grouped.size()
table = counts.unstack()
print(table)
# 0.544 in female and 0.492 in male
# SMOTE generate balanced data but also erase the difference in approval rate by gender
# We should keep gender

bin=df.groupby(['lead_years'])['Approved'].mean()
bin=bin.reset_index()
plt.bar(bin.lead_years, bin.Approved)
plt.title('Approval Rate by Lead Years')
plt.xlabel('Lead Years')
plt.ylabel('Approval Rate')
plt.show()
# it is just a date on which Lead was created, we should drop

# For those insignificant variable in orginal data but significant in balanced data:
# We can drop some based on plots of mean approval rate

# histogram of age and approval is normally distributed,
bin=df.groupby(['age'])['Approved'].mean()
bin=bin.reset_index()
plt.bar(bin.age, bin.Approved)
plt.title('Approval Rate by Age')
plt.xlabel('Age')
plt.ylabel('Approval Rate')
plt.show()
# cant find a trend, drop age

# Loan_Period
bin=df.groupby(['Loan_Period'])['Approved'].mean()
bin=bin.reset_index()
plt.bar(bin.Loan_Period, bin.Approved)
plt.title('Approval Rate by Loan_Period')
plt.xlabel('Loan_Period')
plt.ylabel('Approval Rate')
plt.show()
# drop loan Period

# Some plots in previous sections
# Drop Monthly_Income, EMI
# Keep Loan_Amount 

# Employer_Category1
bin=df.groupby(['Employer_Category1'])['Approved'].mean()
bin=bin.reset_index()
plt.bar(bin.Employer_Category1, bin.Approved)
plt.title('Approval Rate by Employer_Category1')
plt.xlabel('Employer_Category1')
plt.ylabel('Approval Rate')
plt.show()
# Keep

# Primary_Bank_Type
bin=df.groupby(['Primary_Bank_Type'])['Approved'].mean()
bin=bin.reset_index()
plt.bar(bin.Primary_Bank_Type, bin.Approved)
plt.title('Approval Rate by Primary_Bank_Type')
plt.xlabel('Primary_Bank_Type')
plt.ylabel('Approval Rate')
plt.show()
# Keep

# Selected model with original data
y_0 = df_dummy['Approved']
X_0 = df_dummy.drop(['Approved','Monthly_Income', 'Loan_Period','EMI','age','lead_years'], axis=1)

model_0 = sm.Logit(y_0, sm.add_constant(X_0)).fit()
print(model_0.summary())
# insignificant variable: Loan_Amount , Primary_Bank_Type_P
# drop insiginificant
X_0 = df_dummy.drop(['Approved','Monthly_Income', 'Loan_Period','EMI','age','lead_years','Loan_Amount','Primary_Bank_Type_P'], axis=1)
model_0 = sm.Logit(y_0, sm.add_constant(X_0)).fit()
print(model_0.summary())

# Selected model with balanced data
y_1 = resampled_data['Approved']
X_1 = resampled_data.drop(['Approved','Gender_Male','lead_years'], axis=1)

model_1 = sm.Logit(y_1, sm.add_constant(X_1)).fit()
print(model_1.summary())
# no insignificant variable, but we know bias exist, eg., in female because of SMOTE

# Train both and evaluate 

# Train imbalanced model and evaluate the performance
X_train, X_test, y_train, y_test = train_test_split(X_0,y_0, test_size=0.2, random_state=42)

# Define the logistic regression model
lr = LogisticRegression()

# Fit the model on the training set
lr.fit(X_train, y_train)

# Use the model to make predictions on the testing set
y_pred = lr.predict(X_test)

# Evaluate the performance of the model using accuracy score and classification report
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Create a confusion matrix
from sklearn.metrics import confusion_matrix
cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
cm['Total'] = np.sum(cm, axis=1)
cm = cm.append(np.sum(cm, axis=0), ignore_index=True)
cm.columns = ['Predicted No', 'Predicted Yes', 'Total']
cm = cm.set_index([['Actual No', 'Actual Yes', 'Total']])
print(cm)

# Train  balanced model and evaluate the performance
X_train, X_test, y_train, y_test = train_test_split(X_1,y_1, test_size=0.2, random_state=42)

# Define the logistic regression model
lr = LogisticRegression()

# Fit the model on the training set
lr.fit(X_train, y_train)

# Use the model to make predictions on the testing set
y_pred = lr.predict(X_test)

# Evaluate the performance of the model using accuracy score and classification report
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Create a confusion matrix
from sklearn.metrics import confusion_matrix
cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
cm['Total'] = np.sum(cm, axis=1)
cm = cm.append(np.sum(cm, axis=0), ignore_index=True)
cm.columns = ['Predicted No', 'Predicted Yes', 'Total']
cm = cm.set_index([['Actual No', 'Actual Yes', 'Total']])
print(cm)

# ROC Curve

# Compute predicted probabilities for the test set
y_score = lr.predict_proba(X_test)[:, 1]

# Calculate ROC curve values: false positive rate (fpr), true positive rate (tpr), and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_score)

# Calculate the AUC-ROC score
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# usually AUC > 0.8 indicates good fitted

intercept = lr.intercept_
coeff = lr.coef_
coef_list = list(coeff[0,:])
coef_df = pd.DataFrame({'Feature': list(X_train.columns),'Coefficient': coef_list})
print(coef_df)

feat_importances = coef_df #what we created before for coeff
feat_importances['importances'] = np.abs(feat_importances['Coefficient']) #coeff is feature importance
feat_importances.sort_values(by='importances', ascending=False, inplace=True)
print(feat_importances) 
# Var1 and Interest Rate contribute more

# Conclusion


#%%
# Unused Models

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['Monthly_Income', 'Loan_Amount', 'Loan_Period', 'Interest_Rate']], df['Approved'], test_size=0.2, random_state=42)

# Define the logistic regression model
lr = LogisticRegression()

# Fit the model on the training set
lr.fit(X_train, y_train)

# Use the model to make predictions on the testing set
y_pred = lr.predict(X_test)

# Evaluate the performance of the model using accuracy score and classification report
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# An accuracy score of 0.974 in a logistic regression model may indicate overfitting

# KNN model

# Split data into training and test sets
X_amount_EMI = df[['Loan_Amount', 'EMI']]
y_approved= df['Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Evaluate the KNN model
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# An accuracy score of 0.974 in a logistic regression model may indicate overfitting

#%%
