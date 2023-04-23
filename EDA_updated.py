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
# drop ID
df = df.drop('ID', axis=1)
print("Shape of dataset:", df.shape)
print("Data types:\n", df.dtypes)
# check missing data
print("Number of missing values:\n", df.isna().sum())
# for small number of missing data, we can try fill it or drop those observations.
# for large number of categorical missing data, we can consider grouping the categories with missing values into a new category.
# for large number of numeric missing data, we might try to delete that variable.
# convert Employer_Category2 into categorical
df['Employer_Category2'] = df['Employer_Category2'].astype(str)
# convert Var1 (Var1: Anonymized Categorical variable with multiple levels)
df['Var1'] = df['Var1'].astype(str)
# convert Approved (Whether a loan is Approved or not (1-0) . Customer is Qualified Lead or not (1-0))
df['Approved'] = df['Approved'].astype(str)
print("Data types:\n", df.dtypes)
# EDA
# numeric variable summary
print(df.describe()) #Descriptive Statistics
print(df.corr()) #Correlation Analysis
sns.countplot(x='Approved', data=df) # Target variable distribution
# df['Employer_Code'].unique()
# df['Customer_Existing_Primary_Bank_Code'].unique()
# df['Employer_Code'].unique()

#%%
# DATA CLEANING
# missing value and convert data

df = df.drop_duplicates() # Remove duplicates
df = df.dropna() # Remove missing values
df = df[(df['Monthly_Income'] >= 1500) & (df['Monthly_Income'] <= 25000)] # Handle outliers
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
df = df[df['age'] >= 0] 

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
# it is not a good idea to fill with mode as number of mode is large
# drop missing values
df = df.dropna(subset=['City_Code'])
# missing value in City_Category was dropped simultaneously 

# Find the most frequent category in Employer_Code 
most_frequent = df['Employer_Code'].mode()[0]
num_occurrences = (df['Employer_Code'] == most_frequent).sum()
print(f"The most frequent category in Employer_Code is {most_frequent}")
print(f"It occurs {num_occurrences} times")
# if we fill missing value with mode, the distribution will change, better to drop them 
# df['Employer_Code'] = df['Employer_Code'].fillna(most_frequent)
df = df.dropna(subset=['Employer_Code'])
# # missing value in Employer_Category1 was dropped simultaneously 

# Find the most frequent category in Customer_Existing_Primary_Bank_Code
most_frequent = df['Customer_Existing_Primary_Bank_Code'].mode()[0]
num_occurrences = (df['Customer_Existing_Primary_Bank_Code'] == most_frequent).sum()
print(f"The most frequent category in Customer_Existing_Primary_Bank_Code is {most_frequent}")
print(f"It occurs {num_occurrences} times")
# drop missing values
df = df.dropna(subset=['Customer_Existing_Primary_Bank_Code'])
# missing value in Primary_Bank_Type was dropped simultaneously 

# Find the most frequent category in Loan_Amount
most_frequent = df['Loan_Amount'].mode()[0]
num_occurrences = (df['Loan_Amount'] == most_frequent).sum()
print(f"The most frequent category in Loan_Amount is {most_frequent}")
print(f"It occurs {num_occurrences} times")
# drop missing values
df = df.dropna(subset=['Loan_Amount'])
# missing value in Loan_Period was dropped simultaneously

# Find the most frequent category in Interest_Rate 
most_frequent = df['Interest_Rate'].mode()[0]
num_occurrences = (df['Interest_Rate'] == most_frequent).sum()
print(f"The most frequent category in Interest_Rate is {most_frequent}")
print(f"It occurs {num_occurrences} times")
# drop missing values
df = df.dropna(subset=['Interest_Rate'])
# missing value in EMI was dropped simultaneously

# Contacted only left Y,drop it
print(f"Number of Contacted Verified: {(df['Contacted'] == 'Y').sum()}")
df = df.drop('Contacted', axis=1)

# check again if any missing value
print("Number of missing values:\n", df.isna().sum())
print("Shape of dataset:", df.shape)
# Shape of dataset: (13525, 20)

#%% 
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
X = pd.get_dummies(df[['City_Code', 'City_Category']])
pca = PCA(n_components=1)
df['City_Combined'] = pca.fit_transform(X)
# drop original variables
df = df.drop(columns=['City_Code', 'City_Category'])

# (Employer_Code,Employer_Category1,Employer_Category2)(will run for a few minute)
X = pd.get_dummies(df[['Employer_Code','Employer_Category1','Employer_Category2']])
pca = PCA(n_components=1)
df['Employer_Combined'] = pca.fit_transform(X)
# drop original variables
df = df.drop(columns=['Employer_Code','Employer_Category1','Employer_Category2'])

# (Customer_Existing_Primary_Bank_Code,Primary_Bank_Type)
X = pd.get_dummies(df[['Customer_Existing_Primary_Bank_Code','Primary_Bank_Type']])
pca = PCA(n_components=1)
df['Bank_Type_Combined'] = pca.fit_transform(X)
# drop original variables
df = df.drop(columns=['Customer_Existing_Primary_Bank_Code','Primary_Bank_Type'])

# check independence of (Source,Source_Category)
observed = pd.crosstab(df['Source'], df['Source_Category'])
chi2, p, dof, expected = chi2_contingency(observed)
print(f"Chi-square statistic: {chi2:.2f}")
print(f"P-value: {p:.2f}")

# (Source,Source_Category)
X = pd.get_dummies(df[['Source','Source_Category']])
pca = PCA(n_components=1)
df['Source_Combined'] = pca.fit_transform(X)
# drop original variables
df = df.drop(columns=['Source','Source_Category'])

# %%
# bar plot of Approved
plt.figure(figsize=(6,4))
plt.bar(df['Approved'].unique(), df['Approved'].value_counts())
plt.title('Distribution of Approved Status') 
plt.xlabel('Approved Status')
plt.ylabel('Count')
plt.show()
# extreme imbalance data (14060 vs 350), will affect the performance of machine learning
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

# in case of confusion, create a copy of dataset 
df_dummy = df.copy()
df_dummy['Gender'] = df_dummy['Gender'].map({'Female': 0, 'Male': 1})
df_dummy['Approved'] = df_dummy['Approved'].astype(int)
df_dummy['Var1'] = df_dummy['Var1'].astype(int)

# Apply SMOTE to balance binary response variable
smote = SMOTE()
X = df_dummy.drop('Approved', axis=1)
y = df_dummy['Approved']
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print the first few rows of the resampled data
resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
resampled_data.head()
print("Shape of balanced dataset:", resampled_data.shape)
# Shape of balanced dataset: (26356, 15)
# We have a balanced dataset for machine learning
print(resampled_data['Approved'].value_counts())


#%%
# Visualization for feature selection

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

# Interest_Rate vs Existing_EMI
interest_rate = df["Interest_Rate"]
existing_emi = df["Existing_EMI"]
plt.scatter(interest_rate, existing_emi)
plt.xlabel("Interest Rate")
plt.ylabel("Existing EMI")
plt.title("Relationship between Interest Rate and Existing EMI")
plt.show()

# Monthly Income, Existing EMI, and Loan Amount
sns.scatterplot(x='Monthly_Income', y='Existing_EMI', size='Loan_Amount', data=df)
# add labels and title
plt.xlabel('Monthly Income')
plt.ylabel('Existing EMI')
plt.title('Relationship between Monthly Income, Existing EMI, and Loan Amount')

# Loan Amount vs. Age and Gender
approved_df = df[df['Approved'] == 1]
sns.stripplot(x="age", y="Loan_Amount", hue="Gender", data=df, dodge=True)
plt.xlabel("Age")
plt.ylabel("Loan Amount")
plt.title("Loan Amount vs. Age and Gender")
plt.show()

# Loan Amount by Age Group and Gender
bins = [18, 30, 40, 50, 60, 70, 80, 90]
labels = ['18-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
sns.stripplot(x='age_group', y='Loan_Amount', hue='Gender', data=df)
plt.xlabel('Age Group')
plt.ylabel('Loan Amount')
plt.title('Loan Amount by Age Group and Gender')
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

# Scatterplot for monthly income and interest rate
sns.scatterplot(data=df, x='Monthly_Income', y='Interest_Rate')
plt.xlabel('Monthly Income')
plt.ylabel('Interest Rate')
plt.title('Scatterplot of Monthly Income and Interest Rate')
plt.show()

# Scatterplot for loan amount and interest rate
sns.scatterplot(data=df, x='Loan_Amount', y='Interest_Rate')
plt.xlabel('Loan Amount')
plt.ylabel('Interest Rate')
plt.title('Scatterplot of Loan Amount and Interest Rate')
plt.show()

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
# build a full model without feature selection
# logistic regression for the whole dataframe (use original data)

df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
df['Approved'] = df['Approved'].astype(int)
df['Var1'] = df['Var1'].astype(int)

X_0 = df.drop('Approved', axis=1)
y_0 = df['Approved']

model_0 = sm.Logit(y_0, sm.add_constant(X_0)).fit()
print(model_0.summary())

# logistic regression for the whole dataframe (use balanced data)
X = resampled_data.drop('Approved', axis=1)
y = resampled_data['Approved']
model_balanced = sm.Logit(y, sm.add_constant(X)).fit()
print(model_balanced.summary())

# outputs of balanced model is relatively better than the model using the original data
# usually drop insignificant coeffients (p-value > 0.05), but we may decide later
# significant variable: Gender , Existing_EMI , Interest_Rate , Var1 

# Check variance inflation factor (VIF) for multicollinearity
vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
vif_df = pd.DataFrame({'Variable': X.columns, 'VIF': vif})
print(vif_df.sort_values('VIF', ascending=False))
# Note that a high VIF value (e.g., > 10) may indicate multicollinearity.

#%%
# This logistic model tries to drop all high VIF variables
y_1 = resampled_data['Approved']
X_1 = resampled_data[['Gender','Var1','Monthly_Income','lead_years','Existing_EMI','Bank_Type_Combined','City_Combined','Employer_Combined','Source_Combined']]
model_1 = sm.Logit(y_1, sm.add_constant(X_1)).fit()
print(model_1.summary())
# significant variable: Gender , Var1 , Existing_EMI , Bank_Type_Combined , Employer_Combined

# Train the model and evaluate the performance
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

#%%

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

# %%
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
