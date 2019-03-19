# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:12:18 2019

@author: masal
"""
####################
# Import Libraries #
####################


import pandas as pd
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 250)
pd.set_option('display.width', 250)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import statsmodels.formula.api as smf # regression modeling
import sklearn.metrics # more metrics for model performance evaluation
from sklearn.model_selection import cross_val_score # k-folds cross validation
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor # Regression trees
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier # KNN for Regression




######################################
# Load File into Working Environment #
######################################

file = "C:/Users/masal/Desktop/ML_Class/Group_Project/birthweight_feature_set.xlsx"
birthweight = pd.read_excel(file)




###############################################################################
####################             MISSING VALUES            ####################
###############################################################################


# What variables have missing values
print(birthweight.isnull().sum())



# Noting which quartile the birthweight is in then taking the median of each 
# quartile per column with missing values and using those values to 
# impute the missing values


# Flagging Missing Values
for col in birthweight:
    
    if birthweight[col].isnull().any():
        birthweight['m_'+col] = birthweight[col].isnull().astype(
                int)
        
#######################
# Imputation #
#######################
# Adding new column, 'quantile', to note quantile of 'bwght'
birthweight['quantile'] = pd.qcut(birthweight['bwght'], 
                  10, labels=False) + 1



# Subsetting dataset based on quantiles for 'bwght'
bwght_1 = birthweight[birthweight['quantile'] == 1]

bwght_2 = birthweight[birthweight['quantile'] == 2]

bwght_3 = birthweight[birthweight['quantile'] == 3]

bwght_4 = birthweight[birthweight['quantile'] == 4]

bwght_5 = birthweight[birthweight['quantile'] == 5]

bwght_6 = birthweight[birthweight['quantile'] == 6]

bwght_7 = birthweight[birthweight['quantile'] == 7]

bwght_8 = birthweight[birthweight['quantile'] == 8]

bwght_9 = birthweight[birthweight['quantile'] == 9]

bwght_10 = birthweight[birthweight['quantile'] == 10]


bwght_low = pd.DataFrame.sort_index(pd.concat([bwght_1]))

bwght_avg = pd.DataFrame.sort_index(pd.concat([bwght_2, bwght_3, 
                                               bwght_4, bwght_5,
                                               bwght_6, bwght_7, 
                                               bwght_8, bwght_9]))
  
bwght_hi = pd.DataFrame.sort_index(pd.concat([bwght_10]))



# Imputing the missing values based on median on quartile divided dataset


# Filling 'bwght_low' dataframe
fill = bwght_low['meduc'].median()

bwght_low['meduc'] = bwght_low['meduc'].fillna(fill)


fill = bwght_low['npvis'].median()

bwght_low['npvis'] = bwght_low['npvis'].fillna(fill)


fill = bwght_low['feduc'].median()

bwght_low['feduc'] = bwght_low['feduc'].fillna(fill)


# Filling 'bwght_avg' dataframe
fill = bwght_avg['meduc'].median()

bwght_avg['meduc'] = bwght_avg['meduc'].fillna(fill)


fill = bwght_avg['npvis'].median()

bwght_avg['npvis'] = bwght_avg['npvis'].fillna(fill)


fill = bwght_avg['feduc'].median()

bwght_avg['feduc'] = bwght_avg['feduc'].fillna(fill)


# Filling 'bwght_hi' dataframe
fill = bwght_hi['meduc'].median()

bwght_hi['meduc'] = bwght_hi['meduc'].fillna(fill)


fill = bwght_hi['npvis'].median()

bwght_hi['npvis'] = bwght_hi['npvis'].fillna(fill)


fill = bwght_hi['feduc'].median()

bwght_hi['feduc'] = bwght_hi['feduc'].fillna(fill)



# Appending all quartile dataframes
birthweight = pd.DataFrame.sort_index(pd.concat(
        [bwght_low, bwght_avg, bwght_hi]))



# Checking there are no more missing values
print(
      birthweight
      .isnull()
      .sum()
      )


###############################################################################
####################       EXPLORATORY DATA ANALYSIS       ####################
###############################################################################


#####################################
###      UNIVARIATE ANALYSIS      ###
#####################################


#######################
# Continuous Variables #
#######################


# Target Variable - bwght (birthweight)
plt.subplot(2,2,1)
sns.distplot(birthweight['bwght'])
plt.xlabel('Birthweight')

# Mother's Age - mage
plt.subplot(2,2,2)
sns.distplot(birthweight['mage'])
plt.xlabel('Mothers Age')

# Mother's Education - meduc
plt.subplot(2,2,3)
sns.distplot(birthweight['meduc'])
plt.xlabel('Mothers Education')
plt.xticks(np.arange(0,21, step = 2))

# Total Number Prenatal Visits - npvis
plt.subplot(2,2,4)
sns.distplot(birthweight['npvis'])
plt.xlabel('Number Prenatal Visits')

plt.show()

# We see mainly approximate normal distribution, except Mother's Education
# Mother's education has bimodal distribution, maybe meduc should be split in 2 categories


###############################################################################


# Father's Age - fage
plt.subplot(2,2,1)
sns.distplot(birthweight['fage'])
plt.xlabel('Fathers Age')

# Father's Education - feduc
plt.subplot(2,2,2)
sns.distplot(birthweight['feduc'])
plt.xlabel('Fathers Education')
plt.xticks(np.arange(0,21, step = 2))

# Avg. Cigarettes per Day- cigs
plt.subplot(2,2,3)
sns.distplot(birthweight['cigs'])
plt.xlabel('Avg. Cigarettes per Day')

# Avg. Drink per Weekd - drinks
plt.subplot(2,2,4)
sns.distplot(birthweight['drink'])
plt.xlabel('Avg. Drinks per Week')


plt.show()

# We see mainly approximate normal distributions, except Father's Education
# We see that cigarettes and drinks are slightly skewed to the left


#########################
# Categorical Variables #
#########################

# If Baby Male - male
plt.subplot(2,2,1)
sns.distplot(birthweight['male'])
plt.xlabel('If Baby Male : 1 is Yes')
plt.xticks(np.arange(0, 2, step = 1))

# If Mother White - mwhte
plt.subplot(2,2,2)
sns.distplot(birthweight['mwhte'])
plt.xlabel('If Mother White : 1 is Yes')
plt.xticks(np.arange(0, 2, step = 1))

# If Mother Black - mblck
plt.subplot(2,2,3)
sns.distplot(birthweight['mblck'])
plt.xlabel('If Mother Black : 1 is Yes')
plt.xticks(np.arange(0, 2, step = 1))

# If Mother Other Ethnicity - moth
plt.subplot(2,2,4)
sns.distplot(birthweight['moth'])
plt.xlabel('If Mother Other Ethnicity : 1 is Yes')
plt.xticks(np.arange(0, 2, step = 1))

plt.show()

# We see a slightly higher majority of male babies
# We see more mothers who are black or other than white


###############################################################################

# If Father White - fwhte
plt.subplot(2,2,1)
sns.distplot(birthweight['fwhte'])
plt.xlabel('If Father White : 1 is Yes')
plt.xticks(np.arange(0, 2, step = 1))

# If Father Black - fblck
plt.subplot(2,2,2)
sns.distplot(birthweight['fblck'])
plt.xlabel('If Father Black : 1 is Yes')
plt.xticks(np.arange(0, 2, step = 1))

# If Father Other Ethnicity - foth
plt.subplot(2,2,3)
sns.distplot(birthweight['foth'])
plt.xlabel('If Father Other Ethnicity : 1 is Yes')
plt.xticks(np.arange(0, 2, step = 1))

plt.show()


#We see a more even split between ethnicities of the father


###############################################################################


# Month Prenatal Care Began - monpre
plt.subplot(2,2,1)
sns.distplot(birthweight['monpre'])
plt.xlabel('Month Prenatal Care Began')
plt.xticks(np.arange(0, 10, step = 1))

# One-Minute APGAR Score - omaps
plt.subplot(2,2,2)
sns.distplot(birthweight['omaps'])
plt.xlabel('One-Minute APGAR Score')
plt.xticks(np.arange(0, 10, step = 1))

# Five-Minute APGAR Score - fmaps
plt.subplot(2,2,3)
sns.distplot(birthweight['fmaps'])
plt.xlabel('Five-Minute APGAR Score')
plt.xticks(np.arange(0, 10, step = 1))

plt.show()



#####################################
###      BIVARIATE ANALYSIS       ###
#####################################


####################################
# Dep Var. vs. Ind. Continuous Var #
####################################

"""
Assumed Continuous Independent Variables:
Mage
Meduc
Fage
Feduc
Npvis
Cigs
Drink
"""

### Age & Education of Parents ###

# Birthweight vs. Mother's Age
plt.subplot(2,2,1)
sns.scatterplot(x = birthweight['bwght'], y = birthweight['mage'])
plt.xticks(np.arange(0, 5500, step = 1000))

# Birthweight vs Mother's Education
plt.subplot(2,2,2)
sns.scatterplot(x = birthweight['bwght'], y = birthweight['meduc'])
plt.xticks(np.arange(0, 5500, step = 1000))

# Birthweight vs. Father's Age
plt.subplot(2,2,3)
sns.scatterplot(x = birthweight['bwght'], y = birthweight['fage'])
plt.xticks(np.arange(0, 5500, step = 1000))

# Birthweight vs Father's Education
plt.subplot(2,2,4)
sns.scatterplot(x = birthweight['bwght'], y = birthweight['feduc'])
plt.xticks(np.arange(0, 5500, step = 1000))

plt.show()


# No discernable correlations between birthweight and education
# We can see a slight correlation between birthweight and age, as age gets higher 
# birthweight gets lower


###############################################################################


# Birthweight vs. Number of Prenatal Visits
plt.subplot(2,2,1)
sns.scatterplot(x = birthweight['bwght'], y = birthweight['npvis'])
plt.xticks(np.arange(0, 5500, step = 1000))

# Birthweight vs. Avg. Cigarettes per Day
plt.subplot(2,2,2)
sns.scatterplot(x = birthweight['bwght'], y = birthweight['cigs'])
plt.xticks(np.arange(0, 5500, step = 1000))

# Birthweight vs. Avg. Drinks per Week
plt.subplot(2,2,3)
sns.scatterplot(x = birthweight['bwght'], y = birthweight['drink'])
plt.xticks(np.arange(0, 5500, step = 1000))

plt.show()


# Here we can see a clear correlation between birthweight and cigs and drink
# The more you drink and smoke the lower the birthweight will be 


#####################################
# Dep Var. vs. Ind. Categorical Var #
#####################################

""" 
Assumed Categorical Independent Variables:
Male
Mwhte
Mblck
Moth
Fwhte
Fblck
Foth
Monpre
Omaps
Fmaps
"""

# Birthweight vs. If Male
birthweight.boxplot(column = ['bwght'],
                by = ['male'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)

plt.title("Birthweight by If Baby Male")
plt.suptitle("")
plt.show()

# Birthweight vs. If Mother White
birthweight.boxplot(column = ['bwght'],
                by = ['mwhte'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)

plt.title("Birthweight by If Mother White")
plt.suptitle("")
plt.show()

# Birthweight vs. If Mother Black
birthweight.boxplot(column = ['bwght'],
                by = ['mblck'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)

plt.title("Birthweight by If Mother Black")
plt.suptitle("")
plt.show()

# Birthweight vs. If Mother Other
birthweight.boxplot(column = ['bwght'],
                by = ['moth'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)

plt.title("Birthweight by If Mother Other Ethnicity")
plt.suptitle("")
plt.show()


# Birthweight appears to be split almost evenly amongst categories of variables


###############################################################################


# Birthweight vs. If Father White
birthweight.boxplot(column = ['bwght'],
                by = ['fwhte'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)

plt.title("Birthweight by If Father White")
plt.suptitle("")
plt.show()

# Birthweight vs. If Father Black
birthweight.boxplot(column = ['bwght'],
                by = ['fblck'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)

plt.title("Birthweight by If Father Black")
plt.suptitle("")
plt.show()

# Birthweight vs. If Father Other
birthweight.boxplot(column = ['bwght'],
                by = ['foth'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)

plt.title("Birthweight by If Father Other Ethnicity")
plt.suptitle("")
plt.show()

# Birthweight vs. Month Prenatal Care Began
birthweight.boxplot(column = ['bwght'],
                by = ['monpre'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)

plt.title("Birthweight by Month Prenatal Care Began")
plt.suptitle("")
plt.show()


# Birthweight appears to be split almost evenly amongst categories of first 3 variables
# Expected monpre & birthweight higly correlated but not so


###############################################################################


# Birthweight vs. One-Minute APGAR Score
birthweight.boxplot(column = ['bwght'],
                by = ['omaps'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)

plt.title("Birthweight by One-Minute APGAR Score")
plt.suptitle("")
plt.show()

# Birthweight vs. Five-Minute APGAR Score
birthweight.boxplot(column = ['bwght'],
                by = ['fmaps'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)

plt.title("Birthweight by Five-Minute APGAR Score")
plt.suptitle("")
plt.show()


# No observed correlation between omaps and birthweight
# Can observe correlation between fmaps & birthweight, higher fmaps = higer birthweight





###############################################################################
####################          TUNE & FLAG OUTLIERS         ####################
###############################################################################

# Get quantiles of each variable to flag outliers
birthweight_quantiles = birthweight.loc[:, :].quantile([0.05,
                                                0.10,
                                                0.25,
                                                0.50,
                                                0.75,
                                                0.90,
                                                0.95])

# Reason for choosing 0.10 & 0.90 quantiles due to birthweight
  # SGA (small for gestational age) babies weigh less than 10% of newborns
  # LGA (large for gestinational age) babies weigh more than 90% of newborns


bwght_low = 2562   # SGA weight
bwght_hi = 3950    # LGA weight
mage_low = 29
mage_hi = 53
fage_low = 29
fage_hi = 50
meduc_hi = 13      # denotes hihger education beyond hihgh school
meduc_low = 12     # denotes education up to high school only
feduc_hi = 13      # denotes higher education beyond hihgh school
feduc_low = 12     # denotes education up to high school only
npvis_low = 7
npvis_hi = 15



# features not included are categorical variables


##########################
# Creating Outlier Flags #
##########################


# Birthweight High
birthweight['hi_bwght'] = 0

for val in enumerate(birthweight.loc[ : , 'bwght']):
    
    if val[1] >= bwght_hi:
        birthweight.loc[val[0], 'hi_bwght'] = 1
        
        
# Birthweight Low
birthweight['low_bwght'] = 0

for val in enumerate(birthweight.loc[ : , 'bwght']):
    
    if val[1] <= bwght_low:
        birthweight.loc[val[0], 'low_bwght'] = 1 
        

# Mother's Age High
birthweight['hi_mage'] = 0

for val in enumerate(birthweight.loc[ : , 'mage']):
    
    if val[1] >= mage_hi:
        birthweight.loc[val[0], 'hi_mage'] = 1
        
        
# Mother's Age Low
birthweight['low_mage'] = 0

for val in enumerate(birthweight.loc[ : , 'mage']):
    
    if val[1] <= mage_low:
        birthweight.loc[val[0], 'low_mage'] = 1 
        
        
# Father's Age High
birthweight['hi_fage'] = 0

for val in enumerate(birthweight.loc[ : , 'fage']):
    
    if val[1] >= fage_hi:
        birthweight.loc[val[0], 'hi_fage'] = 1
        
        
# Father's Age Low
birthweight['low_fage'] = 0

for val in enumerate(birthweight.loc[ : , 'fage']):
    
    if val[1] <= fage_low:
        birthweight.loc[val[0], 'low_fage'] = 1 
        
        
# Mother's Education High
birthweight['hi_meduc'] = 0

for val in enumerate(birthweight.loc[ : , 'meduc']):
    
    if val[1] >= meduc_hi:
        birthweight.loc[val[0], 'hi_meduc'] = 1
        
        
# Mother's Education Low
birthweight['low_meduc'] = 0

for val in enumerate(birthweight.loc[ : , 'meduc']):
    
    if val[1] <= meduc_low:
        birthweight.loc[val[0], 'low_meduc'] = 1 
        
        
# Father's Education High
birthweight['hi_feduc'] = 0

for val in enumerate(birthweight.loc[ : , 'feduc']):
    
    if val[1] >= feduc_hi:
        birthweight.loc[val[0], 'hi_feduc'] = 1
        
        
# Father's Education Low
birthweight['low_feduc'] = 0

for val in enumerate(birthweight.loc[ : , 'feduc']):
    
    if val[1] <= feduc_low:
        birthweight.loc[val[0], 'low_feduc'] = 1 
        
        
# Number Prenatal Visits High
birthweight['hi_npvis'] = 0

for val in enumerate(birthweight.loc[ : , 'npvis']):
    
    if val[1] >= npvis_hi:
        birthweight.loc[val[0], 'hi_npvis'] = 1
        
        
# Number Prenatal Visits Low
birthweight['low_npvis'] = 0

for val in enumerate(birthweight.loc[ : , 'npvis']):
    
    if val[1] <= npvis_low:
        birthweight.loc[val[0], 'low_npvis'] = 1
        
        
        
###############################################################################
#################            FEATURE ENGINGEERING             #################
###############################################################################        

## Maybe combined age of both parents has a higher correlation than individually
# Combine Mother's Age & Father's Age into Parent's Age
        
birthweight['page'] = birthweight['mage'] + birthweight['fage']/2


## Maybe combined education of both parents has a higher correlation than individually        
# Combine Mother's Education & Father's Education into Parent's Education

birthweight['peduc'] = birthweight['meduc'] + birthweight['feduc']/2



## Maybe the average pre-natal visits per month, is more actionable variable
  # By dividing the number of prenatal visits by the month prenatal visits we 
  # can get avg pre-natal visits
birthweight['avgvis'] = (birthweight['npvis']/
                                 (10 - birthweight['monpre']))


##################################
# Flag Outliers of New Variables #
##################################


# Get quantiles of new variables
birthweight_quantiles = birthweight.loc[:, :].quantile([0.05,
                                                0.10,
                                                0.25,
                                                0.50,
                                                0.75,
                                                0.90,
                                                0.95])


page_low = 46
page_hi = 82
peduc_low = 24  #assume both parents only studied up to highschool
peduc_hi = 25
avgvis_low = 1
avgvis_hi = 3
   
    
    
### Flag High & Low Quantities of Variables


# Parent's Age High
birthweight['hi_page'] = 0

for val in enumerate(birthweight.loc[ : , 'page']):
    
    if val[1] >= page_hi:
        birthweight.loc[val[0], 'hi_page'] = 1
        
        
# Parent's Age Low
birthweight['low_page'] = 0

for val in enumerate(birthweight.loc[ : , 'page']):
    
    if val[1] <= page_low:
        birthweight.loc[val[0], 'low_page'] = 1
        
        
# Parent's Education High
birthweight['hi_peduc'] = 0

for val in enumerate(birthweight.loc[ : , 'peduc']):
    
    if val[1] >= peduc_hi:
        birthweight.loc[val[0], 'hi_peduc'] = 1
        
        
# Parent's Education Low
birthweight['low_peduc'] = 0

for val in enumerate(birthweight.loc[ : , 'peduc']):
    
    if val[1] <= peduc_low:
        birthweight.loc[val[0], 'low_peduc'] = 1



# Avg. Pre-natal visits per Month High
birthweight['hi_avgvis'] = 0

for val in enumerate(birthweight.loc[ : , 'avgvis']):
    
    if val[1] >= avgvis_hi:
        birthweight.loc[val[0], 'hi_avgvis'] = 1
        
        
# Avg. Pre-natal visits per Month Low
birthweight['low_avgvis'] = 0

for val in enumerate(birthweight.loc[ : , 'avgvis']):
    
    if val[1] <= peduc_low:
        birthweight.loc[val[0], 'low_avgvis'] = 1



###############################################################################
#################              ONE HOT ENCODING               #################
###############################################################################

## One hot enocding for monpre 
 # variable observations are numbers 0-10
 # if one hot encode as is, col names result as numbers and can't distinguish variables

 
## To solve this, make new column, let's say monpre_cat
 # the observations in monpre_cat depend on monpre value
 # if monpre = 0 then monpre_cat = monpre_0, and so on for all variables


## Create categories of birthweight and one hot encode for logistic regression
 # if baby underweight = 1, if normal = 2, if overweight = 3

## MONPRE
 # define function detailing inputs of new row monpre_cat

def set_monpre_cat(row):
    if row['monpre'] == 0:
        return 'monpre_0'
    elif row['monpre'] == 1:
        return 'monpre_1' 
    elif row['monpre'] == 2:
        return 'monpre_2'
    elif row['monpre'] == 3:
        return 'monpre_3'
    elif row['monpre'] == 4:
        return 'monpre_4'
    elif row['monpre'] == 5:
        return 'monpre_5'
    elif row['monpre'] == 6:
        return 'monpre_6'
    elif row['monpre'] == 7:
        return 'monpre_7'
    elif row['monpre'] == 8:
        return 'monpre_8'
    else:
        return 'monpre_9'
    
    
  # Take defined function and apply it to birthweight df  
birthweight = birthweight.assign(monpre_cat = birthweight.apply(set_monpre_cat,
                                                                axis=1))    
    

    
## BIRTHWEIGHT
 # define function detailing inputs of new row bwght_cat

def set_bwght_cat(row):
    if row['bwght'] <= bwght_low:
        return 1
    elif row['bwght'] >= bwght_hi:
        return 3 
    else:
        return 2    

  # Take defined function and apply it to birthweight df  
birthweight = birthweight.assign(bwght_cat = birthweight.apply(set_bwght_cat,
                                                                axis=1))


## Now that we can distinguish categories between variables, we can one-hot encode them


# One-Hot Encoding Qualitative Variables
monpre_dummies = pd.get_dummies(list(birthweight['monpre_cat']), drop_first = True)

# Concatenating One-Hot Encoded Values with the Larger DataFrame
birthweight_2 = pd.concat(
        [birthweight.loc[:,:],
         monpre_dummies],
         axis = 1) # must set to 1 or will think you're working with col's not rows



###############################################################################
#################              LINEAR REGRESSION              #################
###############################################################################

## Seperating feature variables from target variable
bwght_feat = birthweight_2.drop(['bwght',       #drop target variable
                                 'bwght_cat',   #drop categorical target variable
                                 'monpre_cat',  #drop becuase only created to dummy code variable
                                 'omaps',       #drop omaps, happens after baby born, no predictive power
                                 'fmaps',       #drop fmaps, happens after baby born, no predictive power
                                 'hi_bwght',    #drop outlier flag for high birthweight
                                 'low_bwght',   #drop outlier flag for low birthweight
                                 'mage',        #drop mother age since engineered parents age
                                 'fage',        #drop father age since engineered parents age
                                 'meduc',       #drop mother education since engineered parents education
                                 'feduc',       #drop father education since engineered parents education
                                 'quantile',    #drop birtweight quantile flags 
                                 'hi_feduc',    #drop high father education flag
                                 'low_feduc',   #drop low father education flag
                                 'hi_meduc',    #drop high mother education flag
                                 'low_meduc',   #drop low mother education flag
                                 'm_meduc',     #drop meduc missing value flag
                                 'm_npvis',     #drop npvis missing value flag
                                 'm_feduc'],    #drop feduc missing value flag 
                                  axis = 1)

bwght_target = birthweight_2.loc[:, 'bwght']

## Split into test and training data

X_train, X_test, y_train, y_test = train_test_split(
            bwght_feat,
            bwght_target,
            test_size = 0.10,
            random_state = 508)


#####################
#  OLS Regression   #
#####################

## Run on all variables without train/test split
lm_full = smf.ols('bwght_target ~ bwght_feat', data = birthweight_2) 

# Fitting Results
results_lm_full = lm_full.fit()


# Printing Summary Statistics
print(results_lm_full.summary())



print(f"""
Summary Statistics:
R-Squared:          {results_lm_full.rsquared.round(3)}
Adjusted R-Squared: {results_lm_full.rsquared_adj.round(3)}
""")

    
    
# Checking predicted birthwegiht v. actual birthweight
pred_lm_full = results_lm_full.predict()
y_hat = pd.DataFrame(pred_lm_full).round(2)
resids_lm_full  = results_lm_full.resid.round(2)



# Plotting residuals
residual_analysis = pd.concat(
        [birthweight_2.loc[:,'bwght'],
         y_hat,
         results_lm_full.resid.round(2)],
         axis = 1)



sns.residplot(x = pred_lm_full,
              y = birthweight_2.loc[:,'bwght'])


plt.show()



 #######################
#  OLS Regression 2   #
#######################


## Run regression model on training data
    # Need to merge X_train and y_train to use in stats model
birthweight_train = pd.concat([X_train, y_train], axis = 1)


## Seperating feature variables from target variable
bwght_feat_train   = birthweight_train.drop(['bwght'], axis = 1)

bwght_target_train = birthweight_train.loc[:, 'bwght']



# Create model
lm_full_train = smf.ols(formula = """bwght_target_train ~ bwght_feat_train""",
                         data = birthweight_train)



# Fit the model based on the data
results_lm_split = lm_full_train.fit()



# Analyze the summary output
print(results_lm_split.summary())


print(f"""
Summary Statistics:
R-Squared:          {results_lm_split.rsquared.round(3)}
Adjusted R-Squared: {results_lm_split.rsquared_adj.round(3)}
""") 



  
#####################
#  KNN Regression   #
#####################


## How many neighbors to use? Run loop to figure optimal neighbors


# Creating two lists, one for training set accuracy and the other for test
# set accuracy
training_accuracy = []
test_accuracy = []


# Building a visualization to check to see  1 to 50
neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


# Plotting the visualization
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# Visualization can be dificult to define max, easier defined numerically
print(test_accuracy.index(max(test_accuracy)))


# Optimal result at n_neighbors = 5

## Now we can build our KNN Regression model
# Creating a regressor object
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 5)


# Teaching (fitting) the algorithm based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)



# Predicting on the X_data that the model has never seen before
y_knn_pred = knn_reg.predict(X_test)


# Printing out prediction values for each test observation
print(f"""
Test set predictions:
{y_knn_pred}
""")


# Calling the score method, which compares the predicted values to the actual
# values
y_score_knn = knn_reg.score(X_test, y_test)


# The score is directly comparable to R-Square
print(y_score_knn)





######################################
#  Does OLS Predict Better Than KNN  #
######################################

# Prepping the Model
lr = LinearRegression(fit_intercept = False)


# Fitting the model
lr_fit = lr.fit(X_train, y_train)


# Predictions
lr_pred = lr_fit.predict(X_test)


print(f"""
Test set predictions:
{y_knn_pred.round(2)}
""")
    
    
    
# Scoring the model
y_score_ols_sklearn = lr_fit.score(X_test, y_test)


# The score is directly comparable to R-Square
print(y_score_ols_sklearn)



# Let's compare the testing score to the training score.

print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))


# Printing model results
model_predictions_df print(f"""
Optimal model KNN score: {y_score_knn.round(3)}
Optimal model OLS-SKL score: {y_score_ols_sklearn.round(3)}
Optimal model OLS-STAT score: {results_lm_split.rsquared_adj.round(3)}
""")




###############################################################################
#################            LINEAR DECISION TREE             #################
###############################################################################



################################################
# Adjust To At Least 10% Observations per Node #
################################################


#Prep the Model
tree_leaf_20 = DecisionTreeRegressor(criterion = 'friedman_mse',
                                     min_samples_leaf = 20, # details number of obsv's in each node
                                     max_depth = 3, random_state = 508)


# Fit the Model
tree_leaf_20_fit = tree_leaf_20.fit(X_train, y_train)


#Print model scores
print('Training Score', tree_leaf_20.score(X_train, y_train).round(4))
print('Testing Score:', tree_leaf_20.score(X_test, y_test).round(4))



## Visualize Tree

dot_data = StringIO()
export_graphviz(decision_tree = tree_leaf_20_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = bwght_feat.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png(),
      height = 500,
      width = 800)



#########################
#   Feature Importance  #
#########################


# Defining function to visualize feature importance
def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_20_Feature_Importance.png')

# Plot 20 per leaf Tree
plot_feature_importances(tree_leaf_20,
                         train = X_train,
                         export = True)

## A lot of emphasis is placed on the number of drinks per week
  # What if we take out drinks per week to see what other variables are important
  
## Seperating feature variables from target variable
bwght_feat_2 = birthweight_2.drop(['bwght',       #drop target variable
                                 'bwght_cat',   #drop target variable log regression  
                                 'monpre_cat',  #drop becuase only created to dummy code variable
                                 'omaps',       #drop omaps, happens after baby born, no predictive power
                                 'fmaps',       #drop fmaps, happens after baby born, no predictive power
                                 'hi_bwght',    #drop outlier flag for high birthweight
                                 'low_bwght',   #drop outlier flag for low birthweight
                                 'mage',        #drop mother age since engineered parents age
                                 'fage',        #drop father age since engineered parents age
                                 'meduc',       #drop mother education since engineered parents education
                                 'feduc',       #drop father education since engineered parents education
                                 'quantile',    #drop birtweight quantile flags 
                                 'hi_feduc',    #drop high father education flag
                                 'low_feduc',   #drop low father education flag
                                 'hi_meduc',    #drop high mother education flag
                                 'low_meduc',   #drop low mother education flag
                                 'm_meduc',     #drop meduc missing value flag
                                 'm_npvis',     #drop npvis missing value flag
                                 'm_feduc',     #drop feduc missing value flag 
                                 'drink',       #drop cigs to see importance of other variables
                                 'cigs'],       #drop drink to see importance of other variables
                                 axis = 1)


## Split into test and training data

X_train, X_test, y_train, y_test = train_test_split(
            bwght_feat_2,
            bwght_target,
            test_size = 0.10,
            random_state = 508)


################################################
# Adjust To At Least 10% Observations per Node #
################################################


#Prep the Model
tree_leaf_20_v2 = DecisionTreeRegressor(criterion = 'friedman_mse',
                                     min_samples_leaf = 20, # details number of obsv's in each node
                                     max_depth = 3, random_state = 508)


# Fit the Model
tree_leaf_20_fit_v2 = tree_leaf_20_v2.fit(X_train, y_train)


#Print model scores
print('Training Score', tree_leaf_20_v2.score(X_train, y_train).round(4))
print('Testing Score:', tree_leaf_20_v2.score(X_test, y_test).round(4))



## Visualize Tree

dot_data = StringIO()
export_graphviz(decision_tree = tree_leaf_20_fit_v2,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = bwght_feat_2.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png(),
      height = 500,
      width = 800)



#########################
#   Feature Importance  #
#########################


# Defining function to visualize feature importance
def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_20_Feature_Importance.png')

# Plot 20 per leaf Tree
plot_feature_importances(tree_leaf_20_v2,
                         train = X_train,
                         export = True)


###############################################################################
#################             LOGISTIC REGRESSION             #################
###############################################################################



## Seperating feature variables from target variable
bwght_feat = birthweight_2.drop(['bwght',       #drop target variable
                                 'bwght_cat',   #drop categorical target variable
                                 'monpre_cat',  #drop becuase only created to dummy code variable
                                 'omaps',       #drop omaps, happens after baby born, no predictive power
                                 'fmaps',       #drop fmaps, happens after baby born, no predictive power
                                 'hi_bwght',    #drop outlier flag for high birthweight
                                 'low_bwght',   #drop outlier flag for low birthweight
                                 'mage',        #drop mother age since engineered parents age
                                 'fage',        #drop father age since engineered parents age
                                 'meduc',       #drop mother education since engineered parents education
                                 'feduc',       #drop father education since engineered parents education
                                 'quantile',    #drop birtweight quantile flags 
                                 'hi_feduc',    #drop high father education flag
                                 'low_feduc',   #drop low father education flag
                                 'hi_meduc',    #drop high mother education flag
                                 'low_meduc',   #drop low mother education flag
                                 'm_meduc',     #drop meduc missing value flag
                                 'm_npvis',     #drop npvis missing value flag
                                 'm_feduc'],    #drop feduc missing value flag 
                                  axis = 1)

bwght_target_2 = birthweight_2.loc[:, 'bwght_cat']

## Split into test and training data

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
            bwght_feat,
            bwght_target_2,
            test_size = 0.10,
            random_state = 508)




##########################
#  Logistic Regression   #
##########################


# Prepping the Model
mul_logreg = LogisticRegression(multi_class = 'multinomial', 
                                solver = 'newton-cg')


# Fitting the Model
mul_logreg_fit = mul_logreg.fit(X_train_log, y_train_log)


# Predictions
mul_logreg_pred = mul_logreg_fit.predict(X_test_log)


# Scoring the model
y_score_mul_logreg = mul_logreg_fit.score(X_test_log, y_test_log)


# The score is directly comparable to R-Square
print(y_score_mul_logreg)



# Let's compare the testing score to the training score.

print('Training Score', mul_logreg.score(X_train_log, y_train_log).round(4))
print('Testing Score:', mul_logreg.score(X_test_log, y_test_log).round(4))




###############################################################################
#################           LOGISTIC DECISION TREE            #################
###############################################################################



################################################
# Adjust To At Least 10% Observations per Node #
################################################


# Prep the Model
tree_leaf_20 = DecisionTreeClassifier(min_samples_leaf = 20, # details number of obsv's in each node
                                     max_depth = 3, random_state = 508)


# Fit the model
tree_leaf_20_fit = tree_leaf_20.fit(X_train_log, y_train_log)



# Model Score
print('Training Score', tree_leaf_20.score(X_train_log, y_train_log).round(4))
print('Testing Score:', tree_leaf_20.score(X_test_log, y_test_log).round(4))



#Visualize Decision Tree
dot_data = StringIO()


export_graphviz(decision_tree = tree_leaf_20_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = bwght_feat.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png(),
      height = 500,
      width = 800)
graph.write_png('Decision Tree No.png')




#########################
#   Feature Importance  #
#########################


# Defining function to visualize feature importance
def plot_feature_importances(model, train = X_train_log, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train_log.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_20_Feature_Importance.png')

plot_feature_importances(tree_leaf_20,
                         train = X_train_log,
                         export = True)



## A lot of emphasis is placed on the number of drinks per week
  # What if we take out drinks per week to see what other variables important
  
## Seperating feature variables from target variable
bwght_feat_2 = birthweight_2.drop(['bwght',       #drop target variable
                                 'monpre_cat',  #drop becuase only created to dummy code variable
                                 'omaps',       #drop omaps, happens after baby born, no predictive power
                                 'fmaps',       #drop fmaps, happens after baby born, no predictive power
                                 'hi_bwght',    #drop outlier flag for high birthweight
                                 'low_bwght',   #drop outlier flag for low birthweight
                                 'mage',        #drop mother age since engineered parents age
                                 'fage',        #drop father age since engineered parents age
                                 'meduc',       #drop mother education since engineered parents education
                                 'feduc',       #drop father education since engineered parents education
                                 'quantile',    #drop birtweight quantile flags 
                                 'hi_feduc',    #drop high father education flag
                                 'low_feduc',   #drop low father education flag
                                 'hi_meduc',    #drop high mother education flag
                                 'low_meduc',   #drop low mother education flag
                                 'm_meduc',     #drop meduc missing value flag
                                 'm_npvis',     #drop npvis missing value flag
                                 'm_feduc',     #drop feduc missing value flag 
                                 'drink',       #drop cigs to see importance of other variables
                                 'cigs',        #drop drink to see importance of other variables
                                 'bwght_cat'],  #drop target variable categories      
                                 axis = 1)


## Split into test and training data

X_train_log2, X_test_log2, y_train_log2, y_test_log2 = train_test_split(
            bwght_feat_2,
            bwght_target_2,
            test_size = 0.10,
            random_state = 508)



################################################
# Adjust To At Least 10% Observations per Node #
################################################


#Prep the Model
tree_leaf_20_v2 = DecisionTreeClassifier(min_samples_leaf = 20, # details number of obsv's in each node
                                     max_depth = 3, random_state = 508)


# Fit the Model
tree_leaf_20_fit_v2 = tree_leaf_20_v2.fit(X_train_log2, y_train_log2)


#Print model scores
print('Training Score', tree_leaf_20_v2.score(X_train_log2, y_train_log2).round(4))
print('Testing Score:', tree_leaf_20_v2.score(X_test_log2, y_test_log2).round(4))



## Visualize Tree

dot_data = StringIO()
export_graphviz(decision_tree = tree_leaf_20_fit_v2,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = bwght_feat_2.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png(),
      height = 500,
      width = 800)
graph.write_png('Decision Tree No Vices.png')



#########################
#   Feature Importance  #
#########################


# Defining function to visualize feature importance
def plot_feature_importances(model, train = X_train_log2, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train_log2.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_20_Feature_Importance.png')

plot_feature_importances(tree_leaf_20_v2,
                         train = X_train_log2,
                         export = True)




#####################
#  KNN Classifier   #
#####################


## Seperating feature variables from target variable
bwght_feat = birthweight_2.drop(['bwght',       #drop target variable
                                 'bwght_cat',   #drop categorical target variable
                                 'monpre_cat',  #drop becuase only created to dummy code variable
                                 'omaps',       #drop omaps, happens after baby born, no predictive power
                                 'fmaps',       #drop fmaps, happens after baby born, no predictive power
                                 'hi_bwght',    #drop outlier flag for high birthweight
                                 'low_bwght',   #drop outlier flag for low birthweight
                                 'mage',        #drop mother age since engineered parents age
                                 'fage',        #drop father age since engineered parents age
                                 'meduc',       #drop mother education since engineered parents education
                                 'feduc',       #drop father education since engineered parents education
                                 'quantile',    #drop birtweight quantile flags 
                                 'hi_feduc',    #drop high father education flag
                                 'low_feduc',   #drop low father education flag
                                 'hi_meduc',    #drop high mother education flag
                                 'low_meduc',   #drop low mother education flag
                                 'm_meduc',     #drop meduc missing value flag
                                 'm_npvis',     #drop npvis missing value flag
                                 'm_feduc'],    #drop feduc missing value flag 
                                  axis = 1)

bwght_target_2 = birthweight_2.loc[:, 'bwght_cat']

## Split into test and training data

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
            bwght_feat,
            bwght_target_2,
            test_size = 0.10,
            random_state = 508)

## How many neighbors to use? Run loop to figure optimal neighbors


# Creating two lists, one for training set accuracy and the other for test
# set accuracy
training_accuracy = []
test_accuracy = []


# Building a visualization to check to see  1 to 50
neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train_log, y_train_log)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train_log, y_train_log))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test_log, y_test_log))


# What is optimal neighbors?
print(test_accuracy.index(max(test_accuracy)))



# Plotting the visualization
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()



# Optimal result at n_neighbors = 3

## Now we can build our KNN Regression model
# Creating a regressor object
knn_class = KNeighborsClassifier(algorithm = 'auto',
                              n_neighbors = 3)


# Teaching (fitting) the algorithm based on the training data
knn_class_fit = knn_class.fit(X_train_log, y_train_log)



# Predicting on the X_data that the model has never seen before
y_knn_class_pred = knn_class.predict(X_test_log)


# Printing out prediction values for each test observation
print(f"""
Test set predictions:
{y_knn_class_pred}
""")


# Calling the score method, which compares the predicted values to the actual
# values
y_score_knn_class = knn_class.score(X_test_log, y_test_log)


# The score is directly comparable to R-Square
print(y_score_knn_class)



# Creating confusion matrix
conmat_knn = confusion_matrix(y_test, y_knn_class_pred)




###############################################################################
##########                 STORING MODEL PREDICTIONS                 ########## 
###############################################################################



# We can store our predictions as a dictionary.
model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'OLS_Predicted': lr_pred})



model_predictions_df.to_excel("Birthweight_Model_Predictions_Team7.xlsx")






















