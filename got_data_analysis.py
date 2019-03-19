# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:53:06 2019

@author: masal
"""

###############################################################################
#########################       GAME OF THRONES       #########################      
#########################      SURVIVAL STRATEGY      #########################
###############################################################################





####################
# Import Libraries #
####################



import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split # train/test split
import statsmodels.formula.api as smf # regression modeling
import sklearn.metrics # more metrics for model performance evaluation
from sklearn.model_selection import cross_val_score # k-folds cross validation
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier # KNN for Regression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score



######################################
# Load File into Working Environment #
######################################


file = "C:/Users/masal/Desktop/ML_Class/Ind_Project/GOT_character_predictions.xlsx"
got_df = pd.read_excel(file)




###################################
# Fundamental Dataset Exploration #
###################################

# Column names
got_df.columns


# Displaying the first rows of the DataFrame
print(got_df.head())


# Dimensions of the DataFrame
got_df.shape


# Information about each variable
got_df.info()


# Descriptive statistics
got_df.describe().round(2)



###############################################################################
####################             MISSING VALUES            ####################
###############################################################################


# What variables have missing values?
print(got_df.isnull().sum())


## Title, DOB, mother, father, heir, spouse, isAliveMother, isAliveFather,
 # isAliveHeir, isAliveSpouse, and age variables can't be imputed by regular means.
 
 
## Imputing by mean/median or using ML algorithm wouldn't make sense. 
 #  There too many missing values to be able to impute them without bias.


## Should I just drop them since at least 75% of values are missing??
 # Maybe DOB or age could be important in predicting whether a character lives
 #   or dies, but there's no efficient way to fill in the values without bias.
 # I can see how who your mother, father, or heir is and if they are alive
 #   can help predicting life or death, but again there is no efficient way
 #   to impute these values. 
 # I would think that the importance of who your mother or father is would be 
 #   reflected in the variable isNoble, and it has no missing values. 


## For these reasons I will drop them in this instance. 


got_df_drop = got_df.drop(['title', 'dateOfBirth', 'mother', 'father',
                           'heir', 'spouse', 'isAliveMother', 'isAliveFather',
                           'isAliveHeir', 'isAliveSpouse', 'age'], axis = 1)


# What variables have missing values now?
print(got_df_drop.isnull().sum())


## Culture and House are only remaining variables with missing values


###############################################################################


############################
#  Culture Missing Values  #
############################


# Maybe we can predict what the culture will be?


# How many unique cultures do we have?

got_df_drop['culture'].nunique() # we have 64 unique cultures 


# After taking closer look at culture, there can be many categories of same culture
# Westermen, Westerman, westermen, Westerlands all are the same as Westeros
# Must replace many categories of same culture


got_df_drop['culture'] = got_df_drop['culture'].replace(['Andals'], 'Andal')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Asshai\'i'], 'Asshai')  


got_df_drop['culture'] = got_df_drop['culture'].replace(['Astapori'], 'Astapor')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Braavosi'], 'Braavos')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Dornish', 'Dornishmen'], 
                                                         'Dorne')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Free folk', 'free folk'],
                                                         'Free Folk')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Ghiscaricari'], 
                                                         'Ghiscari')


got_df_drop['culture'] = got_df_drop['culture'].replace(['ironborn', 'Ironmen'], 
                                                         'Ironborn')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Lhazrene', 'Lhazarene', 
                                                         'Lhazreen'],
                                                         'Lhazareen')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Lyseni'], 'Lysene')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Meereenese'], 'Meereen')


got_df_drop['culture'] = got_df_drop['culture'].replace(['northmen', 
                                                         'Northern mountain clans'],
                                                         'Northmen')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Norvoshi'], 'Norvos')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Qartheen'], 'Qarth')


got_df_drop['culture'] = got_df_drop['culture'].replace(['The Reach', 'Reachmen'], 
                                                         'Reach')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Riverlands'], 
                                                         'Rivermen')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Stormlander'], 
                                                         'Stormlands')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Summer Islander', 
                                                         'Summer Islands'], 
                                                         'Summer Isles')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Vale', 
                                                         'Vale mountain clans'], 
                                                         'Valemen')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Westerman', 'Westermen',
                                                         'Westerlands', 'westermen'],
                                                         'Westeros')


got_df_drop['culture'] = got_df_drop['culture'].replace(['Wildlings'], 'Wildling')


# How many unique cultures do we have now?

got_df_drop['culture'].nunique() # we have 33 unique cultures 



# For logistic classifier we need to turn culture into numeric categories.

# Make a colum with numbers representing each culture 
# Sorted alphabetically and starts at 1

got_df_drop['cul_code'] = pd.factorize(got_df_drop['culture'], sort=True)[0] + 1


# What culture belongs to what code?

cultures_codes = got_df_drop.filter(['cul_code', 'culture'])


###############################################################################


## Now that we have made numeric categories for culture,
## we can continue with predicting the missing values.

# In order to predict culture, we have to split the data

# One split contains missing values (X_test_cul) other does not (X_train_cul)


# Define test set for features

X_test_cul = got_df_drop.loc[got_df_drop['cul_code'] == 0]


# Drop columns of variables that were engineered or not needed
 
X_test_cul = X_test_cul.drop(['culture', 'house', 'cul_code', 'name'], axis = 1)


# Define training set for features

X_train_cul = got_df_drop.loc[got_df_drop['cul_code'] != 0]


# Define target variable training set

y_train_cul = X_train_cul.loc[:, 'cul_code']


# Drop columns not needed or were engineered

X_train_cul = X_train_cul.drop(['culture', 'house', 'cul_code', 'name'], axis = 1)


###############################################################################


##########################
#  Logistic Regression   #
##########################


# Prepping the Model
logreg_cul = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')


# Fitting the Model
logreg_cul_fit = logreg_cul.fit(X_train_cul, y_train_cul)


# Predictions based on the model fit to the training data
y_test_logreg_cul = logreg_cul_fit.predict(X_test_cul)


# Let's compare the testing score to the training score.

print('Training Score', logreg_cul.score(X_train_cul, y_train_cul).round(4))
print('Testing Score:', logreg_cul.score(X_test_cul, y_test_logreg_cul).round(4))


###############################################################################


#####################
#  KNN Classifier   #
#####################


# Prepping the Model

knn_cul = KNeighborsClassifier(algorithm = 'auto')


# Fitting the Model

knn_cul_fit = knn_cul.fit(X_train_cul, y_train_cul)


# Predictions based on the model fit to the training data

y_test_knn_cul = knn_cul_fit.predict(X_test_cul)


# Let's compare the testing score to the training score.

print('Training Score', knn_cul.score(X_train_cul, y_train_cul).round(4))
print('Testing Score:', knn_cul.score(X_test_cul, y_test_knn_cul).round(4))


###############################################################################


#################
# Decision Tree #
#################


# Prep the Model
tree_cul = DecisionTreeClassifier(min_samples_leaf = 196, # 10% of observations
                                  random_state = 508)


# Fit the model
tree_cul_fit = tree_cul.fit(X_train_cul, y_train_cul)


# Predictions based on the model fit to the training data

y_test_tree_cul = tree_cul_fit.predict(X_test_cul)



# Model Score
print('Training Score', tree_cul.score(X_train_cul, y_train_cul).round(4))
print('Testing Score:', tree_cul.score(X_test_cul, y_test_tree_cul).round(4))


###############################################################################

# Which predictive model worked best?

# Model Score
print('Logistic Reg Score   :', logreg_cul.score(X_train_cul, y_train_cul).round(4))
print('KNN Classifier Score :', knn_cul.score(X_train_cul, y_train_cul).round(4))
print('Decision Tree Score  :', tree_cul.score(X_train_cul, y_train_cul).round(4))


# KNN Classifier achieved the best prediction rate at about 40%


###############################################################################


## Now I need to merge and concatenate the train and testing sets with the
 #   predictions made by the logistic regression model
 
# Concatenate the training data sets 
 
train_cul = pd.concat([X_train_cul, y_train_cul], axis = 1)


# Concatenate test set with predictions from KNN Classifier

# Must convert predictions to data frame to be able to merge

y_test_knn_cul = pd.DataFrame(y_test_knn_cul)


# Make X_test into data frame and reset index (will help when concatenating sets)

X_test_cul_2 = pd.DataFrame.reset_index(X_test_cul)


# Drop previous index column

X_test_cul_2 = X_test_cul_2.drop(['index'], axis = 1)


# Concatenate both test sets

test_cul = pd.concat([X_test_cul_2, y_test_knn_cul], axis = 1)


###############################################################################


## We now need to stack train_cul and test_cul to have a clean culture column

# Name test_cul culture column 'cul_code' as it is in train_cul

test_cul['cul_code'] = test_cul[0]


# Drop 0 column from test_cul

test_cul = test_cul.drop([0], axis =1)


# Make test and train into Data Frames and rest index 

test_cul = pd.DataFrame.reset_index(test_cul)

test_cul = test_cul.drop(['index'], axis = 1)

train_cul = pd.DataFrame.reset_index(train_cul)

train_cul = train_cul.drop(['index'], axis = 1)


train_cul.shape

# Now we can stack

cul_clean_df = train_cul.append(test_cul)


# Let's call the new cul_code column cul_clean

cul_clean_df['cul_clean'] = cul_clean_df['cul_code']


# Drop previous cul_code column

cul_clean_df = cul_clean_df.drop(['cul_code'], axis = 1)



###############################################################################

## Now that we have a "clean" cul_code column with no missing values in test_cul,
 #   we can add that column back to the got data frame.
 
## To add it back I'm going to pull the culture_clean column and an identifier
 #  column to merge the predictions into the correct characters.

# Name the column with culture values 'culture_clean'
 

## 'S.No' is a column both data sets have in common

pred_cul = pd.concat([cul_clean_df['S.No'], cul_clean_df['cul_clean']], axis = 1)


## We can now merge our two data sets together on the 'S.No' column

got_df_drop_2 = pd.merge(got_df_drop, pred_cul, on = 'S.No')


# Drop previous cul_code and culture column

got_df_drop_2 = got_df_drop_2.drop(['cul_code', 'culture'], axis = 1)


# Variables with missing values

print(got_df_drop_2.isnull().sum())
 

# We see that now only house has missing values


 
###############################################################################


##########################
#  House Missing Values  #
##########################


# Maybe we can predict what the house will be?


# How many unique houses do we have?

got_df_drop_2['house'].nunique() # we have 347 unique houses


# After taking closer look at house, there can be many categories of same house
# Same issue as seen with culture
# Must replace many categories of same house


got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                           'brotherhood without banners',
                                           'Brotherhood without Banners',
                                           'Brotherhood without banners'], 
                                           'Brotherhood Without Banners')

got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                           'House Baratheon of Dragonstone',
                                           'House Baratheon of King\'s Landing'],
                                           'House Baratheon')

got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                           'House Bolton of the Dreadfort'],
                                           'House Bolton')

got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                           'House Brune of Brownhollow',
                                           'House Brune of the Dyre Den'],
                                           'House Brune')

got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                           'House Dayne of High Hermitage'],
                                           'House Dayne')

got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                           'House Farwynd of the Lonely Light'],
                                           'House Farwynd')

got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                           'House Flint of Widow\'s Watch'],
                                           'House Flint')

got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                           'House Fossoway of Cider Hall',
                                           'House Fossoway of New Barrel'],
                                           'House Fossoway')

got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                           'House Frey of Riverrun'],
                                           'House Frey')

got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                           'House Goodbrother of Shatterstone'],
                                           'House Goodbrother')

got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                           'House Harlaw of Grey Garden',
                                           'House Harlaw of Harlaw Hall',
                                           'House Harlaw of Harridan Hill',
                                           'House Harlaw of the Tower of Glimmering'],
                                           'House Harlaw')

got_df_drop_2['house'] = got_df_drop_2['house'].replace(['House Kenning of Harlaw',
                                           'House Kenning of Kayce'],
                                           'House Kenning')

got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                           'House Lannister of Lannisport',
                                           'House Lannister of Casterly Rock'],
                                           'House Lannister')

got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                            'House Royce of the Gates of the Moon'],
                                            'House Royce')

got_df_drop_2['house'] = got_df_drop_2['house'].replace([
                                           'House Tyrell of Brightwater Keep'],
                                           'House Tyrell')

got_df_drop_2['house'] = got_df_drop_2['house'].replace(['House Vance of Atranta',
                                           'House Vance of Wayfarer\'s Rest'],
                                           'House Vance')

# How many unique houses do we have now?

got_df_drop_2['house'].nunique() # we have 322 unique houses 


# For logistic classifier we need to turn house into numeric categories.

# Make a colum with numbers representing each culture 
# Sorted alphabetically and starts at 1

got_df_drop_2['house_code'] = pd.factorize(got_df_drop_2['house'], sort=True)[0] + 1

# What house belongs to what code?

houses_codes = got_df_drop_2.filter(['house_code', 'house'])


###############################################################################


## Now that we have made numeric categories for house,
## we can continue with predicting the missing values.

# In order to predict culture, we have to split the data

# One split contains missing values (X_test_cul) other does not (X_train_cul)


# Define test set for features

X_test_house = got_df_drop_2.loc[got_df_drop_2['house_code'] == 0]


# Drop columns of variables that were engineered or not needed
 
X_test_house = X_test_house.drop(['house', 'name', 'house_code' ], axis = 1)


# Define training set for features

X_train_house = got_df_drop_2.loc[got_df_drop_2['house_code'] != 0]


# Define target variable training set

y_train_house = X_train_house.loc[:, 'house_code']


# Drop columns not needed or were engineered

X_train_house = X_train_house.drop(['house', 'name', 'house_code' ], axis = 1)


###############################################################################


##########################
#  Logistic Regression   #
##########################


# Prepping the Model
logreg_house = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')


# Fitting the Model
logreg_house_fit = logreg_house.fit(X_train_house, y_train_house)


# Predictions based on the model fit to the training data
y_test_logreg_house = logreg_house_fit.predict(X_test_house)


# Let's compare the testing score to the training score.

print('Training Score', logreg_house.score(X_train_house, y_train_house).round(4))
print('Testing Score:', logreg_house.score(X_test_house, y_test_logreg_house).round(4))


###############################################################################


#####################
#  KNN Classifier   #
#####################


# Prepping the Model

knn_house = KNeighborsClassifier(algorithm = 'auto', n_neighbors = 6)


# Fitting the Model

knn_house_fit = knn_house.fit(X_train_house, y_train_house)


# Predictions based on the model fit to the training data

y_test_knn_house = knn_house_fit.predict(X_test_house)


# Let's compare the testing score to the training score.

print('Training Score', knn_house.score(X_train_house, y_train_house).round(4))
print('Testing Score:', knn_house.score(X_test_house, y_test_knn_house).round(4))


###############################################################################


#################
# Decision Tree #
#################


# Prep the Model
tree_house = DecisionTreeClassifier(min_samples_leaf = 196, # 10% of observations
                                  random_state = 508)


# Fit the model
tree_house_fit = tree_house.fit(X_train_house, y_train_house)


# Predictions based on the model fit to the training data

y_test_tree_house = tree_house_fit.predict(X_test_house)



# Model Score
print('Training Score', tree_house.score(X_train_house, y_train_house).round(4))
print('Testing Score:', tree_house.score(X_test_house, y_test_tree_house).round(4))


###############################################################################

# Which predictive model worked best?

# Model Score
print('Logistic Reg Score   :', logreg_house.score(X_train_house, y_train_house).round(4))
print('KNN Classifier Score :', knn_house.score(X_train_house, y_train_house).round(4))
print('Decision Tree Score  :', tree_house.score(X_train_house, y_train_house).round(4))


# KNN Classifier achieved the best prediction rate at about 25%


###############################################################################


## Now I need to merge and concatenate the train and testing sets with the
 #   predictions made by the logistic regression model
 
# Concatenate the training data sets 
 
train_house = pd.concat([X_train_house, y_train_house], axis = 1)


# Concatenate test set with predictions from KNN Classifier

# Must convert predictions to data frame to be able to merge

y_test_knn_house = pd.DataFrame(y_test_knn_house)


# Make X_test into data frame and reset index (will help when concatenating sets)

X_test_house_2 = pd.DataFrame.reset_index(X_test_house)


# Drop previous index column

X_test_house_2 = X_test_house_2.drop(['index'], axis = 1)


# Concatenate both test sets

test_house = pd.concat([X_test_house_2, y_test_knn_house], axis = 1)


###############################################################################


## We now need to stack train_cul and test_cul to have a clean culture column

# Name test_cul culture column 'cul_code' as it is in train_cul

test_house['house_code'] = test_house[0]


# Drop 0 column from test_cul

test_house = test_house.drop([0], axis =1)


# Make test and train into Data Frames and rest index 

test_house = pd.DataFrame.reset_index(test_house)

test_house = test_house.drop(['index'], axis = 1)

train_house = pd.DataFrame.reset_index(train_house)

train_house = train_house.drop(['index'], axis = 1)


train_house.shape

# Now we can stack

house_clean_df = train_house.append(test_house)


# Let's call the new cul_code column cul_clean

house_clean_df['house_clean'] = house_clean_df['house_code']


# Drop previous cul_code column

house_clean_df = house_clean_df.drop(['house_code'], axis = 1)

house_clean_df.shape

###############################################################################

## Now that we have a "clean" cul_code column with no missing values in test_cul,
 #   we can add that column back to the got data frame.
 
## To add it back I'm going to pull the culture_clean column and an identifier
 #  column to merge the predictions into the correct characters.

# Name the column with culture values 'culture_clean'
 

## 'S.No' is a column both data sets have in common

pred_house = pd.concat([house_clean_df['S.No'], house_clean_df['house_clean']], 
                       axis = 1)


## We can now merge our two data sets together on the 'S.No' column

got_clean = pd.merge(got_df_drop_2, pred_house, on = 'S.No')


# Drop previous cul_code and culture column

got_clean = got_clean.drop(['house_code', 'house'], axis = 1)


# Variables with missing values

print(got_clean.isnull().sum())

# We see that there are no more missing values


###############################################################################
####################       EXPLORATORY DATA ANALYSIS       ####################
###############################################################################


# Column names
got_clean.columns


# Displaying the first rows of the DataFrame
print(got_clean.head())


# Dimensions of the DataFrame
got_clean.shape


# Information about each variable
got_clean.info()


# Descriptive statistics
got_summary = got_clean.describe().round(2)

print(got_summary)


###############################################################################


#####################################
###      UNIVARIATE ANALYSIS      ###
#####################################


########################
# Continuous Variables #
########################


# Number Dead  Relations 

plt.subplot(2,1,1)
sns.distplot(got_clean['numDeadRelations'])
plt.xlabel('Number of Dead Relatives')

# low number of characters have high popularity, to be expected

# Popularity

plt.subplot(2,1,2)
sns.distplot(got_clean['popularity'])
plt.xlabel('Popularity')

plt.show()

# number of dead relatives has a zero inflated distribution

got_clean['numDeadRelations'].value_counts()

# but a few characters have much more "bad luck" than others


###############################################################################


#########################
# Categorical Variables #
#########################


## Target Variable - isAlive ##

got_clean['isAlive'].value_counts()

# Almost 75% of characters in data set live



## Deomographics ##


## If Character Male

got_clean['male'].value_counts()

# About 62% of characters in data set are male



## Is Married

got_clean['isMarried'].value_counts()

# Only 14% of characters in data set married 



## Is Character of Noble Decent

got_clean['isNoble'].value_counts()

# About 45% of characters of noble decent. 


###############################################################################


## Number of Character in Each Book ##


## Book 1: A Game of Thrones

got_clean['book1_A_Game_Of_Thrones'].value_counts()

# Total of 386 characters make appearance



## Book 2: A Clash of Kings

got_clean['book2_A_Clash_Of_Kings'].value_counts()

# Total of 729 characters make appearance



## Book 3: A Storm of Swords

got_clean['book3_A_Storm_Of_Swords'].value_counts()

# Total of 935 characters make appearance



## Book 4: A Feast for Crows

got_clean['book4_A_Feast_For_Crows'].value_counts()

# Total of 1152 characters make appearance, drastic decrease.




## Book 5: A Dance with Dragons

got_clean['book5_A_Dance_with_Dragons'].value_counts()

# Total of 769 characters appear
# Maybe a lot of characters died in book 4?

###############################################################################


## Number of Characters Loyal to each House ##

got_clean['house_clean'].value_counts()

# Top Five Houses: 

#       1) Night's Watch with 150 loyal characters       
#       2) House Frey with 114 loyal characters
#       3) House Stark with 92 loyal characters        
#       4) House Targaryen with 86 loyal characters
#       5) House Greyjoy with 76 loyal characters 



## Number of Characters Belonging to each Culture ##

got_clean['cul_clean'].value_counts()

# Top Five Cultures: 

#       1) Northmen with 471 characters      
#       2) Ironborn with 365 characters
#       3) Braavos with 190 characters        
#       4) Free Folk with 190 characters
#       5) Dorne with 179 characters 


###############################################################################


#####################################
###      BIVARIATE ANALYSIS       ###
#####################################


########################
# Continuous Variables #
########################


# Popularity vs. Is Alive #

got_clean.boxplot(column = ['popularity'],
                  by = ['isAlive'],
                  vert = False,
                  patch_artist = False,
                  meanline = True,
                  showmeans = True)
plt.xticks(np.arange(0, 1.10, step = 0.10))
plt.xlabel('Popularity')

# On average characters with higher popularities have a slightly greater
#   probability of dying, but no too  much.


###############################################################################


#########################
# Categorical Variables #
#########################


# Is Alive vs. If Male #

got_clean.groupby('male').isAlive.value_counts().plot.barh()

got_clean.groupby(['male', 'isAlive']).size()

# About 80% of females in dataset live
# About 70% of males in dataset live



# Is Alive vs. If in Book 1 #

got_clean.groupby('book1_A_Game_Of_Thrones').isAlive.value_counts().plot.barh()

got_clean.groupby(['book1_A_Game_Of_Thrones', 'isAlive']).size()

# About 55% of characters live
# Higher chance of living if not in book 1



# Is Alive vs. If in Book 2 #

got_clean.groupby('book2_A_Clash_Of_Kings').isAlive.value_counts().plot.barh()

got_clean.groupby(['book2_A_Clash_Of_Kings', 'isAlive']).size()

# About 75% of characters live
# Slightly higher chance of living if not in book 2



# Is Alive vs. If in Book 3 #

got_clean.groupby('book3_A_Storm_Of_Swords').isAlive.value_counts().plot.barh()

got_clean.groupby(['book3_A_Storm_Of_Swords', 'isAlive']).size()

# About 75% of characters live.
# Equal chances of living if not book 3 than if you are.



# Is Alive vs. If in Book 4 #

got_clean.groupby('book4_A_Feast_For_Crows').isAlive.value_counts().plot.barh()

got_clean.groupby(['book4_A_Feast_For_Crows', 'isAlive']).size()

# About 80% of characters live.
# Higher probability of living if character makes appearnce in book 4 



# Is Alive vs. If in Book 5 #

got_clean.groupby('book5_A_Dance_with_Dragons').isAlive.value_counts().plot.barh()

got_clean.groupby(['book5_A_Dance_with_Dragons', 'isAlive']).size()

# About 75% of charactesr live.
# Equal chance of living if not in book 5 than if you are.



# Is Alive vs. If Noble #

got_clean.groupby('isNoble').isAlive.value_counts().plot.barh()

got_clean.groupby(['isNoble', 'isAlive']).size()

# About 75% of nobles live.
# Eqaul chances of living if not a noble than if you are.



# Is Alive vs. Number of Dead Relatives #

got_clean.groupby('numDeadRelations').isAlive.value_counts().plot.barh()

got_clean.groupby(['numDeadRelations', 'isAlive']).size()

# As number of dead relatives increases so does chance of dying. 



###############################################################################
####################          TUNE & FLAG OUTLIERS         ####################
###############################################################################

## Get quantiles of each variable to flag outliers

got_quantiles = got_clean.loc[:, :].quantile([0.05,
                                              0.10,
                                              0.20,
                                              0.25,
                                              0.30,
                                              0.40,
                                              0.50,
                                              0.60,
                                              0.70,
                                              0.75,
                                              0.80,
                                              0.90,
                                              0.95])


## Create value for low and high  popularity outliers.
 #  Only appropriate variable to flag outliers as others are categorical
 #  To be considered outlier character must be in lowest 10% or highest 10%

 
popularity_low = .00668896 
popularity_hi = .200669


###############################################################################  


##########################
# Creating Outlier Flags #
##########################


# Popularity High

got_clean['hi_popularity'] = 0

for val in enumerate(got_clean.loc[ : , 'popularity']):
    
    if val[1] >= popularity_hi:
        got_clean.loc[val[0], 'hi_popularity'] = 1
        
        
# Popularity Low
        
got_clean['low_popularity'] = 0

for val in enumerate(got_clean.loc[ : , 'popularity']):
    
    if val[1] <= popularity_low:
        got_clean.loc[val[0], 'low_popularity'] = 1
    
# I assume that being of low or high popularity could have an effect on survival     
        
        
###############################################################################
#################            FEATURE ENGINGEERING             #################
###############################################################################        


#################################
##  Belong to Important House  ##
#################################
        
        
# From research I was able to determine that there are 7 important houses
 # The most important houses are:
 
 # 1) House Arryn, 2) House Greyjoy, 3) House Lannister, 4) House Stark, 
 # 5) House Targaryen, 6) House Tully, and 7) House Frey

 
# Taking this into considertion belonging to one of these houses probably affects
#   the probability of living or dying throughout the story


# Will be using the house codes for the code found in houses_codes df
 

# Belong to Important House
 
important_houses_list = [22, 119, 148, 232, 246, 256, 106]

got_clean['important_house'] = 0

for val in enumerate(got_clean.loc[ : , 'house_clean']):
    
    if val[1] in important_houses_list:
        
        got_clean.loc[val[0], 'important_house'] = 1
         



## There are two groups who's job it is to protect: Night's Watch & King's Guard
 # I assume belonging to these groups might affect chances of survival
 

# Job is to Protect: 1(yes) 0(no)
 
protect_list = [304, 299]

got_clean['protect'] = 0

for val in enumerate(got_clean.loc[ : , 'house_clean']):
    
    if val[1] in protect_list:
        
        got_clean.loc[val[0], 'protect'] = 1




###############################################################################


##############################
#  Alliance with Top Houses  #
##############################


## Researching, I found that there are 3 main houses to ally with:
 #   1) House Stark , 2) House Lannister, or 3) House Targaryen


## I would assume belonging to a house that is an alliance with each of these
 #   houses would affect the characters chances of survival.


## To make feature engineering more efficient, if the house was an alliance at
 #   one point but then switched alliances, that alliance not counted 


# Will be using the house codes for the code found in houses_codes df 
 
 
## Is Character in Stark Alliance?

# Define list of houses in Stark Alliance
 
stark_alliance = [232, 205, 109, 261, 101, 282, 184, 152, 256, 276, 22,
                  145, 246, 24, 56, 60, 90, 136, 164, 177, 198, 232]

got_clean['ally_stark'] = 0

for val in enumerate(got_clean.loc[ : , 'house_clean']):
    
    if val[1] in stark_alliance:
        
        got_clean.loc[val[0], 'ally_stark'] = 1



## Is Character in Lannister Alliance?

# Define list of houses in Lannister Alliance
 
lannister_alliance = [148, 26, 46, 47, 65, 78, 95, 96, 146, 150, 157, 160,
                      166, 190, 206, 200, 229, 244, 247, 106]

got_clean['ally_lannister'] = 0

for val in enumerate(got_clean.loc[ : , 'house_clean']):
    
    if val[1] in lannister_alliance:
        
        got_clean.loc[val[0], 'ally_lannister'] = 1
        
        
        
## Is Character in Targaryen Alliance?

# Define list of houses in Targaryen Alliance
 
targaryen_alliance = [246, 289, 291, 293, 294, 319, 295, 310, 167, 232, 258,
                      297, ]

got_clean['ally_targaryen'] = 0

for val in enumerate(got_clean.loc[ : , 'house_clean']):
    
    if val[1] in targaryen_alliance:
        
        got_clean.loc[val[0], 'ally_targaryen'] = 1



print(got_clean.isnull().sum())


# Export clean df to excel

got_clean.to_excel('got_clean.xlsx')


###############################################################################


### Bi-Variate Analysis of Engineered Features ##


# Is Alive vs. High Popularity #

got_clean.groupby('hi_popularity').isAlive.value_counts().plot.barh()

got_clean.groupby(['hi_popularity', 'isAlive']).size()

# Much higher probability of living if not highly popular



# Is Alive vs. Low Populuarity

got_clean.groupby('low_popularity').isAlive.value_counts().plot.barh()

got_clean.groupby(['low_popularity', 'isAlive']).size()

# About 90% of low popularity characters live



# Is Alive vs. Important House

got_clean.groupby('important_house').isAlive.value_counts().plot.barh()

got_clean.groupby(['important_house', 'isAlive']).size()

# About 70% of important house characters live



# Is Alive vs. Protect

got_clean.groupby('protect').isAlive.value_counts().plot.barh()

got_clean.groupby(['protect', 'isAlive']).size()

# About 70% of protector characters live



# Is Alive vs. Ally_Stark

got_clean.groupby('ally_stark').isAlive.value_counts().plot.barh()

got_clean.groupby(['ally_stark', 'isAlive']).size()

# About 65% of stark ally's live



# Is Alive vs. Ally_Lannister

got_clean.groupby('ally_lannister').isAlive.value_counts().plot.barh()

got_clean.groupby(['ally_lannister', 'isAlive']).size()

# About 75% of lannister ally's live



# Is Alive vs. Ally_Targaryen

got_clean.groupby('ally_targaryen').isAlive.value_counts().plot.barh()

got_clean.groupby(['ally_targaryen', 'isAlive']).size()

# About 65% of targaryen ally's live




###############################################################################
#################                 CORRELATIONS                #################
###############################################################################


got_cors = got_clean.corr().round(3)


## isAlive shows no strong correlation with any variable in the data set



###############################################################################
#################               SPLITTING DATA                #################
###############################################################################


## Seperating feature variables from target variable

got_feats = got_clean.drop(['S.No' ,        # do not need for prediction
                            'name',         # drop string data,
                            'isAlive'],     # this is our target variable
                             axis = 1)

got_target = got_clean.loc[:, 'isAlive']


## Split into test and training data

X_train, X_test, y_train, y_test = train_test_split(
            got_feats,
            got_target,
            test_size = 0.10,
            random_state = 508,
            stratify = got_target)



# Scale the data

feat_scaler = StandardScaler()
feat_scaler.fit(X_train)
X_train_2 = feat_scaler.transform(X_train)
X_test_2 = feat_scaler.transform(X_test)




###############################################################################
#################             LOGISTIC REGRESSION             #################
###############################################################################



######################################
#  Stats Model Logistic Regression   #
######################################


# Concatenate training data

got_train = pd.concat([X_train, y_train], axis = 1)


# Seperate features from  target in got_train

got_train_feats = got_train.drop(['isAlive'], axis = 1)


# Build model


smf_log = smf.logit(formula = """isAlive ~ male + 
                    book1_A_Game_Of_Thrones +
                    book2_A_Clash_Of_Kings + 
                    book3_A_Storm_Of_Swords +
                    book4_A_Feast_For_Crows + 
                    book5_A_Dance_with_Dragons +
                    isMarried + 
                    isNoble + 
                    protect +
                    numDeadRelations + 
                    popularity +
                    cul_clean +
                    house_clean + 
                    hi_popularity +
                    low_popularity +
                    important_house +
                    ally_stark +
                    ally_lannister +
                    ally_targaryen """,
                    data = got_train)

# Model results

results_smf_log = smf_log.fit()

print(results_smf_log.summary())



## The results suggest that the model fits the datat well

## All of our variables are significant except for one.

## I did not expect that being an ally of the lanisters would be our only
 #   insignificant variable in the dataset. 


# Other important summary statistics
print('AIC:', results_smf_log.aic.round(2))
print('BIC:', results_smf_log.bic.round(2))



###############################################################################


## Feature engineered variables ally_lannister and protect found to be non-significant
 # Will drop from data
 # Other variables found non-significant but weren't engineered
 
 
## Seperating feature variables from target variable

got_feats = got_clean.drop(['S.No' ,          # do not need for prediction
                            'name',           # drop string data
                            'ally_lannister', # not significant
                            'protect',        # not significant
                            'isAlive'],       # this is our target variable
                             axis = 1)

got_target = got_clean.loc[:, 'isAlive']


## Split into test and training data

X_train, X_test, y_train, y_test = train_test_split(
            got_feats,
            got_target,
            test_size = 0.10,
            random_state = 508,
            stratify = got_target)



# Scale the data

feat_scaler = StandardScaler()
feat_scaler.fit(X_train)
X_train_2 = feat_scaler.transform(X_train)
X_test_2 = feat_scaler.transform(X_test)




#################################
#  Sklearn Logistic Regression  #
#################################


######################
# Optimal Parameters #
######################


## The following code might take some time to run.
 #   In the interest of time it has already been and results are below
 
## If you would like to run simplye remove the  '''   '''   around the code


'''

# Creating Hyperparameter Grid

C_space = pd.np.arange(0.001, 10, 0.1)

solver_space = ['newton-cg', 'lbfgs']


param_grid = {'C'       : C_space,
              'solver'  : solver_space}


# Building Model Object

logreg_object = LogisticRegression(random_state = 508)


# Creating GridSearchCV object

logreg_grid = GridSearchCV(logreg_object,
                           param_grid,
                           cv = 3,
                           scoring = 'roc_auc',
                           return_train_score = False)


# Fitting to training data

logreg_grid.fit(X_train_2, y_train)


# What are the best parameters?

print("Tuned Logistic Regression Parameter:", logreg_grid.best_params_)
print("Tuned Logistic Regression Accuracy:",  logreg_grid.best_score_.round(4))

'''


# Best parameters are:
#       C      = 0.101
#       solver = newton-cg




# Prepping the Model
logreg = LogisticRegression(C = 0.101,  
                            solver = 'newton-cg')


# Fitting the Model
logreg_fit = logreg.fit(X_train_2, y_train)


# Predictions

pred_log_train = logreg_fit.predict(X_train_2)
pred_log_test = logreg_fit.predict(X_test_2)


# Let's compare the testing score to the training score.

print('Logreg Train AUC Score', roc_auc_score(y_train, pred_log_train).round(4))
print('Logreg Test AUC Score', roc_auc_score(y_test, pred_log_test).round(4))



# Cross-Validating the model with three folds

cv_auc_logreg = cross_val_score(logreg,
                                got_feats,
                                got_target,
                                cv = 3)


print(cv_auc_logreg)


print('\nAverage: ',
      pd.np.mean(cv_auc_logreg).round(3),
      '\nMinimum: ',
      min(cv_auc_logreg).round(3),
      '\nMaximum: ',
      max(cv_auc_logreg).round(3))




###############################################################################
#################                KNN CLASSIFIER               #################
###############################################################################


######################
# Optimal Parameters #
######################


## The following code might take some time to run.
 #   In the interest of time it has already been and results are below
 
## If you would like to run simplye remove the  '''   '''   around the code


'''

# Creating a hyperparameter grid

neighbor_space = pd.np.arange(100, 1350, 250)

leaf_space = pd.np.arange(1, 150, 15)

p_space = [1, 2, 3, 4, 5]


param_grid = {'n_neighbors'      : neighbor_space,
              'leaf_size'        : leaf_space,
              'p'                : p_space}


# Build model object
                  
knn_object = KNeighborsClassifier()


# Creating GridSearchCV object

knn_grid = GridSearchCV(knn_object,
                        param_grid,
                        cv = 3,
                        scoring = 'roc_auc',
                        return_train_score = False)


# Fitting to training data

knn_grid.fit(X_train_2, y_train)


# What are the best parameters?

print("Tuned KNN Parameter:", knn_grid.best_params_)
print("Tuned KNN Accuracy:",  knn_grid.best_score_.round(4))

'''


# Best parameters are:
    # n_neighbors =  100
    # leaf size   =  76
    # p           =  1



############################
# KNN Classification Model #
############################


# Creating a regressor object
knn_class = KNeighborsClassifier(n_neighbors = 100,
                                 leaf_size = 76,
                                 p = 1,
                                 weights = 'distance')


# Fit model
knn_class_fit = knn_class.fit(X_train_2, y_train)


# Predictions

pred_knn_train = knn_class.predict(X_train_2)
pred_knn_test = knn_class.predict(X_test_2)


# Let's compare the testing score to the training score.

print('KNN Train AUC Score', roc_auc_score(y_train, pred_knn_train).round(4))
print('KNN Test AUC Score', roc_auc_score(y_test, pred_knn_test).round(4))




# Cross-Validating the model with three folds

cv_auc_knn = cross_val_score(knn_class,
                             got_feats,
                             got_target,
                             cv = 3)


print(cv_auc_knn)


print('\nAverage: ',
      pd.np.mean(cv_auc_knn).round(3),
      '\nMinimum: ',
      min(cv_auc_knn).round(3),
      '\nMaximum: ',
      max(cv_auc_knn).round(3))




###############################################################################
#################          DECISION TREE CLASSIFIER           #################
###############################################################################


######################
# Optimal Parameters #
######################


## The following code might take some time to run.
 #   In the interest of time it has already been and results are below
 
## If you would like to run simplye remove the  '''   '''   around the code


'''

# Creating a hyperparameter grid

criterion_space = ['gini', 'entropy']

depth_space = pd.np.arange(1, 10, 1)

leaf_space = pd.np.arange(100, 300, 10)


param_grid_3 = {'criterion'        : criterion_space,
                'max_depth'        : depth_space,
                'min_samples_leaf' : leaf_space}


# Building the model object 

tree_object = DecisionTreeClassifier(random_state = 508)


# Creating a GridSearchCV object

tree_grid = GridSearchCV(tree_object, 
                         param_grid_3, 
                         cv = 3,
                         scoring = 'roc_auc',
                         return_train_score = False)


# Fitting to training data

tree_grid.fit(X_train_2, y_train)


# Print the optimal parameters and best score
print("Tuned Tree Parameter:", tree_grid.best_params_)
print("Tuned Tree Accuracy:", tree_grid.best_score_.round(4))

'''


# Best parameters:
    #  criterion        = gini  
    #  max_depth        = 3  
    #  min_samples_leaf = 150
       



#################
#  Build Model  #
#################


# Prep the Model

tree = DecisionTreeClassifier(criterion = 'gini',
                              max_depth = 3, 
                              min_samples_leaf = 150, # at least 10% obesrvations
                              random_state = 508)


# Fit the model

tree_fit = tree.fit(X_train_2, y_train)


# Predictions

pred_tree_train = tree_fit.predict(X_train_2)
pred_tree_test = tree_fit.predict(X_test_2)


# Let's compare the testing score to the training score.

print('Tree Classifier Train AUC Score', roc_auc_score(y_train, pred_tree_train).round(4))
print('Tree Classifier Test AUC Score', roc_auc_score(y_test, pred_tree_test).round(4))



####################
# Cross-Validating #
####################


cv_auc_tree = cross_val_score(tree,
                              got_feats,
                              got_target,
                              cv = 3)


print(cv_auc_tree)


print('\nAverage: ',
      pd.np.mean(cv_auc_tree).round(3),
      '\nMinimum: ',
      min(cv_auc_tree).round(3),
      '\nMaximum: ',
      max(cv_auc_tree).round(3))


###########################
# Visualize Decision Tree #
###########################


dot_data = StringIO()


export_graphviz(decision_tree = tree,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = got_feats.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png(),
      height = 500,
      width = 800)

graph.write_png('Decision Tree GOT Predcitions.png')




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



# GINI - Feature Importance #

plot_feature_importances(tree,
                         train = X_train)

Image(graph.create_png(),
      height = 500,
      width = 800)

graph.write_png('Feature Importance Tree Classifier.png')


# Popuplarity is by bar the most important feature


# Followed by whether or not you're in book 4
 #  This is the book with most characters in our dataset
 #  It makes sense as it is the second to last to book


# If you didn't make it into book 4 and you're in the top 30% of popularity;
 #  you have a higher chance of dying.


# If you didn't make it into book 4 and you're in the bottome 15%  of popularity; 
 #  you have a higher chance of living. 


# If you did make it into book 4, you're in the bottom half of popularity,
 #  and you're in book 3; you a have a very high chance of living.


# If you did make it into book 4, you're in the top half of popularity,
 #  and you're not a woman, you have a high chance of living. 



###############################################################################



## A lot of emphasis is placed popularity and book 4, 
  # What if we take them out to see what other variables important
  

## Seperating feature variables from target variable
  
got_feats_2 = got_clean.drop(['S.No' ,          
                              'name',          
                              'popularity',     
                              'hi_popularity',  
                              'low_popularity', 
                              'book4_A_Feast_For_Crows', 
                              'isAlive'],       
                               axis = 1)

got_target = got_clean.loc[:, 'isAlive']


## Split into test and training data

X_train_3, X_test_3, y_train_2, y_test_2 = train_test_split(
            got_feats_2,
            got_target,
            test_size = 0.10,
            random_state = 508,
            stratify = got_target)


# Prep the Model

tree_no_pop = DecisionTreeClassifier(criterion = 'gini',
                                max_depth = 3, 
                                min_samples_leaf = 149, 
                                random_state = 508)


# Fit the model

tree_no_pop_fit = tree_no_pop.fit(X_train_3, y_train_2)



###########################
# Visualize Decision Tree #
###########################


dot_data = StringIO()


export_graphviz(decision_tree = tree_no_pop,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = got_feats_2.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png(),
      height = 500,
      width = 800)

graph.write_png('Decision Tree GOT Predcitions - No Popularity.png')


#########################
#   Feature Importance  #
#########################


# Defining function to visualize feature importance

def plot_feature_importances(model, train = X_train_3, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train_3.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")   


# GINI - Feature Importance #
        

plot_feature_importances(tree_no_pop,
                         train = X_train_3)

Image(graph.create_png(),
      height = 500,
      width = 800)

graph.write_png('Feature Importance Tree Classifier (No Popularity).png')



## Book 4 is now the most important by far, reiterating the fact that if you 
 #   didn't make into book 4 you probably weren't that important to the overall
 #   story.
 

## Next up is being in book 1 closely followed by cultural background
 #   Culture wasn't important at all with popularity present in data. 
 
 
 
 
###############################################################################
#################               RANDOM FOREST                 #################
###############################################################################


## The following code might take some time to run.
 #   In the interest of time it has already been and results are below
 
## If you would like to run simplye remove the  '''   '''   around the code


'''

# Create Hyperparameter Grid

estimator_space = pd.np.arange(100, 1350, 250)

criterion_space = ['gini', 'entropy']

depth_space = pd.np.arange(1, 10, 1)

leaf_space = pd.np.arange(1, 150, 15)

bootstrap_space = [True, False]

warm_start_space = [True, False]


param_grid_4 = {'n_estimators'     : estimator_space,
              'criterion'        : criterion_space,
              'max_depth'        : depth_space,
              'min_samples_leaf' : leaf_space,
              'bootstrap'        : bootstrap_space,
              'warm_start'       : warm_start_space}


# Building the model object 

forest_object = RandomForestClassifier(random_state = 508)


# Creating a GridSearchCV object

forest_grid = GridSearchCV(forest_object, 
                           param_grid_4, 
                           cv = 3,
                           scoring = 'roc_auc',
                           return_train_score = False)


# Fit it to the training data

forest_grid.fit(X_train_2, y_train)


# Print the optimal parameters and best score

print("Tuned Forest Parameter:", forest_grid.best_params_)
print("Tuned Forest Accuracy:", forest_grid.best_score_.round(4))

'''


## Best parameters: 
    #  n_estimators     = 600
    #  criterion        = entropy
    #  max_depth        = 9
    #  min_samples_leaf = 1
    #  bootstrap        = False (but we will make it True)
    #  warm_start       = True (but we will make False)



#################
#  Build Model  #
#################


# Prep the model
 
forest = RandomForestClassifier(n_estimators = 600, 
                                criterion = 'entropy',
                                max_depth = 9,
                                min_samples_leaf = 1,
                                bootstrap = True,
                                warm_start = False,
                                random_state = 508)



# Fit the model

forest_fit = forest.fit(X_train_2, y_train)



# Predictions

pred_forest_train = forest_fit.predict(X_train_2)
pred_forest_test = forest_fit.predict(X_test_2)



# Let's compare the testing score to the training score.

print('Forest Train AUC Score', roc_auc_score(y_train, pred_forest_train).round(4))
print('Forest Test AUC Score', roc_auc_score(y_test, pred_forest_test).round(4))



####################
# Cross-Validating #
####################


cv_auc_forest = cross_val_score(forest,
                                got_feats,
                                got_target,
                                cv = 3)


print(cv_auc_forest)


print('\nAverage: ',
      pd.np.mean(cv_auc_forest).round(3),
      '\nMinimum: ',
      min(cv_auc_forest).round(3),
      '\nMaximum: ',
      max(cv_auc_forest).round(3))




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



# GINI - Feature Importance #

plot_feature_importances(forest,
                         train = X_train)

Image(graph.create_png(),
      height = 500,
      width = 800)

graph.write_png('Feature Importance Random Forest.png')




###############################################################################
#################         GRADIENT BOOSTED MACHINES           #################
###############################################################################


######################
# Optimal Parameters #
######################


## The following code might take some time to run.
 #   In the interest of time it has already been and results are below
 
## If you would like to run simplye remove the  '''   '''   around the code


'''

# Creating a hyperparameter grid

loss_space = ['deviance', 'exponential']
 
learn_space = pd.np.arange(0.1, 1.6, 0.1)

estimator_space = pd.np.arange(100, 1350, 250)

criterion_space = ['friedman_mse', 'mse', 'mae']

leaf_space = pd.np.arange(1, 150, 15)

depth_space = pd.np.arange(1, 10)


param_grid_5 = {'loss'              : loss_space,
                'learning_rate'     : learn_space,
                'n_estimators'      : estimator_space,
                'criterion'         : criterion_space,
                'min_samples_leaf'  : leaf_space,
                'max_depth'         : depth_space}



# Building the model object one more time

gbm_grid = GradientBoostingClassifier(random_state = 508)


# Creating a GridSearchCV object

gbm_grid_cv = GridSearchCV(gbm_grid, 
                           param_grid_5, 
                           cv = 3,
                           scoring = 'roc_auc',
                           return_train_score = False)


# Fit it to the training data

gbm_grid_cv.fit(X_train_2, y_train)



# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))

'''


# Best parameters:
    #  loss              =
    #  learning_rate     = 
    #  n_estimators      =  
    #  criterion         = 
    #  min_samples_leaf  =
    #  max_depth         =
       



#################
#  Build Model  #
#################


# Prep the model

gbm = GradientBoostingClassifier(loss = 'deviance',
                                 learning_rate = 1.0,
                                 n_estimators = 400,
                                 criterion = 'friedman_mse',
                                 min_samples_leaf = 1,
                                 warm_start = False,
                                 max_depth = 9,
                                 random_state = 508)


# Fit the model

gbm_fit = gbm.fit(X_train_2, y_train)


# Predictions

pred_gbm_train = gbm_fit.predict(X_train_2)
pred_gbm_test = gbm_fit.predict(X_test_2)



# Let's compare the testing score to the training score.

print('GBM Train AUC Score', roc_auc_score(y_train, pred_gbm_train).round(4))
print('GBM Test AUC Score', roc_auc_score(y_test, pred_gbm_test).round(4))



####################
# Cross-Validating #
####################


cv_auc_gbm = cross_val_score(gbm,
                             got_feats,
                             got_target,
                             cv = 3)


print(cv_auc_gbm)


print('\nAverage: ',
      pd.np.mean(cv_auc_gbm).round(3),
      '\nMinimum: ',
      min(cv_auc_gbm).round(3),
      '\nMaximum: ',
      max(cv_auc_gbm).round(3))





###############################################################################
#################         PRINCIPAL COMPONENT ANALYSIS        #################
###############################################################################


# Prep the model with three factors

pca = PCA(n_components = 3,
          random_state = 508)


# Fit the model

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.fit_transform(X_test)


# Apply PCA Classification

logreg_pca = LogisticRegression(solver = 'lbfgs')


# Fit the model

logreg_pca_fit = logreg_pca.fit(X_train_pca, y_train)


# Predictions

pred_pca_train = logreg_pca_fit.predict(X_train_pca)
pred_pca_test = logreg_pca_fit.predict(X_test_pca)


# Let's compare the testing score to the training score.

print('PCA Train AUC Score', roc_auc_score(y_train, pred_pca_train).round(4))
print('PCA Test AUC Score', roc_auc_score(y_test, pred_pca_test).round(4))



####################
# Cross-Validating #
####################


cv_auc_pca = cross_val_score(logreg_pca,
                             got_feats,
                             got_target,
                             cv = 3)


print(cv_auc_pca)


print('\nAverage: ',
      pd.np.mean(cv_auc_pca).round(3),
      '\nMinimum: ',
      min(cv_auc_pca).round(3),
      '\nMaximum: ',
      max(cv_auc_pca).round(3))




###############################################################################
##########            WHICH MODEL HAD HIGEST AVG. AUC SCORE          ########## 
###############################################################################


# AUC Logistic Regression

print('\n Logistic Regression Average: ',
      pd.np.mean(cv_auc_logreg).round(3))


# AUC KNN Classifier

print('\n KNN Classifier Average: ',
      pd.np.mean(cv_auc_knn).round(3))


# AUC Tree Classifier GINI

print('\n Decision Tree Classifier Average: ',
      pd.np.mean(cv_auc_tree).round(3))


# AUC Forest GINI

print('\n Random Forest Classifier Average: ',
      pd.np.mean(cv_auc_forest).round(3))


# AUC GBM

print('\n Gradient Boosted Machine Average: ',
      pd.np.mean(cv_auc_gbm).round(3))


# AUC PCA

print('\n Principal Component Analysis Average: ',
      pd.np.mean(cv_auc_pca).round(3))



## Get best results with Random Forest Classifier Model
 #  AUC Average : 0.802




###############################################################################
#################               CONFUSION MATRIX              #################
###############################################################################


# Create confusion matrix with best model
 
print(confusion_matrix(y_true = y_test,
                       y_pred = pred_forest_test))


# Visualizing a confusion matrix

labels = ['Dead', 'Alive']

cm = confusion_matrix(y_true = y_test,
                      y_pred = pred_forest_test)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'Blues')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()


# Our model did pretty well
  #  Correctly predicted the survival of 141 people
  #  Correctly predicted the death of 20 people
  #  Incorreclty predicted survival of 30 people
  #  Incorrectly predicted death of 4



###############################################################################
#################            CLASSIFICATION REPORT            #################
###############################################################################


print(classification_report(y_true = y_test,
                            y_pred = pred_forest_test,
                            target_names = labels))


## Correctly predicted survival about 80% of the time

## Correctly predicted death about 80% of the time






###############################################################################
##########                 STORING MODEL PREDICTIONS                 ########## 
###############################################################################



# We can store our predictions as a dictionary.
got_predictions_df = pd.DataFrame({'Actual' : y_test,
                                   'Forest Predicted': pred_forest_test})


got_predictions_df.to_excel("Martin Salas - GOT Survival Predcitions.xlsx")









###############################################################################
###############################################################################
##########                        NAVIGATING                         ########## 
##########                          BOOK 4                           ########## 
###############################################################################
###############################################################################


# As Book 4 is the breaking point in the data according to the decision tree model,
#   it seems logical to develop a strategy for navigating book 4. 


# Let's first make a dataset containg only characters appearing in Book 4

got_book4 = got_clean.loc[got_clean['book4_A_Feast_For_Crows'] == 1]



###############################################################################
#################               SPLITTING DATA                #################
###############################################################################


## Seperating feature variables from target variable, drop variables not needed

got_b4_feat = got_book4.drop(['S.No' ,        
                              'name',         
                              'isAlive',
                              'isMarried',
                              'popularity',
                              'book1_A_Game_Of_Thrones',
                              'book2_A_Clash_Of_Kings',
                              'book3_A_Storm_Of_Swords',
                              'book4_A_Feast_For_Crows',
                              'book5_A_Dance_with_Dragons'],
                               axis = 1)

got_b4_trgt = got_book4.loc[:, 'isAlive']


## Split into test and training data

X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(
            got_b4_feat,
            got_b4_trgt,
            test_size = 0.10,
            random_state = 508,
            stratify = got_b4_trgt)



# Scale the data

feat_scaler = StandardScaler()
feat_scaler.fit(X_train_4)
X_train_5 = feat_scaler.transform(X_train_4)
X_test_5 = feat_scaler.transform(X_test_4)



######################################
#  Stats Model Logistic Regression   #
######################################


# Concatenate training data

got_train_2 = pd.concat([X_train_4, y_train_4], axis = 1)


# Seperate features from  target in got_train

got_train_feats_2 = got_train_2.drop(['isAlive'], axis = 1)


# Build model


smf_log_2 = smf.logit(formula = """isAlive ~ male + 
                    isNoble + 
                    numDeadRelations + 
                    popularity +
                    cul_clean +
                    house_clean + 
                    hi_popularity +
                    low_popularity +
                    important_house +
                    ally_stark +
                    ally_lannister +
                    ally_targaryen """,
                    data = got_train)

# Model results

results_smf_log_2 = smf_log_2.fit()

print(results_smf_log_2.summary())


 


###################
#  Decision Tree  #
###################


# Prep the Model

tree_2 = DecisionTreeClassifier(criterion = 'gini',
                                max_depth = 3, 
                                min_samples_leaf = 150, # at least 10% obesrvations
                                random_state = 508)


# Fit the model

tree_fit_2 = tree_2.fit(X_train_4, y_train_4)


# Predictions

pred_tree_train_2 = tree_fit_2.predict(X_train_4)
pred_tree_test_2 = tree_fit_2.predict(X_test_4)



###########################
# Visualize Decision Tree #
###########################


dot_data = StringIO()


export_graphviz(decision_tree = tree_2,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = got_b4_feat.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png(),
      height = 500,
      width = 800)

graph.write_png('Decision Tree GOT B4 Predcitions.png')




#########################
#   Feature Importance  #
#########################


# Defining function to visualize feature importance

def plot_feature_importances(model, train = X_train_4, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train_4.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")



# GINI - Feature Importance #

plot_feature_importances(tree_2,
                         train = X_train_4)

Image(graph.create_png(),
      height = 500,
      width = 800)

graph.write_png('Feature Importance B4 Tree Classifier.png')



##################
#  Forest Model  #
##################


# Prep the model
 
forest_2 = RandomForestClassifier(n_estimators = 600, 
                                  criterion = 'entropy',
                                  max_depth = 9,
                                  min_samples_leaf = 1,
                                  bootstrap = True,
                                  warm_start = False,
                                  random_state = 508)



# Fit the model

forest_fit_2 = forest_2.fit(X_train_4, y_train_4)



# Predictions

pred_forest_train_2 = forest_fit_2.predict(X_train_4)
pred_forest_test_2 = forest_fit_2.predict(X_test_4)



#########################
#   Feature Importance  #
#########################



# Defining function to visualize feature importance

def plot_feature_importances(model, train = X_train_4, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train_4.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")



# GINI - Feature Importance #

plot_feature_importances(forest_2,
                         train = X_train_4)

Image(graph.create_png(),
      height = 500,
      width = 800)

graph.write_png('Feature Importance Random Forest B4.png')











