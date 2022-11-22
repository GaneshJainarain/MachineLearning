# Machine Learning Project using Python 

## General machine learning workflow step-by-step
   - Data cleaning and formatting
   - Exploratory data analysis
   - Feature engineering and selection
   - Compare several machine learning models on a performance metric
   - Perform hyper-parameter tuning on the best model
   - Evaluate the best model on the testing set
   - Interpret the model results
   - Draw conclusions and document work

# Problem Definition
   In this project, we will work with publicly available building energy data from New York City.
   The objective is to use the energy data to build a model that can predict the Energy Star Score of a building and interpret the results to find the factors which influence the score.
   The data includes the `Energy Star Score`, which makes this a supervised regression machine learning task:
   - `Supervised:` we have access to both the features and the target and our goal is to train a model that can learn a mapping between the two.
   - `Regression:` The Energy Star score is a continuous variable
   
   We want to develop a model that is both `accurate` — it can predict the Energy Star Score close to the true value — and `interpretable` — we can understand the model predictions. Once we know the goal, we can use it to guide our decisions as we dig into the data and build models.
# Data Cleaning
Top of our data, Loading in the data as a Pandas DataFrame
```python
import pandas as pd
import numpy as np
# Read in data into a dataframe 
data = pd.read_csv('/Users/richeyjay/Desktop/MachineLearning/env/Code/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv')
# Display top of dataframe
print(data.head())
```
![Top of our Data](env/TerminalOutput/DataHead.png)
If we observe our subset of data that contains 60 columns we can already see some issues.
First, we know that we want to predict the `ENERGY STAR` score but we don’t know what any of the columns mean.
We want to focus on `interpretability`, so its important to understand at least some of the columns.


After some digging on the web pertaining to the file name "Local Law 84" [Local Law 84](https://www.nyc.gov/site/buildings/codes/compliance-instructions.page) I found an NYC law requiring all buildings of a certain size to report their energy use.
Some more searching I found the definitions of the columns. [Definitions of Columns](https://www.nyc.gov/html/gbee/downloads/misc/nyc_benchmarking_disclosure_data_definitions_2017.pdf)

`Energy Star Score` is defined as 'A 1-to-100 percentile ranking based on self-reported energy usage for the reporting year. The Energy Star score is a relative measure used for comparing the energy efficiency of buildings.'

   ## Handling "Not Available" missing values
This is a string in Python which means that even the columns with numbers will be stored as object datatypes because Pandas converts a column with any strings into a column of all strings. We can see the datatypes of the columns using the dataframe.info() method:
```python
# See the column data types and non-missing values
print(data.info())
```
![Column Data](env/TerminalOutput/ColumnData.png)
If we observe some of the columns in the data we can see that some contain numerical metric units such as
ft², kBtu, kWh etc, but we see that they are stored as objects. Since we cant do numerical analysis on strings
they will have to be converted to number (specifically float) data types
## Converting Data to correct types
We convert the columns with numbers into numeric data types by replacing the "Not Available" entries with np.nan which can be interpreted as floats. Then we will convert the columns that contain numeric values (such as square feet or energy usage) into numeric datatypes.

```python
# Replace all occurrences of Not Available with numpy not a number
data = data.replace({'Not Available': np.nan})

# Iterate through the columns
for col in list(data.columns):
    # Select columns that should be numeric
    if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in 
        col or 'therms' in col or 'gal' in col or 'Score' in col):
        # Convert the data type to float
        data[col] = data[col].astype(float)
```
```python
data.describe()
```

![Corrected Data Types](env/TerminalOutput/CorrectedDataTypes.png)

## Handling Missing Values in dataset 
We want to start analyzing our data by looking at the percentage of missing values in each column. 
Missing values are fine when we do `Exploratory Data Analysis`, but they will have to be filled in for machine learning methods.

```python
# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

print(missing_values_table(data))
```
![Missing Value Percentage per col](env/TerminalOutput/MissingValuePercentagePerCol.png)
Although we want to be careful to not discard information and should be careful when dropping columns, if a column has a high percentage of missing values, then it probably will not be of much use; we will remove any columns with more than 50% missing values.

```python
# Get the columns with > 50% missing
missing_df = missing_values_table(data);
missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)
print('We will remove %d columns.' % len(missing_columns))
```
'''
Your selected dataframe has 60 columns.
There are 46 columns that have missing values.
We will remove 11 columns.
'''
```python 
#Drop the columns
data = data.drop(columns = list(missing_columns))
```

## Exploratory Data Analysis

`Exploratory Data Analysis (EDA)` is an open-ended process where we make plots and calculate statistics in order to explore our data. The purpose is to to find anomalies, patterns, trends, or relationships. These may be interesting by themselves (for example finding a correlation between two variables) or they can be used to inform modeling decisions such as which features to use. In short, the goal of EDA is to determine what our data can tell us! EDA generally starts out with a high-level overview, and then narrows in to specific parts of the dataset once as we find interesting areas to examine.

To begin the EDA, we will focus on a single variable, the `Energy Star Score`, because this is the target for our machine learning models. We can rename the column to score for simplicity and then start exploring this value.

## Single Variable Plots
The goal is to predict the Energy Star Score (renamed to score in our data) so a reasonable place to start is examining the distribution of this variable. A histogram is a simple yet effective way to visualize the distribution of a single variable and is easy to make using matplotlib.
```python
# Rename the score 
data = data.rename(columns = {'ENERGY STAR Score': 'score'})

# Histogram of the Energy Star Score
plt.style.use('fivethirtyeight')
plt.hist(data['score'].dropna(), bins = 100, edgecolor = 'k');
plt.xlabel('Score'); plt.ylabel('Number of Buildings'); 
plt.title('Energy Star Score Distribution');
plt.show()
```
![Energy Star Score Distribution](env/TerminalOutput/EnergyStarScoreDistribution.png)


The Energy Star Score is a percentile rank, we would expect to see a completely flat distribution with each score making up 1% of the distribution (about 90 buildings). However, this is definitely not the case as we can see that the two most common scores, 1 and 100, make up a disproportionate number of the overall scores.

If we go back to the definition of the score, we see that it is based on “self-reported energy usage” which might explain the very high scores. As a result, this probably is not the most objective measure of a building’s energy efficiency. our objective is only to predict the score and not to devise a better method of scoring buildings!

## Looking for Relationships

A major part of EDA is searching for relationships between the features and the target. Variables that are correlated with the target are useful to a model because they can be used to predict the target. One way to examine the effect of a categorical variable (which takes on only a limited set of values) on the target is through a density plot using the `seaborn` library. The first plot we will make shows the distribution of scores by the property type. In order to not clutter the plot, we will limit the graph to building types that have more than 100 observations in the dataset.

![Density Plot of Energy Star Scores by Borough](env/TerminalOutput/DensityPlotEnergyStarScoresBorough.png)
The borough of the building does not seem to make as significant a difference in the distribution of the score as does the building type. Nonetheless, it might make sense to include the borough as a categorical variable.