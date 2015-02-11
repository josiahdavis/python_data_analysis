'''
Data Analysis in Python
===================================
    - Spyder Basics
    - Reading, Summarizing data
    - Slicing / Filtering
    - Modifying Columns
    - Handling Missing Values
    - Indexing
    - Analyzing across time
    - Split-Apply-Combine
    - Merging data
    - Plotting
    - Bonus Content
'''

'''
Spyder Basics:
    Keystrokes
    Console / Editor
    IPython vs. Python Interpreter
    Setting the Working Directory
    Customizing the display


Key-strokes I will use:
    Ctrl + L:       Clears the console
    Ctrl + Enter:   Runs the line of code
    Ctrl + S:       Saves the file
    Up-arrow:       Retreives previous commands              
'''

'''
Reading, Summarizing data
'''

import pandas as pd
import numpy as np

# Running this next line of code assumes that your console working directory is set up correctly 
# To set up your working directory
#        1) Put the data and the script in the same working directory
#        2) Select the options buttom in the upper right hand cornder of the editor
#        3) Select "Set console working directory"
# The working directory can also be set programatically with the os module

ufo = pd.read_csv('ufo_sightings.csv')

ufo                 
ufo.head(10)          # Look at the top x observations
ufo.tail()            # Bottom x observations (defaults to 5)
ufo.describe()        # describe any numeric columns (unless all columns are non-numeric)
ufo.index             # "the index" (aka "the labels")
ufo.columns           # column names (which is "an index")
ufo.dtypes            # data types of each column
ufo.values            # underlying numpy array
ufo.info()            # concise summary

# DataFrame vs Series, selecting a column
type(ufo)
isinstance(ufo, pd.DataFrame)
ufo['State']
ufo.State            # equivalent
type(ufo.State)

# summarizing a non-numeric column
ufo.State.describe()        # Valuable if you have numeric columns, which you often will
ufo.State.value_counts() / ufo.shape[0]

'''
Slicing / Filtering / Sorting
'''
# selecting multiple columns
ufo[['State', 'City']]
my_cols = ['State', 'City']
ufo[my_cols]
type(ufo[my_cols])

# loc: filter rows by LABEL, and select columns by LABEL
ufo.loc[1,:]                            # row with label 1
ufo.loc[:3,:]                           # rows with labels 1 through 3
ufo.loc[1:3, 'City':'Shape Reported']   # rows 1-3, columns 'City' through 'Shape Reported'
ufo.loc[:, 'City':'Shape Reported']     # all rows, columns 'City' through 'Shape Reported'
ufo.loc[[1,3], ['City','Shape Reported']]  # rows 1 and 3, columns 'City' and 'Shape Reported'

# iloc: filter rows by POSITION, and select columns by POSITION
ufo.iloc[0,:]                       # row with 0th position (first row)
ufo.iloc[0:3,:]                     # rows with positions 0 through 2 (not 3)
ufo.iloc[0:3, 0:3]                  # rows and columns with positions 0 through 2
ufo.iloc[:, 0:3]                    # all rows, columns with positions 0 through 2
ufo.iloc[[0,2], [0,1]]              # 1st and 3rd row, 1st and 2nd column

# mixing: select columns by LABEL, then filter rows by POSITION
ufo.City[0:3]
ufo[['City', 'Shape Reported']][0:3]
    
# logical filtering
ufo[ufo.State == 'TX']
ufo.City[ufo.State == 'TX']
ufo[ufo.State == 'TX'].City             # Same thing
ufo[(ufo.State == 'CA') | (ufo.State =='TX')]
ufo[ufo.City.isin(['Austin','Dallas', 'Houston'])]

# sorting
ufo.State.order()                               # only works for a Series
ufo.sort_index()                                # sort rows by label
ufo.sort_index(ascending=False)
ufo.sort_index(by='State')                      # sort rows by specific column
ufo.sort_index(by=['State', 'Shape Reported'])  # sort by multiple columns
ufo.sort_index(by=['State', 'Shape Reported'], ascending=[False, True])  # specify sort order

# detecting duplicate rows
ufo.duplicated()                                # Series of logicals
ufo.duplicated().sum()                          # count of duplicates
ufo[ufo.duplicated(['State','Time'])]           # only show duplicates
ufo[ufo.duplicated()==False]                    # only show unique rows
ufo_unique = ufo[~ufo.duplicated()]             # only show unique rows
ufo.duplicated(['State','Time']).sum()          # columns for identifying duplicates

'''
EXERCISE: 
- Read in the dataset
- Find the frequency of UFO sightings by shape
- Determine the top three most common colors of the typical UFO
- Determine the best cities in Virginia to see UFOs
'''

# Read in the dataset
ufo = pd.read_csv('ufo_sightings.csv')

# Find the frequency of UFO sightings by shape
ufo['Shape Reported'].value_counts().order(ascending=False)

# Determine the top three most common colors of the typical UFO
ufo['Colors Reported'].value_counts()[:3]

# Determine the five cities in Virginia with the most UFO sightings
ufo[ufo.State == 'VA'].City.value_counts()[:5]

'''
Modifying Columns
'''

# add a new column as a function of existing columns
ufo['Location'] = ufo['City'] + ', ' + ufo['State']
ufo.head()

# rename columns inplace
ufo.rename(columns={'Colors Reported':'Colors', 'Shape Reported':'Shape'}, inplace=True)
ufo.head()

ufo2 = ufo.rename(columns={'Colors':'Color'})

# hide a column (temporarily)
ufo.drop(['Location'], axis=1)
ufo[ufo.columns[:-1]]

# delete a column (permanently)
del ufo['Location']

'''
Handling Missing Values
'''

# missing values are often just excluded
ufo.describe()                          # excludes missing values
ufo.Shape.value_counts()                # excludes missing values
ufo.Shape.value_counts(dropna=False)    # includes missing values (new in pandas 0.14.1)

# find missing values in a Series
ufo.Shape.isnull()       # True if NaN, False otherwise
ufo.Shape.notnull()      # False if NaN, True otherwise
ufo.Shape.isnull().sum() # count the missing values
ufo[ufo.Shape.notnull()].head()

ufo_shape_not_null = ufo[ufo.Shape.notnull()]

# find missing values in a DataFrame
ufo.isnull()
ufo.isnull().sum()

# drop missing values
ufo.dropna()             # drop a row if ANY values are missing
ufo.dropna(how='all')    # drop a row only if ALL values are missing

# fill in missing values
ufo.Colors.fillna(value='Unknown', inplace=True)
ufo.fillna('Unknown')


'''
Indexing
'''

# Create a new index
ufo.set_index('State', inplace=True)
ufo.head()
ufo.index
ufo.index.is_unique

# Slice using the index
ufo.loc['WY',:]
ufo.loc[['ND', 'WY'],:]
ufo.loc['ND':'WY',:]                # Error because of sorting
ufo.sort_index(inplace=True)
ufo.loc['ND':'WY',:]


# Reset the index
ufo.reset_index(inplace=True)

# Create a multi-index
ufo.set_index(['State', 'City'], inplace=True)
ufo.sort_index(inplace=True)
ufo
ufo.index

# Slice using the multi-index
ufo.loc[['ND', 'WY'],:]
ufo.loc['ND':'WY',:]
ufo.loc[('ND', 'Bismarck'),:]
ufo.loc[('ND', 'Bismarck'):('ND','Casselton'),:]

'''
Analyzing across time
'''

# Reset the index
ufo.reset_index(inplace=True)

# Convert Time column to date-time format (defined in Pandas)
# transforming to time
# Formatting Time: https://docs.python.org/2/library/time.html#time.strftime
ufo.dtypes
ufo['Time'] = pd.to_datetime(ufo['Time'], format="%m/%d/%Y %H:%M")
ufo.dtypes

# Compute date range
ufo.Time.min()
ufo.Time.max()

# Slice using time
ufo[ufo.Time > pd.datetime(1995, 1, 1)]
ufo[(ufo.Time > pd.datetime(1995, 1, 1)) & (ufo.State =='TX')]

# Set the index to time
ufo.set_index('Time', inplace=True)
ufo.sort_index(inplace=True)
ufo.head()

# Access particular times/ranges
ufo.loc['1995']
ufo.loc['1995-01']
ufo.loc['1995-01-01']

# Access range of times/ranges
ufo.loc['1995':]
ufo.loc['1995':'1996']
ufo.loc['1995-12-01':'1996-01']

# Access elements of the timestamp
# Date Componenets: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#time-date-components
ufo.index.year
ufo.index.month
ufo.index.weekday
ufo.index.day
ufo.index.time
ufo.index.hour

# Create a new variable with time element
ufo['Year'] = ufo.index.year
ufo['Day'] = ufo.index.day
ufo['Weekday'] = ufo.index.weekday
ufo['Hour'] = ufo.index.hour

'''
Split-Apply-Combine
'''

# for each year, calculate the count of sightings
ufo.groupby('Hour').City.count()
ufo.Hour.value_counts()             # Same as before

# for each Shape, calculate the first sighting, last sighting, and range of sightings. 
ufo.groupby('Shape').Year.min()
ufo.groupby('Shape').Year.max()

# Specify the variable outside of the apply statement
ufo.groupby('Shape').Year.apply(lambda x: x.max())

# Specifiy the variable within the apply statement
ufo.groupby('Shape').apply(lambda x: x.Year.max() - x.Year.min())

# Specify a custom function to use in the apply statement
def get_max_year(df):
    try:
        return df.Year.max()
    except:
        return ''
        
ufo.groupby('Shape').apply(lambda x: get_max_year(x))

# split/combine can occur on multiple columns at the same time
# for each Weekday / Hour combination, determine first sighting
ufo.groupby(['Weekday','Hour']).City.count()


'''
Merging data
'''

# Read in population data

pop = pd.read_csv('population.csv')
pop.head()

ufo.head()

# Merge the data together
ufo = pd.merge(ufo, pop, on='State', how = 'left')

# Specify keys if columns have different names
ufo = pd.merge(ufo, pop, left_on='State', right_on='State', how = 'left')

# Observe the new Population column
ufo.head()

# Check for values that didn't make it (length)
len(ufo[ufo.Population.isnull()])

# Check for values that didn't make it (values)
ufo[ufo.Population.isnull()]

# Change the records that didn't match up using np.where command
ufo['State'] = np.where(ufo['State'] == 'Fl', 'FL', ufo['State'])

# Alternatively, change the state using native python string functionality
ufo['State'] = ufo['State'].str.upper()

# Merge again, this time get all of the records
ufo = pd.merge(ufo, pop, on='State', how = 'left')

'''
Writing Data
'''
ufo.to_csv('ufo_new.csv')               # First column is an index
ufo.to_csv('ufo_new.csv', index=False)  # First column is no longer index

'''
Other useful features
'''

# map values to other values
ufo['Weekday'] = ufo.Weekday.map({  0:'Mon', 1:'Tue', 2:'Wed', 
                                    3:'Thu', 4:'Fri', 5:'Sat', 
                                    6:'Sun'})

# Pivot rows / columns
ufo.groupby(['Weekday','Hour']).City.count()
ufo.groupby(['Weekday','Hour']).City.count().unstack(0) # Make first row level a column
ufo.groupby(['Weekday','Hour']).City.count().unstack(1) # Make second row level a column

'''
Plotting
'''

# Plot the number of sightings over time
ufo.groupby('Year').City.count().plot(kind='line', color='r', linewidth=3)

# Plot the number of sightings over the time of day
ufo.groupby(['Weekday','Hour']).City.count().unstack(0).plot(kind='line', linewidth=3)

# bar plot of UFOs in July 2014 (remove)
ufo.Shape.value_counts().plot(kind='bar', title='Sightings per shape')

ufo.loc['2014-07'].Day.value_counts().plot(kind='bar', title='Sightings per day')

'''
EXERCISE:   Plot the number of UFOs by day
BONUS:      Plot the number of UFOs by the time of day, for each weekday
'''

# Plot the number of sightings over time of day
ufo['Hour'] = ufo.index.hour

# Split up the plot by days of the week
ufo['Weekday'] = ufo.index.weekday
col = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for i in range(7):    
    ufo[ufo.Weekday == i].groupby(['Hour']).City.count().plot(linewidth=2, colors = col[i])

'''
EXERCISE:   Plot the number of sightings by day of the month for 
            the last 5 months of July
BONUS:      Repeat with each year as an individual line
'''




'''
EXERCISE: Working with drinks data
'''

# Read drinks.csv into a DataFrame called 'drinks' (use the default index)
drinks = pd.read_table('../data/drinks.csv', sep=',')
drinks = pd.read_csv('../data/drinks.csv')              # equivalent

# Print the first 10 rows
drinks.head(10)

# Examine the data types of all columns
drinks.dtypes
drinks.info()

# Print the 'beer_servings' Series
drinks.beer_servings
drinks['beer_servings']

# Calculate the average 'beer_servings' for the entire dataset
drinks.describe()                   # summarize all numeric columns
drinks.beer_servings.describe()     # summarize only the 'beer_servings' Series
drinks.beer_servings.mean()         # only calculate the mean

# Print all columns, but only show rows where the country is in Europe
drinks[drinks.continent=='EU']

# Calculate the average 'beer_servings' for all of Europe
drinks[drinks.continent=='EU'].beer_servings.mean()

# Only show European countries with 'wine_servings' greater than 300
drinks[(drinks.continent=='EU') & (drinks.wine_servings > 300)]

# Determine which 10 countries have the highest 'total_litres_of_pure_alcohol'
drinks.sort_index(by='total_litres_of_pure_alcohol').tail(10)

# Determine which country has the highest value for 'beer_servings'
drinks[drinks.beer_servings==drinks.beer_servings.max()].country

# Count the number of occurrences of each 'continent' value and see if it looks correct
drinks.continent.value_counts()


'''
Bonus Content:

Additional wrangling / plotting
'''

# Create a new dataframe
state_ufo = ufo.groupby('State').Population.mean()

# Convert from series to DataFrame
state_ufo = pd.DataFrame(state_ufo, columns=['Population'])

# Pull in the region of the state
state_ufo['Region'] = ufo.groupby('State').apply(lambda x: x.Region.iloc[0])

# Add in the number of UFO sightings
state_ufo['ufo_sightings'] = ufo.groupby('State').Population.count() 

# Normalize the sightings to account for population differences
state_ufo['sightings_norm'] = 100000 * state_ufo['ufo_sightings'] / state_ufo['Population']

# Observe the results
state_ufo.head()

# Create a histogram
state_ufo.sightings_norm.hist()

# Control the number of bins
state_ufo.sightings_norm.hist(bins=20)

# Create a historgram stratified by a categorical variable
state_ufo.sightings_norm.hist(by=state_ufo.Region, sharex=False)
state_ufo.boxplot(column='sightings_norm', by='Region')

# Scatter plot between two continuous variables
state_ufo.plot(x='ufo_sightings', y='Population', kind='scatter', s=100, alpha=0.3)