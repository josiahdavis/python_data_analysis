'''
    ======================================
    P Y T H O N    for    A N A L Y S I S
    ======================================

    --- UFO data ---
    Scraped from: http://www.nuforc.org/webreports.html
    Write up about this data: http://josiahjdavis.com/2015/01/01/identifying-with-ufos/   
    
    --- Drinks data ---
    Downloaded from: https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption
    Write up about this data: http://fivethirtyeight.com/datalab/dear-mona-followup-where-do-people-drink-the-most-beer-wine-and-spirits/
    
    --- Spyder Reference (Windows) ---
        Ctrl + L        (in the console):       Clears the console
        Up/Down-arrows  (in the console):       Retreives previous commands
        Ctrl + Enter    (in the editor):        Runs the line of code
        Ctrl + S        (in the editor):        Saves the file  
'''



'''
Reading, Summarizing data
'''

import pandas as pd  # Import an already installed python package
import numpy as np

# Running this next line of code assumes that your console working directory is set up correctly 
# To set up your working directory
#        1) Put the data and the script in the same working directory
#        2) Select the options buttom in the upper right hand cornder of the editor
#        3) Select "Set console working directory"

# Read a the csv file from your computer (after setting working directory)
ufo = pd.read_csv('ufo_sightings.csv')

# Alternatively, specify the file path
ufo = pd.read_csv('C:/Users/josdavis/Documents/Personal/GitHub/python_data_analysis/ufo_sightings.csv')

# Alterntively read in the file from the internet
ufo = pd.read_csv('https://raw.githubusercontent.com/josiahdavis/python_data_analysis/master/ufo_sightings.csv')

# Finding help on a function
help(pd.read_csv)

# Summarize the data that was just read in
ufo.head(10)          # Look at the top x observations
ufo.tail()            # Bottom x observations (defaults to 5)
ufo.describe()        # describe any numeric columns (unless all columns are non-numeric)
ufo.index             # "the index" (aka "the labels")
ufo.columns           # column names (which is "an index")
ufo.dtypes            # data types of each column
ufo.values            # underlying numpy array
ufo.info()            # concise summary

# Select a single column (a series)
ufo['State']
ufo.State                       # This is equivalent

# Select multiple columns (a dataframe)
ufo[['State', 'City','Shape Reported']]
my_cols = ['State', 'City', 'Shape Reported']
ufo[my_cols]                    # This is equivalent


'''
Filtering and Sorting Data
'''
    
# Logical filtering
ufo[ufo.State == 'TX']
ufo[~(ufo.State == 'TX')]       # Select everything where the test fails
ufo[ufo.State != 'TX']          # Same thing
ufo.City[ufo.State == 'TX']
ufo[ufo.State == 'TX'].City     # Same thing
ufo[(ufo.State == 'CA') | (ufo.State =='TX')]
ufo_dallas = ufo[(ufo.City == 'Dallas') & (ufo.State =='TX')]
ufo[ufo.City.isin(['Austin','Dallas', 'Houston'])]

# Sorting
ufo.State.order()                               # only works for a Series
ufo.sort_index(ascending=False)                 # sort rows by row labels
ufo.sort_index(ascending=False, inplace=True)   # Sort rows inplace
ufo.sort_values(by='State')                      # sort rows by specific column
ufo.sort_values(by=['State', 'Shape Reported'])  # sort by multiple columns
ufo.sort_values(by=['State', 'Shape Reported'], ascending=[False, True], inplace=True)  # specify sort order

'''
Modifying Columns
'''

# Add a new column as a function of existing columns
ufo['Location'] = ufo['City'] + ', ' + ufo['State']
ufo.head()

# Rename columns
ufo.rename(columns={'Colors Reported':'Colors', 'Shape Reported':'Shape'}, inplace=True)

# Hide a column (temporarily)
ufo.drop(['Location'], axis=1)

# Delete a column (permanently)
del ufo['Location']

'''
Handling Missing Values
'''

# Missing values are often just excluded
ufo.describe()                          # Excludes missing values
ufo.Shape.value_counts()                # Excludes missing values
ufo.Shape.value_counts(dropna=False)    # Includes missing values

# Find missing values in a Series
ufo.Shape.isnull()       # True if NaN, False otherwise
ufo.Shape.notnull()      # False if NaN, True otherwise
ufo.Shape.isnull().sum() # Count the missing values

# Find missing values in a DataFrame
ufo.isnull()
ufo.isnull().sum()
ufo[(ufo.Shape.notnull()) & (ufo.Colors.notnull())]

# Drop missing values
ufo.dropna()             # Drop a row if ANY values are missing
ufo.dropna(how='all')    # Drop a row only if ALL values are missing

# Fill in missing values
ufo.Colors.fillna(value='Unknown', inplace=True)
ufo.fillna('Unknown')

'''
EXERCISE: Working with drinks data
'''

# Read drinks.csv (in the drinks_data folder) into a DataFrame called 'drinks'


# Print the first 10 rows


# Examine the data types of all columns


# Print the 'beer_servings' Series


# Calculate the average 'beer_servings' for the entire dataset


# Print all columns, but only show rows where the country is in Europe


# Calculate the average 'beer_servings' for all of Europe


# Only show European countries with 'wine_servings' greater than 300


# Determine which 10 countries have the highest 'total_litres_of_pure_alcohol'


# Determine which country has the highest value for 'beer_servings'


# Count the number of occurrences of each 'continent' value and see if it looks correct


# Determine which countries do not have continent designations


# Determine the number of countries per continent. Does it look right?

'''
SOLUTIONS: Working with drinks data
'''

# Read drinks.csv (in the drinks_data folder) into a DataFrame called 'drinks'
drinks = pd.read_csv('drinks_data/drinks.csv')

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
drinks[['country', 'beer_servings']].sort_index(by='beer_servings', ascending=False).head(5)

# Count the number of occurrences of each 'continent' value and see if it looks correct
drinks.continent.value_counts()

# Determine which countries do not have continent designations
drinks[drinks.continent.isnull()].country

# Due to "na_filter = True" default within pd.read_csv()
help(pd.read_csv)

'''
Indexing and Slicing Data
'''

# Create a new index
ufo.set_index('State', inplace=True)
ufo.index
ufo.index.is_unique
ufo.sort_index(inplace=True)
ufo.head(25)

# loc: filter rows by LABEL, and select columns by LABEL
ufo.loc['FL',:]                                     # row with label FL
ufo.loc[:'FL',:]                                    # rows with labels 'FL' through ''
ufo.loc['FL':'HI', 'City':'Shape']         # rows FL, columns 'City' through 'Shape Reported'
ufo.loc[:, 'City':'Shape']                 # all rows, columns 'City' through 'Shape Reported'
ufo.loc[['FL', 'TX'], ['City','Shape']]    # rows FL and TX, columns 'City' and 'Shape Reported'

# iloc: filter rows by POSITION, and select columns by POSITION
ufo.iloc[0,:]                       # row with 0th position (first row)
ufo.iloc[0:3,:]                     # rows with positions 0 through 2 (not 3)
ufo.iloc[0:3, 0:3]                  # rows and columns with positions 0 through 2
ufo.iloc[:, 0:3]                    # all rows, columns with positions 0 through 2
ufo.iloc[[0,2], [0,1]]              # 1st and 3rd row, 1st and 2nd column

# Add another level to the index
ufo.set_index('City', inplace=True, append=True) # Adds to existing index
ufo.sort_index(inplace=True)
ufo.head(25)

# Slice using the multi-index
ufo.loc[['ND', 'WY'],:]
ufo.loc['ND':'WY',:]
ufo.loc[('ND', 'Bismarck'),:]
ufo.loc[('ND', 'Bismarck'):('ND','Casselton'),:]

# Reset the index
ufo.reset_index(level='City', inplace=True) # Remove a certain label from the index
ufo.reset_index(inplace=True)               # Remove all labels from the index

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
ufo.loc['1995',:]
ufo.loc['1995-01',:]
ufo.loc['1995-01-01',:]

# Access range of times/ranges
ufo.loc['1995':,:]
ufo.loc['1995':'1996',:]
ufo.loc['1995-12-01':'1996-01',:]

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
ufo['Month'] = ufo.index.month
ufo['Day'] = ufo.index.day
ufo['Weekday'] = ufo.index.weekday
ufo['Hour'] = ufo.index.hour

'''
Split-Apply-Combine
'''

# for each year, calculate the count of sightings
ufo.groupby('Year').size()
ufo.Year.value_counts()             # Same as before

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

# Split/combine can occur on multiple columns at the same time
# For each Weekday / Hour combination, determine a count of sightings
ufo.groupby(['Weekday','Hour']).size()

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
ufo.Population.isnull().sum()

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
ufo.to_csv('ufo_new.csv')               
ufo.to_csv('ufo_new.csv', index=False)  # Index is not included in the csv

'''
Other useful features
'''

# Detect duplicate rows
ufo.duplicated()                                # Series of logicals
ufo.duplicated().sum()                          # count of duplicates
ufo[ufo.duplicated(['State','Time'])]           # only show duplicates
ufo[ufo.duplicated()==False]                    # only show unique rows
ufo_unique = ufo[~ufo.duplicated()]             # only show unique rows
ufo.duplicated(['State','Time']).sum()          # columns for identifying duplicates

# Replace all instances of a value (supports 'inplace=True' argument)
ufo.Shape.replace('DELTA', 'TRIANGLE')   # replace values in a Series
ufo.replace('PYRAMID', 'TRIANGLE')       # replace values throughout a DataFrame

# Replace with a dictionary
ufo['Weekday'] = ufo.Weekday.replace({  0:'Mon', 1:'Tue', 2:'Wed', 
                                    3:'Thu', 4:'Fri', 5:'Sat', 
                                    6:'Sun'})

# Pivot rows to columns
ufo.groupby(['Weekday','Hour']).size()
ufo.groupby(['Weekday','Hour']).size().unstack(0) # Make first row level a column
ufo.groupby(['Weekday','Hour']).size().unstack(1) # Make second row level a column
# Note: .stack transforms columns to rows

# Randomly sample a DataFrame
idxs = np.random.rand(len(ufo)) < 0.66   # create a Series of booleans
train = ufo[idxs]                        # will contain about 66% of the rows
test = ufo[~idxs]                        # will contain the remaining rows



'''
Advanced Examples (w/Plotting)
'''

# Plot the number of sightings over time
ufo.groupby('Year').size().plot(kind='line', 
                                color='r', 
                                linewidth=2, 
                                title='UFO Sightings by year')
                                        
# Plot the number of sightings over the day of week and time of day
ufo.groupby(['Weekday','Hour']).size().unstack(0).plot(   kind='line', 
                                                                linewidth=2,
                                                                title='UFO Sightings by Time of Day')

# Plot the sightings in in July 2014
ufo[(ufo.Year == 2014) & (ufo.Month == 7)].groupby(['Day']).size().plot(  kind='bar',
                                                        color='b', 
                                                        title='UFO Sightings in July 2014')
                                                        
# Plot multiple plots on the same plot (plots neeed to be in column format)              
ufo_fourth = ufo[(ufo.Year.isin([2011, 2012, 2013, 2014])) & (ufo.Month == 7)]
ufo_fourth.groupby(['Year', 'Day']).City.count().unstack(0).plot(   kind = 'bar',
                                                                    subplots=True,
                                                                    figsize=(7,9))