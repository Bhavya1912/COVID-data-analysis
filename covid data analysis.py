from google.colab import files # I can import files directly from my system

# Upload the zip folder
uploaded = files.upload()
# Unzip the data
unzip "/content/covid19-global-forecasting-week-5.zip"
# Import the libraries
import pandas as pd # Library for Data Analysis
import numpy as np # Numerically Python --> Doing computations in an Optimised way (Hardware Optimised)
import matplotlib.pyplot as plt #visualising data line charts and bar graphs



covid_data_train = pd.read_csv("/content/train.csv") #csv is comma seperated values
covid_data_test = pd.read_csv("/content/test.csv")
covid_data_train.head() #will show first 5  rows
covid_data_train.shape # tells(num of rows, num of colums) #NaN---> nor a number also known as num value
covid_data_train.isnull().sum() #this code tells how many null values are there 
covid_data_test.head()
covid_data_train[covid_data_train['Country_Region'] == 'India'] # this code give data for india 
covid_data_train[covid_data_train['Country_Region'] == 'China'] #this code shows me data for china
covid_data_train[covid_data_train['Country_Region'] == 'India']['County'].value_counts() # returns the count for that perticular 
covid_data_test.isnull().sum()
covid_data_train.dropna(axis=1, inplace = True) # Drop the null columns --> County and Province
covid_data_test.dropna(axis=1, inplace=True) # Drop the null columns




import plotly.express as px # Data visualisation using plotly
import plotly.express as px # Data visualisation using plotly

fig = plt.figure(figsize = (45,30)) # intialize the figure
fig = px.pie(covid_data_train, names = 'Country_Region', values = 'TargetValue', color_discrete_sequence = px.colors.sequential.RdBu, hole = 0.4) # Plot the pieplot
fig.update_traces(textposition = 'inside') # Update the tracing 
#fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide') # Update the layout
fig.show()

# Visualise the counts of confimred cases and the fatalities
import seaborn as sns 
sns.barplot(y = 'TargetValue', x='Target', data = covid_data_train)
plt.show()

# Visualise the count of target w.r.t population
sns.barplot(x = 'Target', y= 'Population', data = covid_data_train)
plt.show()

# Visualise the top 10 most populos countries in the world and the covid cases
grouped_data = covid_data_train.groupby('Country_Region').sum()

# Top 10 most populous country
top_10_pop_countries=grouped_data.nlargest(10, 'Population')['TargetValue'] #extract top 10 most populated countries

# Visualise the number of confirmed covid cases and deaths i.e the Target Variable
fig = px.bar(x = top_10_pop_countries.index, y = top_10_pop_countries.values, title='Top 10 most populous countries versus number of covid cases', labels = dict(x='Countries', y='Number of Covid-19 Cases'))
fig.show()

covid_data_train.info()

# Convert the date column into datetime format
covid_data_train['Date'] = pd.to_datetime(covid_data_train['Date'])
covid_data_test['Date'] = pd.to_datetime(covid_data_test['Date'])


# Visualise the worldwide covid growth w.r.t time

# 1) Group the data by date
date_grouped_data = covid_data_train.groupby('Date').sum()

# 2) Plot the date grouped data on a line chart
fig = px.line(x=date_grouped_data.index, y = date_grouped_data['TargetValue'], title = 'Growth of number of COVID-19 cases over time', labels = dict(x='Date', y = 'Number of Coivd-19 Cases'))
fig.show()

covid_data_train[covid_data_train['Country_Region'] == 'India']

# Visulalisng the growth of covid w.r.t country over time
fig = px.line(covid_data_train, x = 'Date', y = 'TargetValue', color='Country_Region')
fig.show()

top_10_populous_countries = list(top_10_pop_countries.index)
top_10_populous_countries

top_10_most_pop_countries = covid_data_train[(covid_data_train['Country_Region'] == 'China') | (covid_data_train['Country_Region'] == 'India')|(covid_data_train['Country_Region'] == 'US')|(covid_data_train['Country_Region'] == 'Indonesia')|(covid_data_train['Country_Region'] == 'Brazil')|(covid_data_train['Country_Region'] == 'Pakistan')|(covid_data_train['Country_Region'] == 'Nigeria')|(covid_data_train['Country_Region'] == 'Bangladesh')|(covid_data_train['Country_Region'] == 'Russia')|(covid_data_train['Country_Region'] == 'Japan')]

# Visualise the growth of Covid-19 numbers in top 10 most populous countries
fig = px.line(top_10_most_pop_countries, x='Date', y='TargetValue', color='Country_Region')
fig.show()

