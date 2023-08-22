#!/usr/bin/env python
# coding: utf-8

# # Project:Covid_19_Worldwide Data Ananlysis
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# ## Introduction
# Dear community,
# 
# Hope you are well. In this area I want to clarify information about the dataset I used, some questions I posed to investigate it, and also communicate my results of the investigation. In this project, I will analyze the Covid_19__worldwide dataset to communicate useful results that help to explore important information that will help to find answers to our questions and maybe taking some decisions. Here I will mention a number of questions I posed to investigate the dataset and why some movies are successful and others are not. That depends on some metrics will discuss through our questions.
# 
# .........................................
# 
# .....................................................
# 
# 1- Exploring the number of countries in each region ?
# 
# 2- what are the countries that are with the highest population of fully vaccinated ?
# 
# 3- What is the region with the highest % of population vaccinated ?
# 
# 4- What are countries with the Highest social support ?
# 
# 5- What are regions with the Highest social support ?
# 
# 6- What are regions with the Highest Healthy life expectancy  ?
# 
# 7- What are the regions with the Highest population vaccinated and fully vaccinated?
# 
# 8- Does vaccination depend on social support  ?
# 
# 9- what is the correlation between the GDP and Vaccination ?
# 
# 10- what is the effect of  social support & healthy life expectancy on % of population vaccinated  ?
# 
# 11- what is the vaccination influence on countries ?
# 
# 
# 
# 
# 

# **Firstly...... Importing some useful libiraries for our investigation.**

# In[41]:


# Wrangling
import pandas as pd
import numpy as np

# Viz
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Cluster
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Your clustering code here
# ...




# ## Data Wrangling
#  
#  **Secondly, I am going to load my dataset, assess it to check for any issues, and do cleaning operations to get clear and clean data for appropriate investigation.**
# 
# ### General Properties

# In[42]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
covid = pd.read_csv('Worldwide Vaccine Data.csv')
covid.head()


# In[43]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
happiness = pd.read_csv('world-happiness-report-2021.csv')
happiness.head()


# * checking data shape number of columns and rows that contain this data.

# In[44]:


covid.shape


# In[45]:


happiness.shape


# * Doing some descriptive statistics.

# In[46]:


covid.describe()


# In[47]:


happiness.describe()


# In[48]:


#Checking some information about the dataset
covid.info()


# In[49]:


#Checking some information about the dataset
happiness.info()


# In[50]:


# Checking if there any Duplicates 

if covid.duplicated().any() == False:
    print("There aren't any duplicates")
else:
    print('There are suplicates')
    
sns.heatmap(covid.isnull())
plt.show()


# In[51]:


#checking for na values
covid.isnull().sum()


# In[52]:


covid[covid['% of population fully vaccinated'].isnull()]


# In[53]:


#checking for na values
happiness.isnull().sum()


# In[54]:


# Checking if there any Duplicates
if happiness.duplicated().any() == False:
    print("There aren't any duplicates")
else:
    print('There are suplicates')

# Nulls check
sns.heatmap(happiness.isnull())
plt.show()


# * Checking for unique data and counting some data

# In[55]:


covid.nunique()


# In[56]:


happiness.nunique()


# # Data Cleaning
# 
# 
# 1- The are not any duplicted values.
# 
# 2- There are not any missing values.
# 
# 3- Dropping columns we do not need in our investigation.
# 
# 4- Dropping zero values.
# 
# 5- Sort values by Country column.
# 
# 6- Merging the Two datasets.
# 
# 7- Making Country column as index of our dataset.

# In[57]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.
# Checking for duplicated values after cleaning it
sum(covid.duplicated())


# In[58]:


sum(happiness.duplicated())


# ### since we will take country column as our key, we will take a glance at  the countries column to check if there are similarities between the two datasets, andalso we tends to merge our datasets and filtering the importantantfeatures that will need in our statistics and analysis

# In[59]:


# Sorting Values by country
print(covid.sort_values(by=["Country"])["Country"].values, '\n', len(covid.sort_values(by=["Country"])["Country"]))



# In[60]:


# Sorting values by the country name.
print(happiness.sort_values(by=['Country name'])['Country name'].values, '\n', len(happiness.sort_values(by=['Country name'])['Country name']))


# In[61]:


# Locating values
display(covid[covid['Country']=='U.K.'])
covid.loc[6,'Country'] = 'United Kingdom'


# In[62]:


# Merging the Tow Datasets and Sort values by the Country Column
df = covid.merge(happiness, left_on='Country', right_on='Country name', how='inner')
del df['Country name']

df = df.sort_values(by=['Country'])
df


# # Descriptive Analysis

# **Since we have 24 features, we will first start selecting the variables that we are going to use.
# The correlation matrix will help us find the the most relevant features.
# I will also select some variables that i find interesting for the analysis.
# Important to remark that the chosen criteria of the variables will be the relationship they have with the feature'% of population vaccinated', since our analysis is based on that.**

# In[63]:


fig, ax = plt.subplots(figsize=(15,15))
nums = list(df.select_dtypes('float64').columns)

sns.heatmap(df[nums].corr(), vmin=-1, vmax=1, cmap=sns.diverging_palette(20, 220, as_cmap=True), annot=True)

plt.show()


# In[64]:


# the chosen features are the wollowing
features = ['Country','Regional indicator', '% of population vaccinated', '% of population fully vaccinated', 
            'Ladder score', 'Logged GDP per capita', 'Social support', 'Healthy life expectancy']


# In[65]:


df = df[features]
df


# In[66]:


display(df.info())


#      * Categorization of our features.
#         Since we hude number of countries, and each one is unique.
#         So we can examine our categorical variables as we distributed to regions(categorical variable).

# In[67]:


df[list(df.select_dtypes('object').columns)]


# ## Exploratory Data Analysis
# 
# * Now, I am going to visualising my Data
# ### Research Question 1 (Exploring the number of the countries in each region?)

# In[68]:


fig = px.histogram(df, x='Regional indicator', template="plotly_white", color_discrete_sequence=["rgb(127,232,186)"]).update_xaxes(categoryorder="total descending")
fig.show()

print(df['Regional indicator'].value_counts())


# ### Research Question 2 (what are countries that are with the highest population of fully vaccinated?)

# In[69]:


# Visualization
figure = px.bar(df, y='% of population fully vaccinated', x='Country',
            title="Countries with Highest population fully vaccinated")
figure.show()


#  
#  **we can  see that the countries with the highest population vaccinated:**
# 
#   
#     1- chile.
#     2- Mauritania.
#     3- New Zealand.
#     4- Venezuela.
#    
#    
#  **for the lowest vaccinated:**
#  
#     1-Zambia.
#     2- Honduras.
#     3- Madagascar.
#     4- Serbia.

# ### Visualizing our variables

# In[70]:


numbers = list(df.select_dtypes('float64').columns)

df[numbers].hist(figsize=(20,10), color='#aaf0d1', edgecolor='white')

plt.show()

df[numbers].describe()


#    **We can see that from our visualzation:**
#    
#            1- The perscentage of the vaccinated are higher than fully vaccinated /    
#    **if we compared our chart of the twos;**
#       
#            1-  we find that about 10% of the countries are not fully vaccinated, and that means most of the countries
#                are still in the first phase of the vaccination.
#                

# ### Research Question 3 (What is the region with the highest  % of population vaccinated ?)

# In[71]:


fig, ax = plt.subplots(figsize=(15,5))

order=list(df.groupby('Regional indicator')['% of population vaccinated'].mean().sort_values(ascending=False).index)
sns.barplot(x='Regional indicator', y='% of population vaccinated', data=df, order=order, palette="Blues_d")

ax.tick_params(labelrotation=90)

plt.show()

print(df.groupby('Regional indicator')['% of population vaccinated'].mean().sort_values(ascending=False))


#  **We can from the graph: the two regions with highest vaccination**
#  
#       1- North America and ANZ.
#       2- Southeast Asia.
#  **The lowest region in vaccination**
#  
#      1- Sub-Saharan Africa.

# In[72]:


#Exploring some information about North America and ANZ
df[df['Regional indicator']=='North America and ANZ']


# ### Research Question 4 (What are countries with the Highest social support?)

# In[73]:


figure = px.bar(df, y='Social support', x='Country',
            title="Countries with Highest Social support")
figure.show()


# **We illustrate about the social support in the countries:**
# 
#   **The highest**
#   
#       1- Iceland.
#       2- Turkmenistan.
#       3- Kazakhistan.
#       
#   **The lowest**
#  
#        1- Afghnistan.
#        2- Burundi.
#        3- Benin.
#        
#        

# 

# ### Research Question 5 (What are regions with the Highest social support?)

# In[74]:


figure = px.bar(df, y='Social support', x='Regional indicator',
            title="Regions with Highest Social support")
figure.show()


# **From the chart we can see that the highest 3 regions with in social support :** 
#             
#             1- Sub-Saharan Africa
#             2- Western Europe
#             3- Latin America and Cribbbean

# ### Research Question 6 (What are regions with the Highest Healthy life expectancy?)

# In[75]:


figure = px.bar(df, y='Healthy life expectancy', x='Regional indicator',
            title="Regions with Highest Healthy life expectancy")
figure.show()


# **From the chart we can see that the highest 3 regions with in Healthy life expectancy :** 
#             
#             1- Sub-Saharan Africa
#             2- Western Europe
#             3- Latin America and Cribbbean

# ### Research Question 7 (What are the regions with the Highest population vaccinated and fully vaccinated?)

# In[76]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=df["Regional indicator"],
    y=df["% of population vaccinated"],
    name='Vaccinated',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=df["Regional indicator"],
    y=df["% of population fully vaccinated"],
    name='Fully Vaccinated',
    marker_color='lightsalmon'
))
fig.update_layout(barmode='group', xaxis_tickangle=-120)
fig.show()


# **From the chart we can see that the highest 3 regions with in vaccinated :** 
#             
#             1- Western Europe
#             2- Latin America and Cribbbean
#             3- Sub-Saharan Africa
#             
#             

# 

# ### Research Question 8 (Does vaccination depend on social support ?)

# In[77]:


# Filter for Western Europe
europe = df['Regional indicator'] == "Western Europe"
europe2 = df[europe]
europe2 = europe2.groupby('Regional indicator')[list(europe2.columns[2:-1])].mean().reset_index()

# Filter for Latin America and Caribbean
latin = df['Regional indicator'] == "Latin America and Caribbean"
latin2 = df[latin]
latin2 = latin2.groupby('Regional indicator')[list(latin2.columns[2:-1])].mean().reset_index()

# Filter for Uruguay
uruguay = df['Country'] == "Uruguay"
uruguay2 = df[uruguay].copy()
del uruguay2['Regional indicator']
uruguay2 = uruguay2.rename(columns={'Country':'Regional indicator'})

# Filter for Saudi Arabia
saudi = df['Country'] == "Saudi Arabia"
saudi2 = df[saudi].copy()
del saudi2['Regional indicator']
saudi2 = saudi2.rename(columns={'Country':'Regional indicator'})

# Concatenate the filtered dataframes
special = pd.concat([europe2, uruguay2, saudi2, latin2], axis=0)

# Create subplots
fig, ax = plt.subplots(3, 2, figsize=(15, 15))

# Plotting
sns.barplot(x='Regional indicator', y='% of population vaccinated', data=special, ax=ax[0, 0], palette="vlag")
sns.barplot(x='Regional indicator', y='% of population fully vaccinated', data=special, ax=ax[0, 1], palette="vlag")
sns.barplot(x='Regional indicator', y='Ladder score', data=special, ax=ax[1, 0], palette="vlag")
ax[1, 0].set_yscale("log")
sns.barplot(x='Regional indicator', y='Logged GDP per capita', data=special, ax=ax[1, 1], palette="vlag")
ax[1, 1].set_yscale("log")
sns.barplot(x='Regional indicator', y='Social support', data=special, ax=ax[2, 0], palette="vlag")
ax[2, 0].set_yscale("log")
sns.barplot(x='Regional indicator', y='Healthy life expectancy', data=special, ax=ax[2, 1], palette="vlag")
ax[2, 1].set_yscale("log")

# Show the plot
plt.tight_layout()
plt.show()

# Display the concatenated dataframe
display(special)


# **Our results show that ;**
#       
#       1- Uragay is the highest one in vaccination through the two phases. Also, it records the highest social support.
#          that takes us to the question " Is Uraguay ranked 1 because of the social support ?"
#          Answer: yes, it records 0.93 and ranked 1 in social support.
#       2- Western Europe ranked 2 in vaccination. Also, it ranked 2 after Uraguay in social support.
#       3- Saudi Arabia ranked 3  in vaccination. Also, it ranked 3 after Uraguay in social support.
#       4- There is something special about Uruguay & Sudia Arabia, They have healthy life expectancy.
#           since Sudia Arabia ranked 3 but only the difference of about 0.03 form Uruguay that ranked 1.
#     That is because of the importance they give to the social support that leads to an increase in the awareness 
#           towards the vaccination.

# ### Research Question 9 (what is the correlation between the GDP and Vaccination?)

# In[81]:


# Cluster
X = df['Logged GDP per capita'].values.reshape(-1, 1)

kmeans = KMeans(n_clusters=3, n_init=3, init="random", random_state=42)
kmeans.fit(X)

df.loc[:, 'GDP_Cluster'] = kmeans.labels_.astype(str)
df.loc[:, 'GDP_Cluster'] = df['GDP_Cluster'].map({'0': 'Low GDP', '1': 'High GDP', '2': 'Medium GDP'})

# Viz


fig = px.scatter(
    data_frame=df,
    x='Logged GDP per capita',
    y='% of population vaccinated',
    color='GDP_Cluster',
    template="plotly_white",
    hover_name='Country',
    hover_data=['Regional indicator', '% of population vaccinated', 'Logged GDP per capita', 'Social support']
)

fig.show()

gdp = df.groupby('GDP_Cluster')['% of population vaccinated'].mean().sort_values()
gdp = pd.DataFrame(gdp, columns=['average % of population vaccinated'])
display(gdp)


#   **We can see from the graph that:**
#                 
#                 1- We can see that GDP affect on the percentage of population vaccinated: 
#                             "the highest of the GDP, the highest of the people vaccinated"
#                 2- We can see that there is one high GDP country with low vaccinated population (Romania).
#                 3- We can see that medium GDPs countries with high vaccinated population (
#                                                                                            1- Vietnam.
#                                                                                             2- Peru.         )
#             And that is because poor level of cooperation between countries’ region .
#                        may also, because of the poor management.
# 
# 

# ### Research Question 10 (what is the effect of  social support & healthy life expectancy on % of population vaccinated ?)

# In[79]:


fig, ax = plt.subplots(figsize=(15,7))

sns.scatterplot(x='Healthy life expectancy', y='% of population vaccinated', data=df, 
                hue='Social support',palette="Blues_d", 
                size='Social support', sizes=(0.3, 300))

plt.show()


# **The tells us that:** 
# 
#         There is a positive relationship between % of populatio vaccinated and social support and heathy life expectancy.

# ### Research Question 11 (what is the vaccination influence on countries ?)

# In[80]:


fig, ax = plt.subplots(figsize=(15,7))
sns.regplot(x="Ladder score", y="% of population vaccinated", data=df)

plt.show()


# **The tells us that:** 
#        
#        while the ladder score is increasing, population vaccinated increase. the more satisfication.

# ## The Results
#   
#      We notice that
#        1- countries especially in the same region, not in the same phase of vaccination.
#        2- Social support has an effect on the vaccination rate in regions and countries in one region.
#        3- the more ladder score the more satisfied region.
#        4- The higher GPD, the highest vaccinated one.
#        5 The highest region in vaccination tells us that this region has good relationships and communication between  its                 countries. And also, has a justified distribution of its resources between its countries. 
#        

# ## Our Recommendation
#       
#       For a better life for the populations and more advanced countries & regions:
#          1- strengthen the relationships and cooperation between countries, especially in the same region.
#          2- Justified distribution support between the countries.
#          3- Achieving equality as much as they can.
#          
#         As a result of that, it will outcomes a highly vaccinated population  &  healthy communities. 
