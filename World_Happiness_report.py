#!/usr/bin/env python
# coding: utf-8

# # Introduction

# The World Happiness Report is a landmark survey of the state of global happiness. The first report was published in 2012, the second in 2013, the third in 2015, and the fourth in the 2016 Update. The World Happiness 2017, which ranks 155 countries by their happiness levels, was released at the United Nations at an event celebrating International Day of Happiness on March 20th. The report continues to gain global recognition as governments, organizations and civil society increasingly use happiness indicators to inform their policy-making decisions. Leading experts across fields – economics, psychology, survey analysis, national statistics, health, public policy and more – describe how measurements of well-being can be used effectively to assess the progress of nations. The reports review the state of happiness in the world today and show how the new science of happiness explains personal and national variations in happiness.

# # Content

# The happiness scores and rankings use data from the Gallup World Poll. The scores are based on answers to the main life evaluation question asked in the poll. This question, known as the Cantril ladder, asks respondents to think of a ladder with the best possible life for them being a 10 and the worst possible life being a 0 and to rate their own current lives on that scale. The scores are from nationally representative samples for the years 2013-2016 and use the Gallup weights to make the estimates representative. The columns following the happiness score estimate the extent to which each of six factors – economic production, social support, life expectancy, freedom, absence of corruption, and generosity – contribute to making life evaluations higher in each country than they are in Dystopia, a hypothetical country that has values equal to the world’s lowest national averages for each of the six factors. They have no impact on the total score reported for each country, but they do explain why some countries rank higher than others.

# # Inspiration

# What countries or regions rank the highest in overall happiness and each of the six factors contributing to happiness? How did country ranks or scores change between the 2015 and 2016 as well as the 2016 and 2017 reports? Did any country experience a significant increase or decrease in happiness?

# # What is Dystopia?

# Dystopia is an imaginary country that has the world’s least-happy people. The purpose in establishing Dystopia is to have a benchmark against which all countries can be favorably compared (no country performs more poorly than Dystopia) in terms of each of the six key variables, thus allowing each sub-bar to be of positive width. The lowest scores observed for the six key variables, therefore, characterize Dystopia. Since life would be very unpleasant in a country with the world’s lowest incomes, lowest life expectancy, lowest generosity, most corruption, least freedom and least social support, it is referred to as “Dystopia,” in contrast to Utopia.

# # What is residual?

# The residuals, or unexplained components, differ for each country, reflecting the extent to which the six variables either over- or under-explain average 2014-2016 life evaluations. These residuals have an average value of approximately zero over the whole set of countries. Figure 2.2 shows the average residual for each country when the equation in Table 2.1 is applied to average 2014- 2016 data for the six variables in that country. We combine these residuals with the estimate for life evaluations in Dystopia so that the combined bar will always have positive values. As can be seen in Figure 2.2, although some life evaluation residuals are quite large, occasionally exceeding one point on the scale from 0 to 10, they are always much smaller than the calculated value in Dystopia, where the average life is rated at 1.85 on the 0 to 10 scale.

# # Table of Contents


A. Importing, cleaning and numerical summaries
B. Indexing and grouping
C. Bar plot of the Happiness Score 
E. Pairwise Scatter plots
F. Correlation
G. Probabilities
H. Matrices
# # A. Importing, cleaning and numerical summaries
Download the from happiness_score_dataset.csv the Resources tab.
Import the data as a pandas DataFrame.
Check the number of observations.
Obtain the column headings.
Check the data type for each column.
Check if there are any missing values.
If necessary remove any observations to ensure that there are no missing values and the values in each column are of the same data type.
Obtain the mean, minimum and maximum value for each column containing numerical data.
List the 10 happiest countries.
List the 10 least happy countries.
# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


WHR = pd.read_csv('happiness_score_dataset.csv')


# In[3]:


WHR.head(10)


# In[4]:


WHR.shape


# In[5]:


print("There are {:,} rows ".format(WHR.shape[0]) + "and {} columns in our data".format(WHR.shape[1]))


# In[6]:


WHR.set_index('Country', inplace=True)


# In[7]:


WHR.info()


# Data types is float64(9),int64(1),object(1)

# In[8]:


WHR.isnull().sum()


# There is no null value

# In[9]:


NULLS = WHR[WHR.isnull().any(axis=1)]


# In[10]:


NULLS.head()


# In[11]:



WHR.duplicated().sum()


# In[12]:


WHR.describe()


# In[13]:


WHR.sort_values(by="Happiness Rank", ascending=True).head(10)


# In[14]:


WHR.sort_values(by="Happiness Rank", ascending=False).head(10)


# # Now we start INdexing and grouping

# In[15]:


WHR_Region = WHR.groupby('Region')


# In[16]:


WHR_Region['Happiness Score'].describe().sort_values(by="mean",ascending=True).head(10)


# In[17]:


WHR[WHR['Region']=='Sub-Saharan Africa'].head()


# In[18]:


WHR_Region['Happiness Score'].describe().sort_values(by="mean",ascending=False).head(10)


# In[19]:


WHR_A = WHR[WHR['Region'] == 'Australia and New Zealand']
WHR_WE = WHR[WHR['Region'] == 'North America']
WHR_EE = WHR[WHR['Region'] == 'Western Europe']
WHR_LA = WHR[WHR['Region'] == 'Latin America and Caribbean']
WHR_AP = WHR[WHR['Region'] == 'Eastern Asia']
WHR_NA = WHR[WHR['Region'] == 'Middle East and Northern']
             


# In[20]:


len(WHR_A[WHR_A['Happiness Score'] > 6])


# In[21]:


print("There are {} countries in Africa that have a happiness score above 6.0 ".format(len(WHR_A[WHR_A['Happiness Score'] > 6])))


# In[22]:



len(WHR_WE[WHR_WE['Happiness Score'] > 6])


# In[23]:


print("There are {} countries in Western Europe that have a happiness score above 6.0 ".format(len(WHR_WE[WHR_WE['Happiness Score'] > 6])))


# In[24]:


len(WHR_EE[WHR_EE['Happiness Score'] > 6])


# In[25]:


print("There is {} country in Eastern Europe that has a happiness score above 6.0 ".format(len(WHR_EE[WHR_EE['Happiness Score'] > 6])))


# In[26]:


len(WHR_AP[WHR_AP['Happiness Score'] > 6])


# In[27]:


print("There are {} countries in the Asia Pacific that have a happiness score above 6.0 ".format(len(WHR_AP[WHR_AP['Happiness Score'] > 6])))


# In[28]:


len(WHR_LA[WHR_LA['Happiness Score'] > 6])


# In[29]:


print("There are {} countries in the Latin America that have a happiness score above 6.0 ".format(len(WHR_LA[WHR_LA['Happiness Score'] > 6])))


# In[30]:


len(WHR_NA[WHR_NA['Happiness Score'] > 6])


# In[31]:


print("There are {} countries in the North America that have a happiness score above 6.0 ".format(len(WHR_NA[WHR_NA['Happiness Score'] > 6])))


# In[32]:


Delta_NA = WHR_NA.max(axis=0)['Happiness Score'] - WHR_NA.min(axis=0)['Happiness Score']
print(Delta_NA)


# In[33]:


Delta_EE = WHR_EE.max(axis=0)['Happiness Score'] - WHR_EE.min(axis=0)['Happiness Score']
print(Delta_EE)


# In[34]:


Delta_WE = WHR_WE.max(axis=0)['Happiness Score'] - WHR_WE.min(axis=0)['Happiness Score']
print(Delta_WE)


# In[35]:


Delta_A = WHR_A.max(axis=0)['Happiness Score'] - WHR_A.min(axis=0)['Happiness Score']
print(Delta_A)


# In[36]:


Delta_LA = WHR_LA.max(axis=0)['Happiness Score'] - WHR_LA.min(axis=0)['Happiness Score']
print(Delta_LA)


# In[37]:


Delta_AP = WHR_AP.max(axis=0)['Happiness Score'] - WHR_AP.min(axis=0)['Happiness Score']
print(Delta_AP)


# In[38]:


Deltas = {}


# In[39]:


Deltas["Australia and New Zealand"] = Delta_NA
Deltas["North America"] = Delta_EE
Deltas["Western Europe"] = Delta_WE
Deltas["Latin America and Caribbean"] = Delta_A
Deltas["Eastern Asia"] = Delta_LA
Deltas["Middle East and Northern Africa"] = Delta_AP


# In[40]:


print("The {} region seems to have the largest range of happiness scores".format(max(Deltas, key=Deltas.get)))


# # Bar plot of the Happiness Score

# In[41]:


WHR['Happiness Score'].head(10).plot(xticks=np.arange(9), kind='barh', figsize= (10, 10))
plt.xlabel("Happiness Score")
plt.title('Happiness Score of the top 10 Countries')


# In[42]:


WHR[['Economy (GDP per Capita)', 'Family','Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)', 'Dystopia Residual']].head(10).plot(kind='barh',
                                                                xticks=np.arange(9), stacked=True, figsize= (10, 10))

plt.xlabel("Happiness Score")
plt.title('Happiness Score of the top 10 Countries')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[43]:


WHR_A[['Economy (GDP per Capita)', 'Family','Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)', 'Dystopia Residual']].head(10).plot(kind='barh',
                                                                xticks=np.arange(9), stacked=True, figsize= (10, 10))

plt.xlabel("Happiness Score")
plt.title('Happiness Score of the top 10 Countries')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# # Pairwise Scatter plots

# Obtain scatter plots of the Happiness Score versus each of the other variables. Your plots should be displayed as multiple plots table and obtained with one command as supposed to separate commands for each plot.

# In[44]:


sns.pairplot(data=WHR, kind='reg', size = 5,
                  x_vars=['Happiness Score'],
                  y_vars=['Economy (GDP per Capita)', 'Family','Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)', 'Dystopia Residual'])


# In[45]:


sns.pairplot(data=WHR, size = 5, hue='Region',
                  x_vars=['Happiness Score'],
                  y_vars=['Economy (GDP per Capita)', 'Family','Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)', 'Dystopia Residual'])


# # Correlation

# Obtain the correlation between the Happiness Score and each of the other variables. Which variable has the highest correlation with the Happiness Score?

# In[46]:


WHR.corr(method="pearson", min_periods=20)["Happiness Score"].sort_values(ascending=False)


# In[47]:


WHR.corr(method="pearson", min_periods=20)["Happiness Score"].abs().sort_values(ascending=False)


# If we ignore the Happiness Rank, Job Satisfaction seems to have the highest correlation with the Happiness Score.

# In[48]:


WHR.corr(method="pearson", min_periods=20)


# In[49]:


corr = WHR.corr(method = "pearson")

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)


# # Probabilities

# Compute the probability that randomly selected country with Happiness score over 6.0 is from Western Europe. You will have to use pandas to count the appropriate quantities.

# In[50]:


WHR[WHR['Happiness Score'] > 6].shape[0]


# In[51]:


WHR[(WHR['Happiness Score'] > 6) & (WHR['Region'] == 'Western Europe')].shape[0]


# In[52]:


float(len(WHR[(WHR['Happiness Score'] > 6) & (WHR['Region'] == 'Western Europe')]))/float(len(WHR[WHR['Happiness Score'] > 6]))


# In[53]:


print("The probability that a randomly selected country with happiness score over 6.0 is form Western Europe is {0:.0%}".format(float(WHR[(WHR['Happiness Score'] > 6) & (WHR['Region'] == 'Western Europe')].shape[0]

)/float(WHR[WHR['Happiness Score'] > 6].shape[0])))


# In[54]:


WHR.shape


# # Matrices

# In[55]:


Western_Europe = []
North_America = []


# In[56]:


for x in WHR['Region']:
    if x == 'Western Europe':
         Western_Europe.append(1)
    else: Western_Europe.append(0)


# In[57]:


for x in WHR['Region']:
    if x == 'North America':
         North_America.append(1)
    else: North_America.append(0)


# In[58]:


Matrix = pd.DataFrame(index=WHR.index)


# In[59]:


Matrix['Western Europe'] = Western_Europe
Matrix['North America'] = North_America


# In[60]:


Matrix.head(20)


# # Conclusion

# Happiness matters. Its primary value is found not within the subjective enjoyment of the feeling itself but in the enthusiasm for creativity, production and problem-solving that it instills in people. We have seen that the national economy, family life, health quality, general freedom, generosity of citizens and trust in government are positively correlated with national happiness levels in varying degrees. GDP, health and family in particular show the strongest correlations with happiness; when a person is healthy, has tight personal relationships and is not perpetually burdened by finanacial concerns, then his or her views on life will likely be more optimistic. The need for all three of these factors to be ranked highly to ensure a high probabilty of happiness can be relaxed to a need for just two of them to be ranked highly to ensure a high probability of happiness. Prosperous countries with great health care, prosperous countries with great familial relationships, and countries with great health care and great familial relationships can expect the average citizen to report a high quality of life. It would therefore be wise that nations make it a priority to provide its citizens with the opportunities and support required to develop these crucial life pillars, and in return for their efforts, to reap the benefits on a national level.
