#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv(r"US_Accidents_March23.csv")


# In[ ]:


df.head(5)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df.describe().T


# In[ ]:


df.describe(include=object)


# In[ ]:


df.isna().sum()


# In[ ]:


df.drop(columns=['End_Lat','End_Lng','Wind_Chill(F)','Precipitation(in)'],axis=1,inplace=True)


# In[ ]:


#storing categorical column names to a new variable
categorical=[i for i in df.columns if df[i].dtype=='O']
#for categorical values we can replace the null values with the Mode of it
for i in categorical:
    df[i].fillna(df[i].mode()[0],inplace=True)


# In[ ]:


df.drop(columns=['Wind_Speed(mph)', 'Visibility(mi)', 'Pressure(in)', 'Humidity(%)', 'Temperature(F)'], axis=1, inplace=True)


# In[ ]:


df.isna().sum()


# In[ ]:


df.duplicated().sum()


# In[ ]:


city_acc = df['City'].value_counts().sort_values(ascending = False).reset_index()


# In[ ]:


city_acc


# In[ ]:


plt.figure(figsize=(15,6))
sns.barplot(x='City',y='count',data= city_acc.head(10),palette='viridis')
plt.title("Top 10 cities with most number of accidents")
plt.ylabel("No of accidents")
plt.show()


# In[ ]:


plt.figure(figsize=(5,6))
sns.barplot(x='City',y='count',data= city_acc.tail(10),palette='viridis')
plt.title("Bottom 10 cities with least number of accidents")
plt.ylabel("No of accidents")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


state_acc = df['State'].value_counts().sort_values(ascending = False).reset_index()
state_acc


# In[ ]:


## States with most number of accidents
plt.figure(figsize=(15,6))
sns.barplot(x='State',y='count',data= state_acc.head(5),palette='rocket')
plt.title("Top 5 States with most number of accidents")
plt.ylabel("No of accidents")
plt.show()

## States with the least number of accidents
plt.figure(figsize=(15,6))
sns.barplot(x='State',y='count',data= state_acc.tail(5),palette='rocket')
plt.title("Bottom 5 States with least number of accidents")
plt.ylabel("No of accidents")
plt.show()


# `Observations:`
# - Miami is the city with most number of accidents
# - starjunction, stomsburg are among the cities with least number of accidents
# - California is the state with highest cases of accident
# - South Dakota(SD) is the state with least cases of accident

# In[ ]:


df['Severity'].value_counts().index


# In[ ]:


plt.figure(figsize=(8,8))
plt.pie(df['Severity'].value_counts(),labels=['severe','more-severe','most-severe','least-severe'],autopct="%1.2f%%")
plt.show()


# In[ ]:


# Convert the 'Start_Time' column to datetime format
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')

# Extract the year from the 'Start_Time' column and store it in a new column called 'Year'
df['Year'] = df['Start_Time'].dt.year

# Display the first few rows to verify
print(df[['Start_Time', 'Year']].head())


# In[ ]:


df['Year'].value_counts()


# ## Years with most number of accidents

# In[ ]:


year_count = df['Year'].value_counts().reset_index()
sns.barplot(x='Year',y='count',data=year_count,palette='rocket')
plt.title("Years with most number of accidents")
plt.show()


# `Observation:`
# - year 2021 had highest accident rates.

# ## Number of accidents at different time zones

# In[ ]:


df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')

# Extract the time and store it in a new column 'Time'
df['Time'] = df['Start_Time'].dt.time

# Function to categorize time
def categorize_time(time):
    if pd.isna(time):
        return 'Unknown'
    if time >= pd.to_datetime('05:00:00').time() and time < pd.to_datetime('12:00:00').time():
        return 'Morning'
    elif time >= pd.to_datetime('12:00:00').time() and time < pd.to_datetime('17:00:00').time():
        return 'Afternoon'
    elif time >= pd.to_datetime('17:00:00').time() and time < pd.to_datetime('21:00:00').time():
        return 'Evening'
    else:
        return 'Night'

# Apply the function to create a new column 'Time_Zone'
df['Time_Zone'] = df['Time'].apply(categorize_time)

# Filter out 'Unknown' time zones for plotting
filtered_df = df[df['Time_Zone'] != 'Unknown']


# Count the number of accidents in each timezone
time_zone_counts = filtered_df['Time_Zone'].value_counts()

# Plot the distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=time_zone_counts.index, y=time_zone_counts.values, palette='rocket')
plt.xlabel('Time of Day')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents by Time of Day')
plt.show()


# ## histogram to show the distribution of accidents through the time zones

# In[ ]:


plt.figure(figsize=(10, 6))
sns.histplot(filtered_df['Time_Zone'], bins=4, kde=False, color='green')
plt.xlabel('Time Zone')
plt.ylabel('Number of Accidents')
plt.title('Distribution of Accidents by Time Zone')
plt.show()


# ## Weather conditions at the time of accidents

# In[ ]:


weather = df['Weather_Condition'].value_counts().sort_values(ascending=False).reset_index()
sns.barplot(x='count',y='Weather_Condition',data=weather[:10],orient='horizontal',palette='viridis')
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
sns.scatterplot(x=df['Start_Lng'],y=df['Start_Lat'],hue=df['State'])
plt.legend(loc="lower right")
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
sns.scatterplot(x=df['Start_Lng'],y=df['Start_Lat'],hue=df['Severity'])
plt.show()


# # `Conclusion:`
# 
# In this traffic data analysis project, we undertook a comprehensive examination of the accident records, including data cleaning, exploratory data analysis (EDA), and visualization. Here are the key findings and observations from the analysis:
# 
# 1. **City-wise Accident Distribution**:
#    - **Miami** stands out as the city with the highest number of accidents. This could be attributed to its high population density and traffic volume.
#    - On the other hand, **Star Junction** and **Stromsburg** are among the cities with the least number of accidents, possibly due to their smaller size and lower traffic.
# 
# 2. **State-wise Accident Distribution**:
#    - **California** reports the highest number of accident cases, which aligns with its large population and extensive road network.
#    - Conversely, **South Dakota (SD)** has the least number of accident cases, likely due to its lower population density and traffic volume.
# 
# 3. **Yearly Accident Rates**:
#    - The year **2021** recorded the highest accident rates, indicating a potential increase in traffic or changes in reporting during this period.
# 
# 4. **Accident Severity**:
#    - The severity of accidents is categorized into four levels: 1, 2, 3, and 4.
#    - Most people experienced accidents at **severity level 2**, suggesting that while many accidents occurred, they were not extremely severe.
# 
# 5. **Weather Conditions During Accidents**:
#    - **Fair weather** conditions were prevalent during most of the accidents, indicating that factors other than adverse weather, such as driver behavior or traffic congestion, may play a significant role in accident occurrences.
# 
# 6. **Time of Day for Accidents**:
#    - Most accidents occurred in the **morning**, potentially due to rush hour traffic when more people are commuting to work or school.
# 
# These observations provide valuable insights into traffic patterns and accident hotspots, which can inform policy decisions, urban planning, and safety measures aimed at reducing the incidence and severity of accidents.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




