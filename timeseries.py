#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
import copy
import numpy as np
import os
import json
import pandas as pd
from PIL import Image, ExifTags


# In[38]:


working_folder = r"C:/Users/nuway/OneDrive/Desktop/Realsense Project/Mushroom Pictures/2nd Round/"
# working_folder = "PATH TO FOLDER WHERE THE THREE CSVs AND THE TIMELAPSE IMAGES ARE SAVED"
environmental_files = os.listdir(working_folder)
environmental_files = [x for x in environmental_files if x.endswith("csv")]
environmental_files


# In[39]:


# Open temperature csv and transform string format of temperature to float
temperature = pd.read_csv(working_folder + environmental_files[2], skiprows=1)
temperature["temperature1"] = temperature["temperature1"].apply(lambda x: float(x[:-3]) if isinstance(x, str) else x)
temperature["temperature2"] = temperature["temperature2"].apply(lambda x: float(x[:-3]) if isinstance(x, str) else x)
# Average the temperatures to get rid of Null values existing in each column
temperature["average_temperature"] = temperature[["temperature1","temperature2"]].mean(axis=1)
# deal with the different columns formats of csv depending on the experiment
try:    
    temperature = temperature.drop(["temperature1","temperature2","Harvested Mass for ","entity_id"],axis=1)
except:
    temperature = temperature.drop(["temperature1","temperature2"],axis=1)
temperature


# In[40]:


# print(temperature["average_temperature"].mean())
# print(temperature["average_temperature"].std())


# In[41]:


humidity = pd.read_csv(working_folder + environmental_files[1],skiprows=1)
humidity["relativeHumidity1"] = humidity["relativeHumidity1"].apply(lambda x: float(x[:-1]) if isinstance(x, str) else x)
humidity["relativeHumidity2"] = humidity["relativeHumidity2"].apply(lambda x: float(x[:-1]) if isinstance(x, str) else x)
humidity["average_relativeHumidity"] = humidity[["relativeHumidity1","relativeHumidity2"]].mean(axis=1)
try:    
    humidity = humidity.drop(["relativeHumidity1","relativeHumidity2","Harvested Mass for ","entity_id"],axis=1)
except:
    humidity = humidity.drop(["relativeHumidity1","relativeHumidity2"],axis=1)
humidity


# In[42]:


co2 = pd.read_csv(working_folder + environmental_files[0],skiprows=1)
co2
co2["co2_1"] = co2["co2_1"].apply(lambda x: float(x[:-4]) if isinstance(x, str) else x)
co2["co2_2"] = co2["co2_2"].apply(lambda x: float(x[:-4]) if isinstance(x, str) else x)
co2["average_co2"] = co2[["co2_1","co2_2"]].mean(axis=1)
try:    
    co2 = co2.drop(["co2_1","co2_2","Harvested Mass for ","entity_id"],axis=1)
except:
    co2 = co2.drop(["co2_1","co2_2"],axis=1)
co2


# In[43]:

print(type(temperature["Time"][0]))

# Transform csv date to pandas datetime
temperature["Time"] = pd.to_datetime(temperature["Time"])#,format = '%d/%m/%Y %H:%M:%S')
humidity["Time"] = pd.to_datetime(humidity["Time"])#,format = '%d/%m/%Y %H:%M:%S')
co2["Time"] = pd.to_datetime(co2["Time"])#,format = '%d/%m/%Y %H:%M:%S')

# Find if a measurement is missing in any of the CSVs. Example: in experiment 1 CO2 has one less measurement than temperature and humidity
try:
    missing_dates = [np.setdiff1d(temperature["Time"], co2["Time"])[0]]
except:
    pass
try:
    missing_dates += [np.setdiff1d(temperature["Time"], humidity["Time"])[0]]
except:
    pass
try:    
    missing_dates += [np.setdiff1d(humidity["Time"], co2["Time"])[0]]
except:
    pass
print(np.unique(missing_dates))

# Delete the extra values from the other variables based on the date that has the problem.
# In our case, a couple of such values have a problem so deletion is selected. If numerous values were missing another apporach should have been used.
for value_to_delete in np.unique(missing_dates):
    print(value_to_delete)
    if temperature["Time"].index[temperature["Time"] == value_to_delete].tolist():
        print("temp")
        temperature = temperature[temperature["Time"] != value_to_delete]
        temperature.reset_index(drop=True,inplace=True)
    if humidity["Time"].index[humidity["Time"] == value_to_delete].tolist():
        print("hum")
        humidity = humidity[humidity["Time"] != value_to_delete]
        humidity.reset_index(drop=True,inplace=True)
    if co2["Time"].index[co2["Time"] == value_to_delete].tolist():
        print("co2")
        co2 = co2[co2["Time"] != value_to_delete]
        co2.reset_index(drop=True,inplace=True)
    print("-------------")



# In[44]:


plt.plot(temperature["average_temperature"])
#plt.plot(humidity["average_relativeHumidity"])
#plt.plot(co2["average_co2"])


# In[48]:


# Get the image files
image_files = os.listdir(working_folder)
image_files = [x for x in image_files if x.endswith("JPG")]
image_files


# In[60]:


# Create a dataframe with the image filename and the date that it was collected (extracted from the image metadata)
images_df = pd.DataFrame(columns=["Time","filename"])
for image in image_files:
    img = Image.open(working_folder + image)
    exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
    # In experiment 3 a problem with the recorded year of the images happened and all have 2016. This makes the first value of the environmentals as the closest existing date to every iamge.
    images_df.loc[len(images_df.index)] = [pd.to_datetime(exif["DateTime"].replace('2016', '2022'),format = '%Y:%m:%d %H:%M:%S'), image] 

# experiment 2 has images starting from ~7000 then at 9999 overflows and then goes from 1 until ~2000. So sorting based on the date is mandatory for the correct order, instead of sorting on filenames
images_df = images_df.sort_values(by='Time').reset_index(drop=True)
images_df['Time'] = images_df['Time'].replace('2016', '2022', regex=True)
images_df


# In[61]:


# Create a combined dataset of image filenames and environmental data, based on the closest existing date.

# Function to find the closest datetime in another DataFrame along with its data
def find_closest_datetime(dt, other_df,data_column_name):
    time_diff = np.abs(other_df['Time'] - dt)
    closest_index = time_diff.idxmin()
    closest_datetime = other_df.loc[closest_index, 'Time']
    closest_data = other_df.loc[closest_index, data_column_name]
    return closest_datetime, closest_data

# Apply the function to each row in df1
result_list = [find_closest_datetime(row['Time'], temperature, data_column_name="average_temperature") for _, row in images_df.iterrows()]
result_df = pd.DataFrame(result_list, columns=['closest_Time', 'closest_average_temperature'])
# Concatenate the result with df1
df1 = pd.concat([images_df, result_df], axis=1)

# Apply the function to each row in df1
result_list = [find_closest_datetime(row['Time'], humidity, data_column_name="average_relativeHumidity") for _, row in df1.iterrows()]
result_df = pd.DataFrame(result_list, columns=['closest_Time', 'closest_average_relativeHumidity'])
result_df = result_df.drop(['closest_Time'], axis=1)
# Concatenate the result with df1
df1 = pd.concat([df1, result_df], axis=1)

# Apply the function to each row in df1
result_list = [find_closest_datetime(row['Time'], co2, data_column_name="average_co2") for _, row in df1.iterrows()]
result_df = pd.DataFrame(result_list, columns=['closest_Time', 'closest_average_co2'])
result_df = result_df.drop(['closest_Time'], axis=1)
# Concatenate the result with df1
df1 = pd.concat([df1, result_df], axis=1)
# Display the result
df1.to_csv(working_folder + 'timeseries_and_files_combined.csv', index=False)
df1



# ## Experimenting for combining the produced dataset with Sari's tracking algo result dataframe

# In[36]:


# "cluster_0_pixel_count","cluster_0_mushroom_count",..,"cluster_1_pixel_count","cluster_1_mushroom_count",..,"cluster_2_pixel_count","cluster_2_mushroom_count",..
clusters_df = pd.read_csv("../sari_code/results/image_data.csv",index_col=False)
clusters_df = clusters_df.rename(columns={"Timestamp": "Time"})
clusters_df["Time"] = pd.to_datetime(clusters_df["Time"],format = '%Y:%m:%d %H:%M:%S')
clusters_df


# In[37]:


final_df = pd.merge(df1, clusters_df, on='Time', how='outer')
final_df = final_df.drop(columns=[""])
final_df


# ## Experimenting with different apporaches of combining the two measurements of environmental variables instead of averaging to avoid discontinuity problem.

# In[4]:


temperature = pd.read_csv(working_folder + environmental_files[2],skiprows=1)
temperature["temperature1"] = temperature["temperature1"].apply(lambda x: float(x[:-3]) if isinstance(x, str) else x)
temperature["temperature2"] = temperature["temperature2"].apply(lambda x: float(x[:-3]) if isinstance(x, str) else x)
print(temperature)
print(temperature["temperature1"].mean())
print(temperature["temperature1"].std())
print(temperature["temperature2"].mean())
print(temperature["temperature2"].std())
print(temperature["temperature1"].mean() - temperature["temperature1"].std()/5)
print(temperature["temperature2"].mean() + temperature["temperature2"].std()/5)
print((temperature["temperature1"].mean() - temperature["temperature2"].mean())/2)

temperature["average_temperature"] = temperature[["temperature1","temperature2"]].mean(axis=1)
temperature = temperature.drop(["temperature1","temperature2"],axis=1)
print(temperature)
plt.figure()
plt.plot(temperature["average_temperature"])


temperature = pd.read_csv(working_folder + environmental_files[2],skiprows=1)
temperature["temperature1"] = temperature["temperature1"].apply(lambda x: float(x[:-3]) if isinstance(x, str) else x)
temperature["temperature2"] = temperature["temperature2"].apply(lambda x: float(x[:-3]) if isinstance(x, str) else x)
temperature["temperature1"] = temperature["temperature1"].apply(lambda x: x - (temperature["temperature1"].mean() - temperature["temperature2"].mean())/2)
temperature["temperature2"] = temperature["temperature2"].apply(lambda x: x + (temperature["temperature1"].mean() - temperature["temperature2"].mean())/2)
temperature["average_temperature"] = temperature[["temperature1","temperature2"]].mean(axis=1)
temperature = temperature.drop(["temperature1","temperature2"],axis=1)
print(temperature)
plt.figure()
plt.plot(temperature["average_temperature"])


print("-----------------------------------------------------------------------------")
humidity = pd.read_csv(working_folder + environmental_files[1],skiprows=1)
humidity["relativeHumidity1"] = humidity["relativeHumidity1"].apply(lambda x: float(x[:-1]) if isinstance(x, str) else x)
humidity["relativeHumidity2"] = humidity["relativeHumidity2"].apply(lambda x: float(x[:-1]) if isinstance(x, str) else x)
print(humidity)
print(humidity["relativeHumidity1"].mean())
print(humidity["relativeHumidity1"].std())
print(humidity["relativeHumidity2"].mean())
print(humidity["relativeHumidity2"].std())
print(humidity["relativeHumidity1"].mean() + humidity["relativeHumidity1"].std()/3)
print(humidity["relativeHumidity2"].mean() - humidity["relativeHumidity2"].std()/3)

print((humidity["relativeHumidity2"].mean() - humidity["relativeHumidity1"].mean())/2)

print("-----------------------------------------------------------------------------")
co2 = pd.read_csv(working_folder + environmental_files[0],skiprows=1)
co2["co2_1"] = co2["co2_1"].apply(lambda x: float(x[:-4]) if isinstance(x, str) else x)
co2["co2_2"] = co2["co2_2"].apply(lambda x: float(x[:-4]) if isinstance(x, str) else x)
print(co2)
print(co2["co2_1"].mean())
print(co2["co2_1"].std())
print(co2["co2_2"].mean())
print(co2["co2_2"].std())
print(co2["co2_1"].mean() + co2["co2_1"].std()/3)
print(co2["co2_2"].mean() - co2["co2_2"].std()/3)

print((co2["co2_1"].mean() - co2["co2_2"].mean())/2)


# In[ ]:


tup = (40,10)
plt.figure(figsize=tup)
plt.plot(humidity["relativeHumidity1"])
plt.figure(figsize=tup)
plt.plot(humidity["relativeHumidity2"])
plt.figure(figsize=tup)
plt.plot(humidity[["relativeHumidity1","relativeHumidity2"]].mean(axis=1))


# In[12]:


temperature = pd.read_csv(working_folder + environmental_files[2],skiprows=1)
temperature["temperature1"] = temperature["temperature1"].apply(lambda x: float(x[:-3]) if isinstance(x, str) else x)
temperature["temperature2"] = temperature["temperature2"].apply(lambda x: float(x[:-3]) if isinstance(x, str) else x)
# temperature["temperature1"] = temperature["temperature1"].apply(lambda x: x - (temperature["temperature1"].mean() - temperature["temperature2"].mean())/2)
# temperature["temperature2"] = temperature["temperature2"].apply(lambda x: x + (temperature["temperature1"].mean() - temperature["temperature2"].mean())/2)
t = temperature["temperature1"] - temperature["temperature2"]
print(t.mean())
print(t.std())
print(temperature)
plt.figure()
plt.plot(t)


# In[ ]:





# In[ ]:





# In[ ]:




