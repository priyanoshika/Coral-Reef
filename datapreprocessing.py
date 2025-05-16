# Importing the necessary libraries
import pandas as pd

# Accessing the dataset

df = pd.read_csv('NOAA_Reef_Check__Bleaching_Data .csv')
df.head()
df.info()
# Replacing object in Bleaching to int
df['Bleaching'] = df['Bleaching'].str.replace('No','0')
df['Bleaching'] = df['Bleaching'].str.replace('Yes','1')
df['Bleaching'] = df['Bleaching'].astype('int')
# Replacing object in Storms to int
df['Storms'] = df['Storms'].str.replace('no','0')
df['Storms'] = df['Storms'].str.replace('yes','1')
df['Storms'] = df['Storms'].astype('int')
# Replacing object in HumanImpact to int
df['HumanImpact'] = df['HumanImpact'].str.replace('high','3')
df['HumanImpact'] = df['HumanImpact'].str.replace('moderate','2')
df['HumanImpact'] = df['HumanImpact'].str.replace('low','1')
df['HumanImpact'] = df['HumanImpact'].str.replace('none','0')
df['HumanImpact'] = df['HumanImpact'].astype('int')
# Replacing object in Siltation to int
df['Siltation'] = df['Siltation'].str.replace('always','3')
df['Siltation'] = df['Siltation'].str.replace('often','2')
df['Siltation'] = df['Siltation'].str.replace('occasionally','1')
df['Siltation'] = df['Siltation'].str.replace('never','0')
df['Siltation'] = df['Siltation'].astype('int')
# Replacing object in Dynamite to int
df['Dynamite'] = df['Dynamite'].str.replace('high','3')
df['Dynamite'] = df['Dynamite'].str.replace('moderate','2')
df['Dynamite'] = df['Dynamite'].str.replace('low','1')
df['Dynamite'] = df['Dynamite'].str.replace('none','0')
df['Dynamite'] = df['Dynamite'].astype('int')
# Replacing object in Poison to int
df['Poison'] = df['Poison'].str.replace('high','3')
df['Poison'] = df['Poison'].str.replace('moderate','2')
df['Poison'] = df['Poison'].str.replace('low','1')
df['Poison'] = df['Poison'].str.replace('none','0')
df['Poison'] = df['Poison'].astype('int')
# Replacing object in Sewage to int
df['Sewage'] = df['Sewage'].str.replace('high','3')
df['Sewage'] = df['Sewage'].str.replace('moderate','2')
df['Sewage'] = df['Sewage'].str.replace('low','1')
df['Sewage'] = df['Sewage'].str.replace('none','0')
df['Sewage'] = df['Sewage'].astype('int')
# Replacing object in Industrial to int
df['Industrial'] = df['Industrial'].str.replace('high','3')
df['Industrial'] = df['Industrial'].str.replace('moderate','2')
df['Industrial'] = df['Industrial'].str.replace('low','1')
df['Industrial'] = df['Industrial'].str.replace('none','0')
df['Industrial'] = df['Industrial'].astype('int')
# Replacing object in Commercial to int
df['Commercial'] = df['Commercial'].str.replace('high', '3')
df['Commercial'] = df['Commercial'].str.replace('moderate', '2')
df['Commercial'] = df['Commercial'].str.replace('low', '1')
df['Commercial'] = df['Commercial'].str.replace('none', '0')
df['Commercial'] = df['Commercial'].astype('int')
# Checking that the datatypes of the variables have been changed
df.info()
# Exporting to CSV
df.to_csv('NOAA_int.csv',index=False)