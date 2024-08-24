import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#DataSet

#Retrive dataset

df1 = pd.read_csv('/home/hulkim/Documents/FInal Project/Harta_Rega/houses.csv')


df1.info()

# Strip 'RM' and convert to numeric, remove ',' between numbers.
df1['Price'] = df1['Price'].str.lstrip('RM').str.replace(',', '')


#Rename the column
df1.rename(columns={
    'Price': 'Price (RM)',
    'Property Type': 'Type',
    }, inplace=True)


# Fill NaN values in 'Price' with 0 (or any other suitable value)
df1['Price (RM)'] = df1['Price (RM)'].fillna(0)
df1['Price (RM)'] = df1['Price (RM)'].astype(int)

# Split the 'Rooms' column
df1[['Bedrooms', 'Store']] = df1['Rooms'].str.split('+', expand=True)

# Drop the original 'Rooms' column
df1.drop('Rooms', axis=1, inplace=True)

df1['Size'] = df1['Size'].str.rstrip('sq.ft. ')  # Remove 'sq.ft' or 'sq.' or space from the right
df1['Size'] = df1['Size'].str.lstrip('Built-up:') # Remove 'Built-up:' from the left
df1['Size'] = df1['Size'].str.lstrip('Land area:') # Remove 'Land area:' from the left

df1['Size'] = df1['Size'].str.replace(',', '')  # Remove commas
df1['Store'] = df1['Store'].str.replace(' ', '')  # Remove whitespace

"""### Cleaning"""

df1['Size'] = df1['Size'].str.replace(' ', '')
df1['Size'] = df1['Size'].str.replace("'", "")
df1['Size'] = df1['Size'].str.lower()

df1['Bedrooms'] = df1['Bedrooms'].replace('Studio', '1')
df1['Bedrooms'] = df1['Bedrooms'].replace('20 Above', '20')
df1['Store'] = df1['Store'].replace('', '0')

replacements = {
    '25x75': 1875, '20x75': 1500, '16x55': 880, '38x75': 2850, '22x83': 1826,
    '24x75': 1800, '22x75': 1650, '33x85': 2805, '24x80': 1920, '27x70': 1890,
    '24x85': 2040, '22x89': 1958, '65x90': 5850, '22x62': 1364, '24x55': 1320,
    '10+24x80': 1930, '646sf~1001': 646, '850sf~1000': 850, '850-1000': 850,
    '285038x75': 2850, '23x75': 1725, '24x801920': 1920, '40+30X80': 2440,
    '20x95': 1900, '22x95': 2090, '48x85': 4080, '45x80': 3600, '20x80': 1600,
    '22x80': 1760, '32x75': 2400, '58x80': 4640, '20x65': 1300, '22x85': 1870,
    '26x80': 2080, '24x70': 1760, '31x70': 2080, '45x82': 3690, '20x60': 1200,
    '40x100': 4000, '24x93': 2232, '22x100': 2200, '40x80': 3200, '22x70': 1540,
    '39x85': 3315, '16x60': 960
}

# Replace values in the 'Size' column
df1['Size'] = df1['Size'].replace(replacements)

#Delete certain row with specific value.
df1 = df1.drop(df1[df1['Size'] == '22&#8217;x100&#8217;'].index)
df1 = df1.drop(df1[df1['Size'] == '10+24x80'].index)
df1 = df1.drop(df1[df1['Size'] == '285038x75'].index)
df1 = df1.drop(df1[df1['Size'] == '24x801920'].index)
df1 = df1.drop(df1[df1['Size'] == '40+30x80'].index)
df1 = df1.drop(df1[df1['Size'] == '38x752850'].index)
df1 = df1.drop(df1[df1['Size'] == 'kualalumpur'].index)
df1 = df1.drop(df1[df1['Size'] == '~'].index)

df1 = df1.drop(df1[df1['Price (RM)'] == 0].index)

# Replace 'x' with '*' in the 'Size' column if it exists
if 'Size' in df1.columns:
    df1['Size'] = df1['Size'].str.replace('x', '*')
else:
    print("Warning: 'Size' column not found in the DataFrame.")

df1.info()

#Clear value pattern '38*803040' in column size
# import re

# def has_multiplication_pattern(value):
#     if isinstance(value, str):
#         return bool(re.search(r'\d+\*\d+', value))  # Check if the value matches the pattern
#     return False

# # Apply the function to each cell in the DataFrame
# masks = [df1[col].map(has_multiplication_pattern) for col in df1.columns]

# # Combine the masks to find rows with the pattern in any column
# final_mask = masks[0]
# for mask in masks[1:]:
#     final_mask |= mask

# # Get the indices of rows with the multiplication pattern in any column
# rows_with_pattern = df1[final_mask].index

# print(rows_with_pattern)

import re

def has_multiplication_pattern(value):
    if isinstance(value, str):
        return bool(re.search(r'\d+\*\d+\d+', value))
    return False

# Apply the function to each cell in the DataFrame
mask = df1['Size'].map(has_multiplication_pattern)

# Filter the DataFrame to keep only rows WITHOUT the pattern
df1 = df1[~mask]


df1.info()

#Function to make multiplication the value with asterisk
def multiply_asterisk_values(value):
    if isinstance(value, str) and '*' in value:
        try:
            num1, num2 = value.split('*')
            return int(num1) * int(num2)
        except ValueError:
            return value  # Handle cases where the split doesn't result in numbers
    else:
        return value

df1['Size'] = df1['Size'].apply(multiply_asterisk_values)

# Ensure 'Store' is numeric before filling NaNs
df1['Store'] = pd.to_numeric(df1['Store'], errors='coerce')  # Convert to numeric, non-convertibles become NaN

# Now fill NaNs in selected column with 0
df1[['Bedrooms', 'Car Parks', 'Bathrooms', 'Size', 'Store']] = df1[['Bedrooms', 'Car Parks', 'Bathrooms', 'Size', 'Store']].fillna(0)

#Change type of column
df1 = df1.astype({'Bedrooms': 'int', 'Bathrooms': 'int','Car Parks': 'int', 'Store': 'int'})

# Ensure 'Size' is numeric before encoding
df1['Size'] = pd.to_numeric(df1['Size'], errors='coerce')

#Label Encoder
le = LabelEncoder()
df1 ['Furnishing'].unique()

df1['Size_Encoded'] = le.fit_transform(df1['Size'])
df1['Furnishing'] = le.fit_transform(df1['Furnishing'])
df1['Type'] = le.fit_transform(df1['Type'])
df1['Location'] = le.fit_transform(df1['Location'])

#Drop rows with value NaN after Encoded.
df1 = df1.drop(df1[df1['Size_Encoded'] == 0].index)

df1.info()

# # Apply the filter to each cell in the DataFrame
# for col in df1.columns:
#     if 'Size' in df1.columns:  # Check if 'Size' column exists
#         df1 = df1[df1['Size'].map(lambda x: '&#' not in str(x))]
#     else:
#         print("Column 'Size' not found in DataFrame.")
#         break  # Exit the loop if 'Size' column is not found

# # Iterate through all columns in the DataFrame
# for column in df1.columns:
# # Filter only the "Size" column
#   if df1['Size'].dtype == 'object':
#   # Filter out rows where the "size" column value contains a "+"
#      df1 = df1[~df1['Size'].str.contains('\+', na=False)]
#   # Filter out rows where the "size" column value contains a "~"
#      df1 = df1[~df1['Size'].str.contains('~', na=False)]
#   # Filter out rows where the "size" column value contains a "-"
#      df1 = df1[~df1['Size'].str.contains('-', na=False)]
#   # Filter out rows where the "size" column value contains a "."
#      df1 = df1[~df1['Size'].str.contains('.', na=False)]

#Remove rows with missing 'Size' value
df1 = df1.dropna(subset=['Size'])
#Try to convert 'Size' to integer
df1 = df1.astype({'Size': 'int'})

#Check if there any value is NaN in DataFrame
df1.isna().any()

duplicate_rows = df1.duplicated()
print('Number of duplicate rows:', duplicate_rows.sum())

"""### Searching some data"""

#This code to find anything in dataset.
df1[df1.isin(['38*80'])].stack()

df1['Size'].isna().sum()

#Display selected row index with all column included.
pd.set_option('display.max_rows', None)
print(df1.iloc[780:790])

#Explore Data

df1.info()

df1.describe()


#Saving the cleaned dataset.
df1.to_csv('cleaned_houses.csv', index=False)