import pandas as pd

# Load the xlsx file into a pandas dataframe
df = pd.read_excel('/data/Lanhong/Pancreas_radiomics/radiomics_dataset.xlsx')

# Convert the dataframe to a csv file
df.to_csv('/data/Lanhong/Pancreas_radiomics/radiomics_dataset.csv', index=False)