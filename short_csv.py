import pandas as pd
import csv


# make a shorter CSV

cs61 = pd.read_csv("cs61a_files.csv")
cs61_short = cs61.head(100).drop('vector', axis=1)
cs61_short.to_csv("cs61a_short.csv", index=False)