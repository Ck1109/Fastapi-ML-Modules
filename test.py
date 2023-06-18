import pandas as pd
import pandas_profiling

data=pd.read_json(r'C:\Users\kulka\PycharmProjects\API_Python\Data\sample (4).json')
print(data.isna().sum())