import numpy as np
import pandas as pd
import os
import kagglehub

path = kagglehub.dataset_download('playokdata/edtech-online-instructor-dataset')
file = os.listdir(path)[0]
df = pd.read_csv(os.path.join(path, file))
print('shape', df.shape)
print(df.dtypes)
column = 'completion_rate'
try:
    X = df.drop(column, axis=1).values.astype(np.float32)
    y = df[column].values.astype(np.float32).reshape(-1, 1)
    print('X shape', X.shape, 'y shape', y.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
