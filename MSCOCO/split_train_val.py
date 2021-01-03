import pandas as pd
import numpy as np
import os
np.random.seed(437)


val_array = []
num_data = 0

for i, noise in enumerate([[0,20],[20,0]]):
    
    clean_label = pd.read_csv("./{0}_{1}trainAnnotation.csv".format(noise[0], noise[1]),header=None)
    if i == 0:
        num_data = len(clean_label)
        train_array = [j for j in range(num_data)]
        val_array = np.random.choice(train_array, size=int(num_data * 0.1), replace=False)

    val_datas = pd.DataFrame([])
    train_datas = pd.DataFrame([])

    for j in range(num_data):
        if j in val_array:
            val_datas = val_datas.append(clean_label.iloc[j,:])
        else:
            train_datas = train_datas.append(clean_label.iloc[j,:])


    val_datas.iloc[:,1:] = val_datas.select_dtypes(include=float).astype(int)
    print(len(val_datas))
    print(len(train_datas))
    
    val_datas.to_csv('./annotation/{0}_{1}_valAnnotation.csv'.format(noise[0], noise[1]), header=False, index=False)
    train_datas.to_csv('./annotation/{0}_{1}_trainAnnotation.csv'.format(noise[0], noise[1]), header=False, index=False)