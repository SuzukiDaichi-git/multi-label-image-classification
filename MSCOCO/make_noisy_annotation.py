import pandas as pd
import numpy as np
import os

clean_label_path = "./0_0trainAnnotation.csv"
clean_label = pd.read_csv(clean_label_path,header=None)

THRESs = [[0,20],[20,0]]

for THRES in THRESs:
    noisy_label = clean_label.copy(deep=True)
    num_instances = [0] * 81
    for i in range(clean_label.shape[0]):
        for j in range(1, clean_label.shape[1]):
            if clean_label.iat[i,j]==1:
                num_instances[j] += 1
                if np.random.randint(0, 100) < THRES[0]:
                    noisy_label.iat[i,j]=0

    for i in range(clean_label.shape[0]):
        for j in range(1, clean_label.shape[1]):
            if clean_label.iat[i,j]==0:
                if np.random.rand() * 100 < float(THRES[1]) * float(num_instances[j]) / float(clean_label.shape[0] - num_instances[j]):
                    noisy_label.iat[i,j]=1


    OUT_FILENAME = f'./annotation{THRES[0]}_{THRES[1]}trainAnnotation.csv'

    noisy_label.to_csv(OUT_FILENAME, header=False, index=False)

    print(OUT_FILENAME)