from sklearn import preprocessing
import numpy as np

def OneHot(X, num):
    out = np.zeros((X.shape[0], num))
    for i in range(X.shape[0]):
        out[i, X[i]] = 1
    return out

# enc = preprocessing.OneHotEncoder()

# Y  =np.array(range(5)).reshape(-1,1)

# print(Y)

# enc.fit(Y)

# re = enc.transform([[3]]).toarray()

# print(re)

X = np.array(range(5)).reshape(-1,1)
print(OneHot(X, 8))