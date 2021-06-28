import pandas as pd
import numpy as np
from numpy import array
from pickle import dump


X_Train_3emb = pd.read_pickle("lis_X_all_Train.pkl")

print(len(X_Train_3emb))

print(type(X_Train_3emb))

print(type(X_Train_3emb))

print(type(X_Train_3emb[0]))


#print(type(X_Train_3emb[0][0]))


#print(X_Train_3emb[0].shape)

#print(X_Train_3emb[2][0].shape)

#print(X_Train_3emb[2][1].shape)

#print(X_Train_3emb[2][2].shape)
#'''

X_Test_3embed = pd.read_pickle("X_Train_3embeding.pkl")

print(type(X_Test_3embed))
print(len(X_Test_3embed))
print(type(X_Test_3embed[67]))
print(X_Test_3embed[0].shape)

#print(type(X_Test_3embed[0][0]))

#print(type(X_Test_3embed[0][1]))

#print(type(X_Test_3embed[0][2]))

#print(X_Test_3embed[0][0].shape)
#print(X_Test_3embed[0][0])

#print(X_Test_3embed[0][2].shape)


print(X_Test_3embed[1].shape)
print(X_Test_3embed[2].shape)
print(X_Test_3embed[3].shape)
print(X_Test_3embed[6].shape)
print(X_Test_3embed[30].shape)



XX_Test_3embeding = list()

for i in X_Test_3embed:
    result = i[:, 0, :]
    print(result.shape)
    #c=i.astype(object)
    #result=np.squeeze(result)
    #print(result.shape)
    XX_Test_3embeding.append(result)

dump(XX_Test_3embeding, open('XX_Train_3embeding.pkl', 'wb'))


print(type(XX_Test_3embeding))
print(type(XX_Test_3embeding[0]))
#print(type(XX_Test_3embeding[1][2]))
print(XX_Test_3embeding[103].shape)
print(len(XX_Test_3embeding[103]))



'''
#w=np.array(XX_Test_3embeding)


print(type(X_Test_3embed))
print(X_Test_3embed[3].shape)
print(X_Test_3embed[2][0].shape)
print(X_Test_3embed[2][0][0].shape)

narrayConvrt=array(X_Test_3embed)

narrayConvrt=np.array(X_Test_3embed)

print(len(narrayConvrt[67]))
print(type(narrayConvrt[67]))

print(type(narrayConvrt))
print(narrayConvrt[67].shape)
print(narrayConvrt[2][0].shape)
print(narrayConvrt[2][0][0].shape)



#print(X_Test_3embed[0][0][0].shape)

#print(X_Test_3embed[0].shape)

#print(X_Test_3embed[0][1].shape)

#print(X_Test_3embed[0][2].shape)

#print(X_Test_3embed[0][2].shape)


#print(X_Test_3embed[0].shape)
#print(type(X_Test_3embed[0]))


XX_Test_3embeding = list()

for i in X_Test_3embed:
    b=np.array(i)
    XX_Test_3embeding.append(b)

print(XX_Test_3embeding[0].shape)
print(type(XX_Test_3embeding[0]))

'''