import numpy as np 
from sklearn import svm 

#amostras e classes
X = np.array([[-1, -1],[-2, -1],[1, 1],[2, 1]])
Y = np.array([1, 1, 2, 2])

clf= svm.SVC(gamma='auto')

clf.fit(X, Y)

print (clf.support_vectors_)
print (clf.n_support_)

V1 = [-2, -2]
V2 = [2, 2]

V = [V1, V2]

#print (clf.predict(V1])) não é mais preciso
#print (clf.predict([V2])) não é mais preciso

print (clf.predict(V))