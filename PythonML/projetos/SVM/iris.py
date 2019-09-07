import numpy as np
from sklearn import svm, datasets, metrics

#iris = datasets.load_iris()

#iris = datasets.load_digits()

iris = datasets.load_breast_cancer()

X = iris.data #atributos
Y = iris.target # classes

#print(Y)
#print(X)
np.random.seed(0)
n_amostras = len(X)
ordem = np.random.permutation(n_amostras)

porcentagem = 0.7 # quantos por cento quero da base para treino e quantos para teste

X = X[ordem] # passando vetor com indices embaralhados
Y = Y[ordem]

X_treino = X[:int(porcentagem * n_amostras)] # pegando amostras do começo do vetor ate 70% dos valores Base de treino
Y_treino = Y[:int (porcentagem * n_amostras)]

X_teste = X[int(porcentagem * n_amostras):] # pegando amostras até os ultimos 30% sobrando Base de teste
Y_teste = Y[int(porcentagem * n_amostras):] 

#clf = svm.SVC(gamma='auto')
clf = svm.SVC(gamma='scale')
clf.fit(X_treino, Y_treino) # passa os treinos

print(clf.support_vectors_)
print(clf.n_support_)

predicao = clf.predict(X_teste)


print(predicao)

taxa_acerto = clf.score(X_teste, Y_teste)
print(taxa_acerto)

matriz = metrics.confusion_matrix(Y_teste, predicao)

for item in matriz:
    print(item)