from sklearn.datasets import load_breast_cancer  
from sklearn import svm 
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt 

dataset_cancer = load_breast_cancer()

print(dataset_cancer.feature_names)
print(dataset_cancer.target_names)

X_treino_can, X_teste_can, y_treino_can, y_teste_can = train_test_split(
   dataset_cancer.data, dataset_cancer.target, stratify=dataset_cancer.target, random_state=42)

precisao_treino = []
precisao_teste = []


kernels = ['linear', 'rbf', 'sigmoid']

for kernel in kernels:
   modelo_svm = svm.SVC(kernel=kernel)  
   modelo_svm.fit(X_treino_can, y_treino_can)
   precisao_treino.append(modelo_svm.score(X_treino_can, y_treino_can)) 
   precisao_teste.append(modelo_svm.score(X_teste_can, y_teste_can)) 
   
# Geração dos resultados
plt.plot(kernels, precisao_treino, label='Precisão no conj. treino')  
plt.plot(kernels, precisao_teste, label='Precisão no conj. teste')  
plt.ylabel('Precisão')
plt.xlabel('Kernels') 
plt.legend()  
plt.show()  
