# Importação das bibliotecas
from sklearn.datasets import load_breast_cancer  
from sklearn import svm 
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt 

# Carregando os datasets
dataset_cancer = load_breast_cancer()

# Imprimindo nomes das características e alvos do dataset de câncer
print(dataset_cancer.feature_names)
print(dataset_cancer.target_names)

# Separando os dados em conjuntos de treino e teste
X_treino_can, X_teste_can, y_treino_can, y_teste_can = train_test_split(
   dataset_cancer.data, dataset_cancer.target, stratify=dataset_cancer.target, random_state=42)
# train_test_split divide o dataset em dados de treino e teste,
# stratify mantém a mesma proporção de classes nos dados de treino e teste,
# random_state garante a reprodutibilidade da divisão

# Lista para armazenar as métricas de precisão
precisao_treino = []
precisao_teste = []

# Lista de kernels para SVM
kernels = ['linear', 'rbf', 'sigmoid']

# Loop sobre os diferentes kernels
for kernel in kernels:
   modelo_svm = svm.SVC(kernel=kernel)  # Cria um modelo SVM com o kernel atual
   modelo_svm.fit(X_treino_can, y_treino_can)  # Treina o modelo SVM com os dados de treino
   precisao_treino.append(modelo_svm.score(X_treino_can, y_treino_can))  # Calcula a precisão no treino e armazena
   precisao_teste.append(modelo_svm.score(X_teste_can, y_teste_can))  # Calcula a precisão no teste e armazena
   
# Geração dos resultados
plt.plot(kernels, precisao_treino, label='Precisão no conj. treino')  # Gera gráfico de precisão de treino
plt.plot(kernels, precisao_teste, label='Precisão no conj. teste')  # Gera gráfico de precisão de teste
plt.ylabel('Precisão')
plt.xlabel('Kernels') 
plt.legend()  
plt.show()  