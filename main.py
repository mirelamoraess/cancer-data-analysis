# Importação das bibliotecas
from sklearn.datasets import load_breast_cancer # Funções para carregar datasets
from sklearn import svm  # Módulo SVM (Support Vector Machine) para classificação e regressão
from sklearn.model_selection import train_test_split  # Função para dividir dados em treino e teste
import matplotlib.pyplot as plt  # Biblioteca para plotagem de gráficos


# Carregando os datasets
dataset_cancer = load_breast_cancer()

# Imprimindo nomes das características e alvos do dataset de câncer
print(dataset_cancer.feature_names)
print(dataset_cancer.target_names)


# Separando os dados em conjuntos de treino e teste
X_train_can, X_test_can, y_train_can, y_test_can = train_test_split(
   dataset_cancer.data, dataset_cancer.target, stratify=dataset_cancer.target, random_state=42)
# train_test_split divide o dataset em dados de treino e teste,
# stratify mantém a mesma proporção de classes nos dados de treino e teste,
# random_state garante a reprodutibilidade da divisão

# Lista para armazenar as métricas de acurácia
training_accuracy = []
test_accuracy = []

# Lista de kernels para SVM
kernels = ['linear', 'rbf', 'sigmoid']

# Loop sobre os diferentes kernels
for kernel in kernels:
   svm_model = svm.SVC(kernel=kernel)  # Cria um modelo SVM com o kernel atual
   svm_model.fit(X_train_can, y_train_can)  # Treina o modelo SVM com os dados de treino
   training_accuracy.append(svm_model.score(X_train_can, y_train_can))  # Calcula a acurácia no treino e armazena
   test_accuracy.append(svm_model.score(X_test_can, y_test_can))  # Calcula a acurácia no teste e armazena
   
   
# Plotagem dos resultados
plt.plot(kernels, training_accuracy, label='Acurácia no conj. treino')  # Plota acurácia de treino
plt.plot(kernels, test_accuracy, label='Acurácia no conj. teste')  # Plota acurácia de teste
plt.ylabel('Acurácia')  # Rótulo do eixo y
plt.xlabel('Kernels')  # Rótulo do eixo x
plt.legend()  # Adiciona legenda
plt.show()  # Exibe o gráfico