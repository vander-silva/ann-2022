# region pip installs
# > pip install tensorflow
# > pip install sklearn
# > pip install mlxtend
# > pip install numpy
#
# endregion
# region Imports
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

from aula_ann_2022_bib import *

# endregion
### ---------------------------------------------------------------- ###
###                         Aula ANN - 2022
### Vanderlei A. Silva  -  https://www.linkedin.com/in/vanderlei-silva/
###        Fontes web:
###        https://www.circuitbasics.com/neural-networks-in-python-ann/
###        https://keras.io/api/
###        http://yann.lecun.com/exdb/mnist/
### ---------------------------------------------------------------- ###
### Agenda:
### ---------------------------------------------------------------- ###
###   [1] ANÁLISE DE DADOS                                           ###
### ---------------------------------------------------------------- ###
# region Bloco Análise de Dados
###                             https://keras.io/api/datasets/mnist/
### Objetivos:
###       i) [Carregar,  Visualizar] base original mnist do keras
###      ii) [Verificar Equalização] das classes
###     iii) [Equalizar,Distribuir,Embaralhar] dataset Tre, Val, Tes
###
### Notação: [Entradas] xin  : inputs  da ann
###          [Alvos   ] ytar : targets para treinamento da ann
###          [Saidas  ] yout : outputs da ann
### ---------------------------------------------------------------- ###
### i) [Carregar,  Visualizar]
# region code
# carregar dataset mnist
mnist = keras.datasets.mnist
(xin_tre_orig, ytar_tre_orig), (xin_tes_orig, ytar_tes_orig) = mnist.load_data()
print('-->Tre   input:',xin_tre_orig.shape,', target:',ytar_tre_orig.shape,'\n' +
      '-->Tes   input:',xin_tes_orig.shape,', target:',ytar_tes_orig.shape,'')
# visualizer imagens do dataset
plotar = False
if plotar:
    fig, (ax1,ax2) = plt.subplots(1,2)
    for i in range(1):
        ax1.cla();    ax1.imshow(xin_tre_orig[i])
        ax2.cla();    ax2.imshow(xin_tes_orig[i])
        ax1.set_title("Dataset Tre,    Frame {},    Alvo = {}".format(i,ytar_tre_orig[i]))
        ax2.set_title("Dataset Tes,    Frame {},    Alvo = {}".format(i,ytar_tes_orig[i]))
        # Note that using time.sleep does *not* work here!
        plt.pause(1)
# visualizer imagens do dataset alvos e sua aleatoriedade
plotar = False
if plotar:
    fig, (ax1,ax2) = plt.subplots(2,1)
    ax1.stem(ytar_tre_orig)
    ax2.stem(ytar_tes_orig)
# endregion
### ii) [Verificar Equalização]
# region code
# Definir objeto de análise de dados
o_mnist_data_analysis = MnistDataSetAnalysis()
#
o_mnist_data_analysis.set_dataset_tre_original(xin_tre_orig, ytar_tre_orig)
o_mnist_data_analysis.set_dataset_tes_original(xin_tes_orig, ytar_tes_orig)
#
# Verificar equalização entre classes
plotar = False
o_mnist_data_analysis.show_class_balance_orig(plotar)
# endregion
### iii) [Equalizar,Distribuir,Embaralhar]
# region -> Equalizar,Distribuir,Embaralhar as amostras e verificar o resultado
o_mnist_data_analysis.set_class_balance(plotar)
o_mnist_data_analysis.show_class_balance(plotar)
#
# Verificar datasets finais, distribuídos e equilibrados:
(xin_tre, ytar_tre) = o_mnist_data_analysis.get_dataset_tre(normalizar=255)
(xin_val, ytar_val) = o_mnist_data_analysis.get_dataset_val(normalizar=255)
(xin_tes, ytar_tes) = o_mnist_data_analysis.get_dataset_tes(normalizar=255)
#
# Visualizar aleatoriedade de datasets alvos:
plotar = False
o_mnist_data_analysis.verificar_maior_repeticao(ytar_tre, 'Dataset Tre: verif. aleatoriedade')
o_mnist_data_analysis.plot_stem_dataset_target(ytar_tre,  'Dataset Tre final: Targets',plotar)
#
o_mnist_data_analysis.verificar_maior_repeticao(ytar_val, 'Dataset Val: verif. aleatoriedade')
o_mnist_data_analysis.plot_stem_dataset_target(ytar_val,  'Dataset Val final: Targets',plotar)
#
o_mnist_data_analysis.verificar_maior_repeticao(ytar_tes, 'Dataset Tes: verif. aleatoriedade')
o_mnist_data_analysis.plot_stem_dataset_target(ytar_tes,  'Dataset Tes final: Targets',plotar)
# endregion
# region -> Visualizar imagens dos Datasets criados
plotar = False
if plotar:
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    for i in range(100):
        ax1.cla();    ax1.imshow(xin_tre[i])
        ax2.cla();    ax2.imshow(xin_val[i])
        ax3.cla();    ax3.imshow(xin_tes[i])
        ax1.set_title("Dataset Tre,    Frame {},    Alvo = {}".format(i,ytar_tre[i]))
        ax2.set_title("Dataset Val,    Frame {},    Alvo = {}".format(i,ytar_val[i]))
        ax3.set_title("Dataset Tes,    Frame {},    Alvo = {}".format(i,ytar_tes[i]))
        # Note that using time.sleep does *not* work here!
        plt.pause(1)
# endregion
# endregion
### ---------------------------------------------------------------- ###
###   [2] MACHINE LEARNING                                           ###
### ---------------------------------------------------------------- ###
# region Bloco Machine Learning
###    i) model:           crie       o modelo
###   ii) model.compile:   configure  o modelo
###  iii) model.fit:       treine     o modelo
###
# i) Model Creation
# region -> Criar e configurar o modelo
#
# Activation functions: https://keras.io/api/layers/activations/
#                       https://www.tensorflow.org/api_docs/python/tf/keras/activations
#                       https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
#       relu    -> rectified linear unit activation function; poslin no Matlab - positiva linear
#       tanh    -> tangente hiperbólica; tansig no Matlab
#       softmax -> softmax no Matlab
#
# Crie o modelo
model = keras.models.Sequential()
#
# Crie a camada de entrada
model.add(keras.layers.Flatten(input_shape=[28,28]))
#
# Camada(s) oculta(s)
model.add(keras.layers.Dense(50, activation="relu"))
model.add(keras.layers.Dense(50, activation="relu"))
#
# Camada de saida
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()
# endregion
# ii) model.compile
# region -> Model.compile
#
# Loss functions:  https://keras.io/api/losses/
#       *** São as funções para medir o erro durante o treinamento, o qual deve ser minimizado pelo optimizer
#       binary_crossentropy             -> classificaçõo binária (duas classes)
#       categorical_crossentropy        -> classificação com 2 ou + classes em modo one_hot
#                                          https://en.wikipedia.org/wiki/One-hot
#       sparse_categorical_crossentropy -> classificação com 2 ou + classes cujos alvos são inteiros
#       mean_squared_error              -> mse: erro médio quadrático
#
# Optimizers:   https://keras.io/api/optimizers/
#       *** São os algoritmos de aprendizado para minimizar o erro durante o treinamento
#       SGD  -> Gradient Descent Optimizer
#       Adam -> Stochastic gradient descent (que implementa o algoritmo Adam)
#               ***baixo cutosto computacional, adequado p/ probl. que requerem grande qtdade dados e parametros
#
# Metrics:   https://keras.io/api/metrics/
#       *** São funções utilizadas para avaliar o desempenho do seu modelo, pós treinamento.
#           As funções de métrica são semelhantes às funções de perda, exceto que os resultados
#           da avaliação de uma métrica não são usados ao treinar o modelo.
#           Observe que você pode usar qualquer função de perda como métrica.
#       accuracy                    -> Calcula a frequência com que as previsões são iguais aos rótulos|alvos.
#       binary_accuracy             -> accuracy pora rótulos binários.
#       sparse_categorical_accuracy -> Calcula a frequência com que as previsões correspondem a rótulos inteiros.
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["sparse_categorical_accuracy"])
# endregion
# iii) model.fit
# region -> Model.fit
early_stopping = EarlyStopping(monitor='val_loss', patience=6)
checkpointer = ModelCheckpoint(filepath="ann_weights_2022.hdf5", verbose=1, save_best_only=True)
#     ------------- Tabela para rede 50x50x10 ---------------
#     batch sizes:   [    32,    604,   1007,  15098,  45290]
#     epoch sizes:   [    32,     50,     70,    400,    900]
#     -------------------------------------------------------
#     blocks         [  1416,     75,     45,      3,      1]
#     total erros:   [   256,    267,    263,    311,    331]
#     desempenho(%): [ 97.13,  97.01,  97.05,  96.51,  96.29]
#     epoch stop:    [    12,     36,     41,    231,    549]
model_history = model.fit(xin_tre,
                          ytar_tre,
                          epochs=32,
                          batch_size=32,
                          validation_data=(xin_val,ytar_val),
                          callbacks=[early_stopping, checkpointer])
model.load_weights('ann_weights_2022.hdf5')
#
# plote a acurácia em treinamento do modelo: curvas treinamento e validação
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,5))
ax1.plot(model_history.history['loss'])
ax1.plot(model_history.history['val_loss'])
ax2.plot(model_history.history['sparse_categorical_accuracy'])
ax2.plot(model_history.history['val_sparse_categorical_accuracy'])

ax1.set_yscale('log')
ax1.grid(axis='y',which='both', color='lightgray', linestyle='--', linewidth=0.5)
ax1.set_title('Aula ANN 2022    -    Acurácia do Modelo')
ax1.set_ylabel('Acurácia')
ax1.set_xlabel('Época')
ax1.legend(['Trainamento', 'Validação'], loc='upper left')

ax2.grid(axis='y',which='both', color='lightgray', linestyle='--', linewidth=0.5)
ax2.set_title('Aula ANN 2022    -    Acurácia Por Classe (Rótulo Inteiro)')
ax2.set_ylim( [ 0.9, 1.01 ] )
ax2.set_yticks((0.9, 0.91, 0.92, 0.93,  0.94,  0.95, 0.96,  0.97, 0.98,  0.99, 1.0))
ax2.set_yticklabels(("0.9", "", "0.92", "",  "0.94",  "", "0.96",  "", "0.98",  "", "1.0"))
ax2.set_ylabel('Categorical Accuracy')
ax2.set_xlabel('Época')
ax2.legend(['Trainamento', 'Validação'], loc='upper left')

plt.show()
# region Desempenho percentual validação
print('Melhor época de validação (desde a época 0): [val_sparse_categorical_accuracy]')
best_epoch = len(model_history.history['val_sparse_categorical_accuracy']) - 6 -1 # para patience = 6
print('*** Melhor Época: {:>2d},   {:>6.2f}% (validação)'.format(best_epoch,model_history.history['val_sparse_categorical_accuracy'][best_epoch]*100))
# endregion
# endregion
# endregion
### ---------------------------------------------------------------- ###
###   [3] ANÁLISE DE RESULTADOS                                      ###
### ---------------------------------------------------------------- ###
# region Bloco Análise de Resultados do Modelo
# Saída da ANN
# region -> Saída da ANN
yout_tes__10_elementos = model.predict(xin_tes)  # Valores prováveis
yout_tes = []                                    # Rotulos de saída ANN
for item in range(len(yout_tes__10_elementos)):
    yout_tes.append(np.argmax(yout_tes__10_elementos[item]))
# comparação inicial
print(list(ytar_tes[:30]))
print(yout_tes[:30])
# endregion
# Erros de classe dataset TES
# region -> Erros de Classe
erros_de_classe = []
indice_de_classes_que_a_rede_errou = []
for item in range(10):
    erros_de_classe.append(0)
for item in range(len(ytar_tes)):
    if ytar_tes[item] != yout_tes[item]:
        erros_de_classe[ytar_tes[item]] += 1
        indice_de_classes_que_a_rede_errou.append(item)
total_erros_de_classe = sum(erros_de_classe)
for classe in range(10):
    print('Classe {:d}: {:d} erros'.format(classe, erros_de_classe[classe]))
# endregion
# Desempenho percentual do modelo ANN
# region -> Desempenho percentual testes
percentual_acertos_classe = 100 * ( 1-(total_erros_de_classe / len(ytar_tes) ) )
print('Total erros classe: {:d}'.format(total_erros_de_classe))
print('*** Percentual de Acerto do Modelo: {:>5.2f}% (teste),'.format(percentual_acertos_classe))
# endregion
# Matriz de Confusão
# region -> Matriz de Confusão
conf_matrix_sklearn = confusion_matrix(ytar_tes, yout_tes)
print(conf_matrix_sklearn)
class_names = ['classe 0', 'classe 1', 'classe 2', 'classe 3', 'classe 4',
               'classe 5', 'classe 6', 'classe 7', 'classe 8', 'classe 9']
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix_sklearn,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                class_names=class_names,
                                figsize=(12,8)
                                )
ax.set_title('Matriz de Confusão')
ax.set(xlabel='Valores Preditos (saída da ANN)', ylabel='Valores Verdadeiros (alvos)')
plt.show()
# endregion
# Visualizando erros de dataset TES
# region -> Visualizar imagens que a rede errou
#
# Apenas imprime a lista de classes que a rede errou
print('Lista de Índices de Classes Erradas:')
list_len = len(indice_de_classes_que_a_rede_errou)
for index in range(int(list_len/20)):
    print(indice_de_classes_que_a_rede_errou[index*20:(index+1)*20])
print(indice_de_classes_que_a_rede_errou[(index+1)*20:(index+1)*20+(list_len % 20)])
#
# Imagens...
plotar = True
if plotar:
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(18, 5.3))
    for indice in indice_de_classes_que_a_rede_errou:
        ax1.cla()
        ax1.text(0.25,0.2,str(ytar_tes[indice]), fontsize=228, color='black')
        ax1.set_title(' ALVO ')
        ax1.set_xticks([]);        ax1.set_xticklabels([])
        ax1.set_yticks([]);        ax1.set_yticklabels([])
        #
        ax2.cla()
        ax2.imshow(xin_tes[indice])
        ax2.set_title("Dataset Tes, Frame {}, Alvo = {}, Saida {}".format(indice,ytar_tes[indice],yout_tes[indice]))
        #
        ax3.cla()
        ax3.text(0.25,0.2,str(yout_tes[indice]), fontsize=228, color='red')
        ax3.set_title(' SAÍDA ')
        ax3.set_xticks([]);        ax3.set_xticklabels([])
        ax3.set_yticks([]);        ax3.set_yticklabels([])
        # Note that using time.sleep does *not* work here!
        plt.pause(3)
# endregion
# endregion
