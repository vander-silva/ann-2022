## aula_ann_2022_bib
# @author Vanderlei A. Silva - 2022/08/29

import matplotlib.pyplot as plt
#import time
from itertools import chain
import random
from operator import itemgetter


class DatasetMinistAulaANN2022:

    def __init__(self, imagens=[], alvos=[], nome=''):
        self.nome = nome
        self.classes_def = []
        self.classes_length = []
        self.lista2d_alvo_indice = []
        self.imagens = imagens
        self.alvos = alvos
        for classe in range(10):
            self.classes_def.append(classe)

    def set_dataset_imagens_alvos(self, imagens, alvos):
        self.imagens = imagens
        self.alvos = alvos

    def set_lista2d_alvo_indice(self, lista2d_alvo_indice, classes_length):
        self.lista2d_alvo_indice = lista2d_alvo_indice
        self.classes_length = classes_length


def plot_imagens_tre_tes_originais(df_tre_orig, df_tes_orig, modo='Matriz', plotar=True):
    if plotar and modo == 'Matriz':
        linhas = 4
        colunas = 6
        fig, axs = plt.subplots(linhas, colunas, figsize=(14, 9))
        for lin in range(linhas):
            for col in range(colunas):
                axs[lin, col].axes.get_xaxis().set_visible(False)
                axs[lin, col].axes.get_yaxis().set_visible(False)
                axs[lin, col].cla();
                if col < 3:
                    frame = random.sample(range(df_tre_orig.imagens.shape[0]), 1)[0]
                    axs[lin, col].imshow(df_tre_orig.imagens[frame])
                    axs[lin, col].set_title("Tre[{}], Alvo {}".format(frame, df_tre_orig.alvos[frame]), color='blue')
                else:
                    frame = random.sample(range(df_tes_orig.imagens.shape[0]), 1)[0]
                    axs[lin, col].imshow(df_tes_orig.imagens[frame])
                    axs[lin, col].set_title("Tes[{}], Alvo {}".format(frame, df_tes_orig.alvos[frame]), color='red')
        plt.show()

    if plotar and modo == 'Temporal':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        for i in range(10):
            ax1.cla();
            ax1.imshow(df_tre_orig.imagens[i])
            ax2.cla();
            ax2.imshow(df_tes_orig.imagens[i])
            ax1.set_title("Dataset Tre,    Frame {},    Alvo = {}".format(i, df_tre_orig.alvos[i]))
            ax2.set_title("Dataset Tes,    Frame {},    Alvo = {}".format(i, df_tes_orig.alvos[i]))
            # Note that using time.sleep does *not* work here!
            plt.pause(1)


def separe_classes_por_indice(dataset):
    lista2d_alvo_indice = []
    # Criar lista vazia com 10 classes
    for classe in range(10):
        lista2d_alvo_indice.append([])
    # Preencher a lista
    for indice in range(dataset.alvos.shape[0]):
        # Se o elemento é igual a zero, então guarde o indice
        lista2d_alvo_indice[dataset.alvos[indice]].append(indice)
    # Criar lista com comprimentos de cada classe
    classe_len_list = []
    for classe in range(10):
        classe_len_list.append(len(lista2d_alvo_indice[classe]))
    #
    return lista2d_alvo_indice, classe_len_list


def plot_barras_totais_por_classe_orig(df_tre_orig, df_tes_orig):
    # def plot_barras_totais_por_classe_orig(classes_def, classlen_tre_orig, classlen_tes_orig):
    #         self.lista2d_alvo_indice = lista2d_alvo_indice
    #         self.classes_length = classes_length
    # Conferindo a soma
    print('Dataset Tre: soma total de elementos = ', sum(df_tre_orig.classes_length))
    print('Dataset Tes: soma total de elementos = ', sum(df_tes_orig.classes_length))
    # Plotando em gráfico de barras os totais por classe
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    dataset_names = [df_tre_orig.nome, df_tes_orig.nome]
    dataset_ref = -1
    for ax in axs.flat:
        dataset_ref += 1
        ax.set(xlabel='Classes', ylabel='Amostras Por Classe')
        ax.set_title(dataset_names[dataset_ref])
        ax.set_xticks(df_tre_orig.classes_def)
        ax.set_xticklabels(df_tre_orig.classes_def)
        if dataset_ref == 0:
            p0 = ax.bar(df_tre_orig.classes_def, df_tre_orig.classes_length, width=1, edgecolor="white", linewidth=0.7)
            ax.bar_label(p0, label_type='edge')
            ax.set_ylim([0, 1.1 * max(df_tre_orig.classes_length)])
        else:
            p1 = ax.bar(df_tes_orig.classes_def, df_tes_orig.classes_length, width=1, edgecolor="white", linewidth=0.7)
            ax.bar_label(p1, label_type='edge')
            ax.set_ylim([0, 1.1 * max(df_tes_orig.classes_length)])
    plt.show()


def set_class_balance(df_tre_orig, df_tes_orig, plotar=True):
    # Criando as listas que devem armazenar os indices por classe, em qtdade igual para todas as classes
    indices_por_classe_lengthmin_tre = []
    indices_por_classe_lengthmin_val = []
    indices_por_classe_lengthmin_tes = []
    # Distribuição de tamanho de dataset
    df_tre_orig.classes_length
    tre_plus_val_size = min(df_tre_orig.classes_length)  # minimo entre comprimento de todas as classes
    tes_size = min(df_tes_orig.classes_length)
    val_size = tes_size
    tre_size = tre_plus_val_size - val_size
    print('Size datasets [Tre+Val, Tre, Val, Tes] = [', tre_plus_val_size * 10, tre_size * 10, val_size * 10,
          tes_size * 10, ']')
    # Distribuindo os indices originais, já separados por classe, nos datasets Tre, Val, Tes
    for classe in range(10):
        indices_por_classe_lengthmin_val.append(df_tre_orig.lista2d_alvo_indice[classe][:val_size])
        indices_por_classe_lengthmin_tre.append(df_tre_orig.lista2d_alvo_indice[classe][val_size:tre_plus_val_size])
        indices_por_classe_lengthmin_tes.append(df_tes_orig.lista2d_alvo_indice[classe][:tes_size])
    # Passando a lista 2D para 1D com a bib chain:
    indices_balanceados_tre = list(chain.from_iterable(indices_por_classe_lengthmin_tre))
    indices_balanceados_val = list(chain.from_iterable(indices_por_classe_lengthmin_val))
    indices_balanceados_tes = list(chain.from_iterable(indices_por_classe_lengthmin_tes))
    # Embaralhando os indices
    indices_embaralhados_tre = embaralhar_indices(indices_balanceados_tre)
    indices_embaralhados_val = embaralhar_indices(indices_balanceados_val)
    indices_embaralhados_tes = embaralhar_indices(indices_balanceados_tes)
    if plotar:
        plot_stem_dataset_target(df_tre_orig.alvos[indices_balanceados_tre],
                                      'Dataset Tre: Antes de Embaralhar')
        plot_stem_dataset_target(df_tre_orig.alvos[indices_embaralhados_tre],
                                      'Dataset Tre: Depois de Embaralhar')
    # Atualizando os datasets de originais para equilibrados
   # self.set_dataset_tre(self.input_dataset_tre_orig[indices_embaralhados_tre],
   #                      self.taget_dataset_tre_orig[indices_embaralhados_tre])
   # #
   # self.set_dataset_val(self.input_dataset_tre_orig[indices_embaralhados_val],
   #                      self.taget_dataset_tre_orig[indices_embaralhados_val])
   # #
   # self.set_dataset_tes(self.input_dataset_tes_orig[indices_embaralhados_tes],
   #                      self.taget_dataset_tes_orig[indices_embaralhados_tes])

    return indices_embaralhados_tre, indices_embaralhados_val, indices_embaralhados_tes


def embaralhar_indices(indices_originais):
    indices_len = len(indices_originais)
    random_index = random.sample(range(indices_len), indices_len)
    indices_embaralhados = list(itemgetter(*random_index)(indices_originais))
    return indices_embaralhados


def plot_stem_dataset_target(target, title='Title here', plotar=True):
    if plotar:
        fig, ax = plt.subplots()
        ax.stem(target)
        ax.set_title(title)
        plt.grid(axis='y')
        plt.ylim([-1, 10])
        plt.show()


def print_metrica_de_distribuicao_entre_datasets(df_tre, df_val, df_tes):
    total_amostras = df_tre.imagens.shape[0] + df_val.imagens.shape[0] + df_tes.imagens.shape[0]
    print('Distribuição de Amostras:')
    print('Tre:', '{:>6d},'.format(df_tre.imagens.shape[0]), '   ' +
          'Val:', '{:>6d},'.format(df_val.imagens.shape[0]), '   ' +
          'Tes:', '{:>6d},'.format(df_tes.imagens.shape[0]), '   ' +
          'Total: ', total_amostras)
    print('Tre:', '{:>5.2f}%,'.format(df_tre.imagens.shape[0] / total_amostras * 100), '   ' +
          'Val:', '{:>5.2f}%,'.format(df_val.imagens.shape[0] / total_amostras * 100), '   ' +
          'Tes:', '{:>5.2f}%,'.format(df_tes.imagens.shape[0] / total_amostras * 100), '   ' +
          'Total:   100%')


def plot_barras_totais_por_classe(df_tre, df_val, df_tes):
    # Conferindo a soma
    print('Dataset Tre: soma total de elementos = ', sum(df_tre.classes_length))
    print('Dataset Val: soma total de elementos = ', sum(df_val.classes_length))
    print('Dataset Tes: soma total de elementos = ', sum(df_tes.classes_length))
    # Plotando em gráfico de barras os totais por classe
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    dataset_names = ['Dataset de Treinamento - Métricas', 'Dataset de Validação - Métricas',
                     'Dataset de Testes - Métricas']
    dataset_ref = -1
    for ax in axs.flat:
        dataset_ref += 1
        ax.set(xlabel='Classes', ylabel='Amostras Por Classe')
        ax.set_title(dataset_names[dataset_ref])
        ax.set_xticks(df_tre.classes_def)
        ax.set_xticklabels(df_tre.classes_def)
        if dataset_ref == 0:
            p0 = ax.bar(df_tre.classes_def, df_tre.classes_length, width=1, edgecolor="white", linewidth=0.7)
            ax.bar_label(p0, label_type='edge')
            ax.set_ylim([0, 1.1 * max(df_tre.classes_length)])
        elif dataset_ref == 1:
            p1 = ax.bar(df_val.classes_def, df_val.classes_length, width=1, edgecolor="white", linewidth=0.7)
            ax.bar_label(p1, label_type='edge')
            ax.set_ylim([0, 1.1 * max(df_val.classes_length)])
        else:
            p2 = ax.bar(df_tes.classes_def, df_tes.classes_length, width=1, edgecolor="white", linewidth=0.7)
            ax.bar_label(p2, label_type='edge')
            ax.set_ylim([0, 1.1 * max(df_tes.classes_length)])
    plt.show()


def get_dataset_normalizado(dataset, valor):
    return dataset.imagens / valor, dataset.alvos


def verificar_maior_repeticao(target_dataset, title='Title Here'):
    saida = []
    print(title)
    for classe in range(10):
        elemento_anterior = -1
        maior_repeticao = 1
        repeticao_atual = 1
        for index in range(len(target_dataset)):
            if target_dataset[index] == classe:
                if elemento_anterior == classe:
                    repeticao_atual += 1
                else:
                    elemento_anterior = classe
            else:
                elemento_anterior = -1
                repeticao_atual = 1
            if repeticao_atual > maior_repeticao:
                maior_repeticao = repeticao_atual
                indice_inicio_repeticao = index - repeticao_atual + 1
        saida.append({'Maior Repeticao': maior_repeticao, 'Indice de Inicio': indice_inicio_repeticao})
        print('Classe:', classe, '  Maior repetição:', maior_repeticao, '  Índice de início:',
              indice_inicio_repeticao)


def print_dataset_info(dataset):
    aux = dataset.nome+':'
    print(aux,
          '\n',
          dataset.imagens.shape[0], 'Imagens ',
          dataset.imagens.shape[1], 'x',
          dataset.imagens.shape[2], '; ',
          dataset.alvos.shape[0], 'Alvos;')


def plotar_ann_em_operacao(xin_tes, ytar_tes, yout_tes, lista_de_indices):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.3))
    #for indice in indice_de_classes_que_a_rede_errou:
    for indice in lista_de_indices:
        ax1.cla()
        ax1.text(0.25, 0.2, str(ytar_tes[indice]), fontsize=228, color='black')
        ax1.set_title(' ALVO ')
        ax1.set_xticks([]);
        ax1.set_xticklabels([])
        ax1.set_yticks([]);
        ax1.set_yticklabels([])
        #
        ax2.cla()
        ax2.imshow(xin_tes[indice])
        ax2.set_title(
            "Dataset Tes, Frame {}, Alvo = {}, Saida {}".format(indice, ytar_tes[indice], yout_tes[indice]))
        #
        ax3.cla()
        ax3.text(0.25, 0.2, str(yout_tes[indice]), fontsize=228, color='red')
        ax3.set_title(' SAÍDA ')
        ax3.set_xticks([]);
        ax3.set_xticklabels([])
        ax3.set_yticks([]);
        ax3.set_yticklabels([])
        # Note that using time.sleep does *not* work here!
        plt.pause(3)
