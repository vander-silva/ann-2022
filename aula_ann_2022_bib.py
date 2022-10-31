## aula_ann_2022_bib
# @author Vanderlei A. Silva - 2022/08/29

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from itertools import chain
import random
from operator import itemgetter

## Plotar imagens Mnist animadas na tela de forma sequencial
# @param imagem_data_set: subconjunto dataset mnist de imagens de entradas
# @param target_data_set: subconjunto dataset mnist de targets
# @param output_data_set: subconjunto dataset mnist de ann outputs
# @param delay_frames: (tempo ms) entre um frame e outro
# @param frames: define quais frames exibir. Exemplos: frames = 8; frames = [10,15,20];  frames = range(10,45)
# @return apenas exibe imagens mnist na tela de forma sequencial
class MnistAnimePlot():
    artist_list = []

    def __init__(self,  input_data_set,
                        target_data_set,
                        output_data_set=0,
                        delay_frames=1000,
                        repeat=False,
                        frames=10):
        self.delay_frames = delay_frames
        self.input_data_set = input_data_set
        self.target_data_set = target_data_set
        self.frames = frames
        self.repeat = repeat
        self.end_index = 0
        self.ini_index = 0
        if output_data_set == 0:
            self.output_data_set = target_data_set
        else:
            self.output_data_set = output_data_set
        if isinstance(self.frames, list):
            self.ini_index =  self.frames[0]
            self.end_index =  self.frames[-1]
            print('frames is a list')
        elif isinstance(self.frames, range):
            self.ini_index =  self.frames.start
            self.end_index =  self.frames.stop
            print('frames is a range')
        else:
            self.ini_index = 0
            self.end_index = self.frames - 1
            print('frames is a number',type(self.frames))

    def fun_init(self):
        plt.title('Índice ' + str(self.ini_index) + ',     ' +
                  'Alvo = ' + str(self.target_data_set[self.ini_index]) + ',  ' +
                  'Saída = ' + str(self.target_data_set[self.ini_index]))
        self.ax.imshow(self.input_data_set[self.ini_index])
        self.line.set_data(self.input_data_set[self.ini_index])
        return self.line

    def fun_animate(self, frame):
        plt.title('Índice ' + str(frame) + ',     ' +
                  'Alvo = ' + str(self.target_data_set[frame]) + ',  ' +
                  'Saída = ' + str(
            self.target_data_set[frame]))
        self.ax.imshow(self.input_data_set[frame])
        # Caso seja o último frame, fechar a figura!
        if frame == self.end_index:
            time.sleep(self.delay_frames/1000)
            plt.close(self.fig)
        #
        return self.line

    def show_plot_animation(self):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([],[],lw=3)
        ani = FuncAnimation(self.fig,
                            self.fun_animate,
                            init_func=self.fun_init,
                            interval=self.delay_frames,
                            blit=True,
                            frames=self.frames,
                            repeat=self.repeat)
        plt.show()


class MnistDataSetAnalysis():

    def __init__(self):
        self.classes_def = []
        self.input_dataset_tre = []
        self.target_dataset_tre = []
        self.input_dataset_val = []
        self.target_dataset_val = []
        self.input_dataset_tes = []
        self.target_dataset_tes = []
        self.amostras_dataset_tre = 0
        self.amostras_dataset_val = 0
        self.amostras_dataset_tes = 0

        for classe in range(10):
            # Definições de classes: lista contando as 10 classes [0..9]
            self.classes_def.append(classe)

    def set_dataset_tre_original(self,input_tre,target_tre):
        self.dataset_name_tre_orig = 'Dataset de Treinamento Original'
        self.input_dataset_tre_orig = input_tre
        self.taget_dataset_tre_orig = target_tre
        self.amostras_dataset_tre_orig, lin, col = self.input_dataset_tre_orig.shape

    def set_dataset_tes_original(self,input_tes,target_tre):
        self.dataset_name_tes_orig = 'Dataset de Teste Original'
        self.input_dataset_tes_orig = input_tes
        self.taget_dataset_tes_orig = target_tre
        self.amostras_dataset_tre_orig, lin, col = self.input_dataset_tes_orig.shape

    def set_dataset_tre(self,input_tre,target_tre):
        self.dataset_name_tre = 'Dataset de Treinamento'
        self.input_dataset_tre = input_tre
        self.target_dataset_tre = target_tre
        self.amostras_dataset_tre, lin, col = self.input_dataset_tre.shape

    def set_dataset_val(self,input_val,target_val):
        self.dataset_name_val = 'Dataset de Validação'
        self.input_dataset_val = input_val
        self.target_dataset_val = target_val
        self.amostras_dataset_val, lin, col = self.input_dataset_val.shape

    def set_dataset_tes(self, input_tes, target_tes):
        self.dataset_name_tes = 'Dataset de Teste'
        self.input_dataset_tes = input_tes
        self.target_dataset_tes = target_tes
        self.amostras_dataset_tes, lin, col = self.input_dataset_tes.shape

    def get_dataset_tre(self,normalizar=1):
        if normalizar == 1:
            return self.input_dataset_tre, self.target_dataset_tre
        else:
            return self.input_dataset_tre/normalizar, self.target_dataset_tre

    def get_dataset_val(self,normalizar=1):
        if normalizar == 1:
            return self.input_dataset_val, self.target_dataset_val
        else:
            return self.input_dataset_val/normalizar, self.target_dataset_val

    def get_dataset_tes(self,normalizar=1):
        if normalizar == 1:
            return self.input_dataset_tes, self.target_dataset_tes
        else:
            return self.input_dataset_tes/normalizar, self.target_dataset_tes

    def separe_classes_por_indice(self,target):
        indice_2d_list = []
        for classe in range(10):
            indice_2d_list.append([])
        for indice in range(len(target)):
            # Se o elemento é igual a zero, então guarde o indice
            if target[indice] == 0:
                indice_2d_list[0].append(indice)
            if target[indice] == 1:
                indice_2d_list[1].append(indice)
            if target[indice] == 2:
                indice_2d_list[2].append(indice)
            if target[indice] == 3:
                indice_2d_list[3].append(indice)
            if target[indice] == 4:
                indice_2d_list[4].append(indice)
            if target[indice] == 5:
                indice_2d_list[5].append(indice)
            if target[indice] == 6:
                indice_2d_list[6].append(indice)
            if target[indice] == 7:
                indice_2d_list[7].append(indice)
            if target[indice] == 8:
                indice_2d_list[8].append(indice)
            if target[indice] == 9:
                indice_2d_list[9].append(indice)
        classe_len_1d_list = []
        for classe in range(10):
            classe_len_1d_list.append(len(indice_2d_list[classe]))
        return indice_2d_list, classe_len_1d_list

    ## Analisa o dataset quanto ao equilíbrio entre classes
    def show_class_balance_orig(self,plotar=True):
        # Criando Lista 2d [classe][indice] para armazenar indices de datasets
        self.clas_x_ind_tre_orig, self.classlen_tre_orig = self.separe_classes_por_indice(self.taget_dataset_tre_orig)
        self.clas_x_ind_tes_orig, self.classlen_tes_orig = self.separe_classes_por_indice(self.taget_dataset_tes_orig)
        if plotar:
            self.plot_barras_totais_por_classe_orig()

    ## Analisa o dataset quanto ao equilíbrio entre classes
    def show_class_balance(self,plotar=True):
        # Criando Lista 2d [classe][indice] para armazenar indices de datasets
        self.clas_x_ind_tre, self.classlen_tre = self.separe_classes_por_indice(self.target_dataset_tre)
        self.clas_x_ind_val, self.classlen_val = self.separe_classes_por_indice(self.target_dataset_val)
        self.clas_x_ind_tes, self.classlen_tes = self.separe_classes_por_indice(self.target_dataset_tes)
        if plotar:
            self.plot_barras_totais_por_classe()

    def set_class_balance(self,plotar=True):
        # Criando as listas que devem armazenar os indices por classe, em qtdade igual para todas as classes
        indices_por_classe_lengthmin_tre = []
        indices_por_classe_lengthmin_val = []
        indices_por_classe_lengthmin_tes = []
        # Distribuição de tamanho de dataset
        tre_plus_val_size = min(self.classlen_tre_orig) # minimo entre comprimento de todas as classes
        tes_size = min(self.classlen_tes_orig)
        val_size = tes_size
        tre_size = tre_plus_val_size - val_size
        print('Size datasets [Tre+Val, Tre, Val, Tes] = [',tre_plus_val_size*10,tre_size*10,val_size*10,tes_size*10,']')
        # Distribuindo os indices originais, já separados por classe, nos datasets Tre, Val, Tes
        for classe in range(10):
            indices_por_classe_lengthmin_val.append(self.clas_x_ind_tre_orig[classe][:val_size])
            indices_por_classe_lengthmin_tre.append(self.clas_x_ind_tre_orig[classe][val_size:tre_plus_val_size])
            indices_por_classe_lengthmin_tes.append(self.clas_x_ind_tes_orig[classe][:tes_size])
        # Passando a lista 2D para 1D com a bib chain:
        indices_balanceados_tre = list(chain.from_iterable(indices_por_classe_lengthmin_tre))
        indices_balanceados_val = list(chain.from_iterable(indices_por_classe_lengthmin_val))
        indices_balanceados_tes = list(chain.from_iterable(indices_por_classe_lengthmin_tes))
        # Embaralhando os indices
        indices_embaralhados_tre = self.embaralhar_indices(indices_balanceados_tre)
        indices_embaralhados_val = self.embaralhar_indices(indices_balanceados_val)
        indices_embaralhados_tes = self.embaralhar_indices(indices_balanceados_tes)
        if plotar:
            self.plot_stem_dataset_target(self.taget_dataset_tre_orig[indices_balanceados_tre],  'Dataset Tre: Antes de Embaralhar')
            self.plot_stem_dataset_target(self.taget_dataset_tre_orig[indices_embaralhados_tre], 'Dataset Tre: Depois de Embaralhar')
        # Atualizando os datasets de originais para equilibrados
        self.set_dataset_tre(self.input_dataset_tre_orig[indices_embaralhados_tre],
                             self.taget_dataset_tre_orig[indices_embaralhados_tre])
        #
        self.set_dataset_val(self.input_dataset_tre_orig[indices_embaralhados_val],
                             self.taget_dataset_tre_orig[indices_embaralhados_val])
        #
        self.set_dataset_tes(self.input_dataset_tes_orig[indices_embaralhados_tes],
                             self.taget_dataset_tes_orig[indices_embaralhados_tes])
        # Plotando a métrica de distribuição de datasets
        total_amostras = self.amostras_dataset_tre + self.amostras_dataset_val + self.amostras_dataset_tes
        print('Distribuição de Amostras:')
        print('Tre:', '{:>6d},'.format(self.amostras_dataset_tre), '   ' +
              'Val:', '{:>6d},'.format(self.amostras_dataset_val), '   ' +
              'Tes:', '{:>6d},'.format(self.amostras_dataset_tes), '   ' +
              'Total: ', total_amostras)
        print('Tre:', '{:>5.2f}%,'.format(self.amostras_dataset_tre/total_amostras * 100), '   ' +
              'Val:', '{:>5.2f}%,'.format(self.amostras_dataset_val/total_amostras * 100), '   ' +
              'Tes:', '{:>5.2f}%,'.format(self.amostras_dataset_tes/total_amostras * 100), '   ' +
              'Total:   100%')

    def embaralhar_indices(self,indices_originais):
        indices_len = len(indices_originais)
        random_index = random.sample(range(indices_len), indices_len)
        indices_embaralhados = list(itemgetter(*random_index)(indices_originais))
        return indices_embaralhados

    def verificar_maior_repeticao(self, target_dataset, title='Title Here'):
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
            print('Classe:', classe, '  Maior repetição:', maior_repeticao, '  Índice de início:', indice_inicio_repeticao)

    def plot_stem_dataset_target(self,target,title='Title here',plotar=True):
        if plotar:
            fig, ax = plt.subplots()
            ax.stem(target)
            ax.set_title(title)
            plt.grid(axis = 'y')
            plt.ylim([-1, 10])
            plt.show()

    def plot_barras_totais_por_classe_orig(self):
        # Conferindo a soma
        print('Dataset Tre: soma total de elementos = ', sum(self.classlen_tre_orig))
        print('Dataset Tes: soma total de elementos = ', sum(self.classlen_tes_orig))
        # Plotando em gráfico de barras os totais por classe
        fig, axs = plt.subplots(1,2)
        dataset_names = ['Dataset de Treinamento Original', 'Dataset de Testes Original']
        dataset_ref = -1
        for ax in axs.flat:
            dataset_ref += 1
            ax.set(xlabel='Classes', ylabel='Amostras Por Classe')
            ax.set_title(dataset_names[dataset_ref])
            ax.set_xticks(self.classes_def)
            ax.set_xticklabels(self.classes_def)
            if dataset_ref == 0:
                p0 = ax.bar(self.classes_def , self.classlen_tre_orig, width=1, edgecolor="white", linewidth=0.7)
                ax.bar_label(p0, label_type='edge')
                ax.set_ylim([0, 1.1*max(self.classlen_tre_orig)])
            else:
                p1 = ax.bar(self.classes_def , self.classlen_tes_orig, width=1, edgecolor="white", linewidth=0.7)
                ax.bar_label(p1, label_type='edge')
                ax.set_ylim([0, 1.1 * max(self.classlen_tes_orig)])
        plt.show()

    def plot_barras_totais_por_classe(self):
        # Conferindo a soma
        print('Dataset Tre: soma total de elementos = ', sum(self.classlen_tre))
        print('Dataset Val: soma total de elementos = ', sum(self.classlen_val))
        print('Dataset Tes: soma total de elementos = ', sum(self.classlen_tes))
        # Plotando em gráfico de barras os totais por classe
        fig, axs = plt.subplots(1,3)
        dataset_names = ['Dataset de Treinamento - Métricas', 'Dataset de Validação - Métricas', 'Dataset de Testes - Métricas']
        dataset_ref = -1
        for ax in axs.flat:
            dataset_ref += 1
            ax.set(xlabel='Classes', ylabel='Amostras Por Classe')
            ax.set_title(dataset_names[dataset_ref])
            ax.set_xticks(self.classes_def)
            ax.set_xticklabels(self.classes_def)
            if dataset_ref == 0:
                p0 = ax.bar(self.classes_def , self.classlen_tre, width=1, edgecolor="white", linewidth=0.7)
                ax.bar_label(p0, label_type='edge')
                ax.set_ylim([0, 1.1*max(self.classlen_tre)])
            elif dataset_ref == 1:
                p1 = ax.bar(self.classes_def , self.classlen_val, width=1, edgecolor="white", linewidth=0.7)
                ax.bar_label(p1, label_type='edge')
                ax.set_ylim([0, 1.1 * max(self.classlen_val)])
            else:
                p2 = ax.bar(self.classes_def , self.classlen_tes, width=1, edgecolor="white", linewidth=0.7)
                ax.bar_label(p2, label_type='edge')
                ax.set_ylim([0, 1.1 * max(self.classlen_tes)])
        plt.show()


