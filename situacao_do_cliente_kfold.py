#!/usr/bin/env python3.7
#!-*- coding: utf8 -*-

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import cross_val_score


"""Lê o arquivo utilizando o pandas e guarda suas características de
dados e marcações em X e Y"""
df = pd.read_csv('situacao_do_cliente.csv')
X_df = df[['recencia','frequencia', 'semanas_de_inscricao']]
Y_df = df['situacao']
"""Esta seção é utilizada para converter as variáveis categóricas (caso existam)
em variáveis numéricas. Por exemplo um arquivo que possui uma coluna com valores
classificados em string (tópicos de pesquisa) é convertida em números, tendo um
valor para cada texto encontrado na coluna"""
Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df
"""Guarda apenas os valores de X e Y sem os valores herdados o data_frame"""
X = Xdummies_df.values
Y = Ydummies_df.values 
"""Define a quantidade de dados que serão utilizados para treinar e validar 
o algoritmo"""
porcentagem_de_treino = 0.8
tamanho_de_treino = int(porcentagem_de_treino * len(Y))
"""Recupera os dados de treino e validação guardando em locais distintos"""
treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]
validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]

"""Função criada para generalizar o teste dos algoritmos utilizados neste código"""
def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes)->int:
    """
    nome: Nome do algoritmo utilizado para o teste
    modelo: modelo utilizado para o aprendizado
    treino_dados: Dados utilizados para o treino do algoritmo (X)
    treino_marcacoes: Markings que indicam o que são cada dado em X (Y)

    Utiliza o método de K-Fold para fazer o teste de diversos modos nos dados
    de acordo com a constante K definida e retorna a média de acertos para
    o processo.
    """
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k)
    taxa_de_acerto = np.mean(scores)
    msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto

"""Função utilizada para testar o algoritmo com dados reais"""
def teste_real(modelo, validacao_dados, validacao_marcacoes):
    """
    modelo: Modelo que será utilizado para testar com dados reais
    (de validação) o algoritmo
    validacao_dados: Dados de validação que foram separados para o
    teste final (X)
    validacao_marcacoes: Marcoções que indicam a que se referem os
    dados em X (Y)

    Faz o predict e recupera um array contendo apenas os resutados
    iguais aos que deveriam ser em relação aos dados reais.
    """
    resultado = modelo.predict(validacao_dados)
    acertos = resultado == validacao_marcacoes
    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
    print(msg)

"""Este dicionário armazena a taxa de acerto de cada algoritmo"""
resultados = {}

"""Testa alguns algoritmos estudados e guarda a taxa de acerto dentro de resultados"""
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

"""Mostra todas as taxas de acerto para cada algoritmo testado"""
print(resultados)
"""Pega o modelo com a maior taxa de acerto, atribui a vencedor e informa no console"""
maximo = max(resultados)
vencedor = resultados[maximo]
print("Vencerdor: ")
print(vencedor)

"""Executa método fit no modelo vencedor e faz o teste real
para verificar o acerto com os dados de validação"""
vencedor.fit(treino_dados, treino_marcacoes)
teste_real(vencedor, validacao_dados, validacao_marcacoes)

"""Este é o algoritmo que chuta o valor de maior frequência para
classificar os dados de entrada. Utilizado para comparar com o modelo
testado utilizando algoritmos consagrados"""
acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

total_de_elementos = len(validacao_dados)
print("Total de teste: %d" % total_de_elementos)