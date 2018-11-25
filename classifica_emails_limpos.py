#!/usr/bin/env python3.7
#!-*- coding: utf8 -*-

import nltk
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score


#Pré-processamento dos dados de e-mail
""" Limpeza dos dados
1. Utilizado stop_wrods para remover palavras desnecessárias
2. Utilizado word_tokenize para manter apenas as palavras sem pontuação (quebrar as palavras com pontos)
3. Utilizado filtro para inserir no dicionário apenas as palavras com mais de 2 caracteres
4. Utilizado Stemmer para analisar apenas as raízes dos textos
"""
#nltk.download("stopwords")
#nltk.download("rslp") #Removedor de sulfixo da língua portuguesa RSLP
#nltk.download("punkt") #Trabalha com a pontuação
""""Lê os dados contidos no arquivo de e-mail e trata os textos"""
classificacoes_df = pd.read_csv("emails.csv", encoding="utf8")
textos_puros = classificacoes_df["email"]
"""Lista contendo todos os emails quebrados em listas"""
frases = textos_puros.str.lower()
"""Remove a pontuação das frases antes de atualizar o valor dos textos
quebrados em lista"""
textos_quebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]
"""Recupera stopWords e raízes de palavras para limpar os textos"""
stop_words = nltk.corpus.stopwords.words("portuguese")
stemmer = nltk.RSLPStemmer()
"""Cria um conjunto de dados para que as chaves não sejam repetidas"""
dicionario_classificador = set()
"""Adiciona as chaves em uma lista para mapear as palavras.
A palavra só entra no dicionário se antes de recuperar sua raíz, a mesma possuir
pelo menos 3 caracteres, pois normalmente palavras com menos de 3 caracteres que
não são raízes não fazem muita diferença para o alagoritmo"""
for lista in textos_quebrados:
    validas = [stemmer.stem(palavra) for palavra in lista if palavra not in stop_words and len(palavra) > 2]
    dicionario_classificador.update(validas)

"""Vetoriza o texto do e-mail"""
def vetorizar_Texto(texto: list, tradutor: dict):
    """
    Função que recebe um texto e contabiliza as raízes de cada palavra
    da lista recebida, adicionando um valor representado no array vetor
    """
    vetor = [0] * len(tradutor)
    for palavra in texto:
        if len(palavra) > 0:
            raiz = stemmer.stem(palavra)
            if raiz in tradutor:
                posicao = tradutor[raiz]
                vetor[posicao] += 1
    return vetor

total_de_palavras = len(dicionario_classificador)
print(total_de_palavras)
print(dicionario_classificador)
"""Cria uma tupla contendo as chaves da lista e um índice
que indica qual a posição do array em relação à quantidade
de palavras encontradas"""
tuplas = zip(dicionario_classificador, range(total_de_palavras))
"""Cria um dicionário palavra/índice à partir da tupla para
permitir a indexação por texto retornando o índice do array"""
tradutor = {palavra:indice for palavra, indice in tuplas}
"""Textos vetorizados (Lista contendo os textos representados em números"""
vetores_de_texto = [vetorizar_Texto(texto, tradutor) for texto in textos_quebrados]
"""Cria os markings que definem as classes a que pertencem cada resultado"""
marcas = classificacoes_df['classificacao']

#Algoritmo de machine learning
"""Vetores de texto são os dados do eixo X para o histograma e as classes de marcação que 
indicam o que significa cada posição do vetor é o Y"""
X = np.array(vetores_de_texto)
Y = np.array(marcas.tolist())
"""Define a quantidade de dados que vão para treino e para validaçã final (80/20)"""
porcentagem_de_treino = 0.8
tamanho_do_treino = int(porcentagem_de_treino * len(Y))
tamanho_validacao = len(Y) - tamanho_do_treino
"""Recupera dados de treino e de validação que estão dentro dos dados originais"""
treino_dados = X[0:tamanho_do_treino]
treino_marcas = Y[0:tamanho_do_treino]
validacao_dados = X[tamanho_do_treino:]
validacao_marcas = Y[tamanho_do_treino:]
print(tamanho_do_treino)

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes)->int:
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k)
    taxa_de_acerto = np.mean(scores)
    logger = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(logger)
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

"""Faz o teste em todos os algoritmos estudados"""
resultados = {}
#OneVsRest
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcas)
resultados[resultadoOneVsRest] = modeloOneVsRest
#OneVsOne
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcas)
resultados[resultadoOneVsOne] = modeloOneVsOne
#MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcas)
resultados[resultadoMultinomial] = modeloMultinomial
#Adaboost
modeloAdaBoost = AdaBoostClassifier(random_state=0)
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcas)
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
vencedor.fit(treino_dados, treino_marcas)
teste_real(vencedor, validacao_dados, validacao_marcas)
"""Este é o algoritmo que chuta o valor de maior frequência para
classificar os dados de entrada. Utilizado para comparar com o modelo
testado utilizando algoritmos consagrados"""
acerto_base = max(Counter(validacao_marcas).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcas)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)
total_de_elementos = len(validacao_dados)
print("Total de teste: %d" % total_de_elementos)