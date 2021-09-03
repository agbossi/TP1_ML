import os
import NaiveBayes as model
import pandas as pd

blacklist = {'a', 'ante', 'bajo', 'contra', 'desde', 'hacia', 'para', 'cabe', 'con', 'de', 'del'
             , 'durante', 'en', 'entre', 'hasta', 'mediante', 'según', 'sin', 'sobre', 'tras'
             , 'versus', 'vía', 'el', 'la', 'los', 'las', 'le', 'les', 'un', 'una', 'unos', 'unas'
             , 'algo', 'alguien', 'algún', 'como', 'donde', 'cual', 'cómo', 'dónde', 'cuál', 'que'
             , 'qué', 'y', 'además', 'aparte', 'asimismo', 'también', 'encima', 'más', 'es', 'sin'
             , 'no', 'si', 'pues', 'se', 'junto', 'su', 'sus', 'otra', 'otro', 'otras', 'otros'
             , 'antes', 'después', 'al', 'cada', 'así', 'ella', 'fue', 'menos', 'quien', 'quienes'
             , 'quién', 'me', 'mi', 'son', 'es', 'mis'}





# path = os.path.abspath('../Data/PreferenciasBritanicos(1).xlsx')
# df = pd.read_excel(path)
# classifier = model.DiscreteNaiveBayes()
# classifier.train(df)

path = os.path.abspath('../Data/Noticias_argentinas.xlsx')
df = pd.read_excel(path)
useful_df = df[['titular', 'categoria']]
filtered_df = filter_words(useful_df)
print(filtered_df)