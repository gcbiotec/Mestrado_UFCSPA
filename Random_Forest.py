import nltk
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

amostras_excel = r"C:\Users\gabriel.silva\Documents\Mestrado\Planilha_EHR.xlsx"
amostras_df = pd.read_excel(amostras_excel)

'''Retirando os números e parenteses de CID'''
pattern = r'\(\d+\)'
pattern_evol = r"[,;.:#-+/)(\d+>]"
amostras_df['CID'] = amostras_df['CID'].str.replace(pattern, '')
amostras_df['EHR'] = amostras_df['EHR'].str.replace(pattern_evol, '')

# Retirando nomes mais comuns da língua Portuguesa:

def remove_names_from_dataframe(csv_file_path, dataframe, column_name):
    try:
        # Read the CSV file into a list of names
        with open(csv_file_path, 'r') as file:
            names_to_remove = [line.strip() for line in file]

        # Check if the specified column exists in the DataFrame
        if column_name not in dataframe.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

        # Remove rows containing names from the DataFrame
        dataframe = dataframe[~dataframe[column_name].isin(names_to_remove)]

        return dataframe
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Carregamento das bases de nomes comuns, em formato .csv:
nomes_masc = r'C:\Users\gabriel.silva\Downloads\ibge-mas-10000.csv'
nomes_fem = r'C:\Users\gabriel.silva\Downloads\ibge-fem-10000.csv'

# Retirada de nomes Masculinos:
amostras_df = remove_names_from_dataframe(nomes_masc,amostras_df,'EHR')

# Retirando os nomes Femininos:
amostras_df = remove_names_from_dataframe(nomes_fem,amostras_df,'EHR')

stopWordNLTK = nltk.corpus.stopwords.words('portuguese')
def remove_stop_words(df):
    evolucao = df['EHR']
    cid = df['CID']

    if pd.isna(evolucao) or evolucao.strip() == '':
        return pd.Series([None, cid], index=['EHR', 'CID'])

    semStopWord = ' '.join([p for p in evolucao.split() if p not in stopWordNLTK])
    return pd.Series([semStopWord, cid], index=['EHR', 'CID'])

amostrasSemStopWords = amostras_df.apply(remove_stop_words, axis=1)

def aplicaStemmer(df):
    stemmer = nltk.stem.RSLPStemmer()
    evolucao = df['EHR']
    cid = df['CID']

    evolComStem = ' '.join([str(stemmer.stem(p)) for p in evolucao.split()])
    return pd.Series([evolComStem, cid], index=['EHR', 'CID'])

amostrasComStemmer = amostrasSemStopWords.apply(aplicaStemmer, axis=1)
# print(amostrasComStemmer)

def aplicaTokenizer(df):
    tokenizer = nltk.word_tokenize
    evolucao = df['EHR']
    cid = df['CID']

    evolComTok = ' '.join([str(tokenizer(p,language='portuguese')) for p in evolucao.split()])
    return pd.Series([evolComTok, cid], index=['EHR', 'CID'])

amostrasComTok = amostrasComStemmer.apply(aplicaTokenizer, axis=1)
#print(amostrasComTok)

# Divisão das amostras:
X = amostrasComTok['EHR']
y = amostrasComTok['CID']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
f1 = f1_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

print(f'Acurácia: {accuracy}')
print(f'Precisão (Precision): {precision}')
print(f'Revocação (Recall): {recall}')
print(f'F1-Score: {f1}')