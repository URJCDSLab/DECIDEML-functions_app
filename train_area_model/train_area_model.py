import argparse
import os
import pickle
import re
from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
import psycopg2
import sklearn
from azureml.core import Dataset, Datastore, Model, Workspace
from azureml.data.datapath import DataPath
from azureml.core.authentication import ServicePrincipalAuthentication
from bs4 import BeautifulSoup
from ensemble_model import EnsembleModel
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


parser = argparse.ArgumentParser(description='Run models')
parser.add_argument(
    '--topic-clf-name',
    dest='topics_clf_name',
    help='Name of the model in Azure ML models.'
)

parser.add_argument(
    '--datastore',
    dest='datastore',
    action='store',
    default=None,
    help='Name of the model in Azure ML models.'
)

args = parser.parse_args()

sp = ServicePrincipalAuthentication(tenant_id=os.getenv("AZURE_TENANT_ID"), # tenantID
                                    service_principal_id=os.getenv("AZURE_SP_ID"), # clientId
                                    service_principal_password=os.getenv("AZURE_SP_PASSWORD")) # clientSecret

workspace = Workspace(os.getenv("AZURE_SUBSCRIPTION_ID"), os.getenv("AZURE_RESOURCE_GROUP"), os.getenv("AZURE_ML_WORKSPACE"), auth=sp)
datastore = Datastore(workspace, args.datastore)

def preprocess_text(df, columns=['title']):
    df = df.copy()
    df['full_text'] = df[columns].apply(" ".join, axis=1)
    df['full_text'] = df['full_text'].apply(
        lambda t: BeautifulSoup(t, "lxml").text)
    df['full_text'] = df['full_text'].apply(
        lambda s: re.sub(r'[\W_]+', ' ', str(s)))
    df['full_text'] = df['full_text'].apply(
        lambda s: re.sub(r'http\S+', ' ', s))
    df['full_text'] = df['full_text'].apply(
        lambda s: re.sub(r'\s+[a-zA-Z]\s+', ' ', s))
    df['full_text'] = df['full_text'].apply(
        lambda s: re.sub(r'\s+', ' ', s, flags=re.I))
    return df

def preprocess_topics(df, max_topics=1000, min_occurrencies=1):
    c = [f'topic{x}' for x in range(1, 12)]
    c.append('topic')
    
    df = df.copy()
    df['topics'] = df.loc[:, c].values.tolist()
    df['topics'] = df.topics.apply(lambda l: [x for x in l if type(x) is str])
    
    c = Counter(chain.from_iterable(df.topics.to_list())).most_common(max_topics)
    mc = set(k for k, v in c if v > min_occurrencies)
    df['topics'] = df['topics'].apply(lambda t:  set(t) & mc)
    return df

def fetch_table(name, **kwargs):
    query = f'SELECT * FROM {name}'
    if kwargs:
        where = " AND ".join([
            f"{k} = '{kwargs[k]}'" 
            if k != "created_at" else
            " created_at >= NOW - INTERVAL '{kwargs[created_at]}'"
            for k in kwargs if kwargs[k] 
        ])
        if where:
            query += f" WHERE {where}"
    query = DataPath(datastore, query)
    tabular = Dataset.Tabular.from_sql_query(query)
    return tabular

def get_table(name='', full_text_columns=[], **kwargs):
    try:
        tabular = fetch_table(name, **kwargs)
    except:
        return pd.DataFrame(columns=['id', 'full_text'])
    df = preprocess_text(
        tabular.to_pandas_dataframe(), columns=full_text_columns)
    return df

working_attributes = ['full_text', 'topics']


budgets = get_table(
        name="participatory_budgets",
        full_text_columns=['title', 'description'])[working_attributes]

debates = get_table(
        name="debates", full_text_columns=['title', 'summary'])[working_attributes]

proposals = get_table(
        name="proposals", full_text_columns=['title', 'summary'])[working_attributes]

df = pd.concat((budgets, debates, proposals))
df_prep = preprocess_topics(df.copy(), min_occurrencies=2)

stop_words = set(stopwords.words('spanish')) | set(['madrid', 'barrio', 'calle', 'barrio', 'la', 'el', 'mucho'])
X, y = df_prep['full_text']. df_prep['topics']

pipeline  = Pipeline([
   ('count_vectorizer', CountVectorizer(min_df=5, encoding='utf8', ngram_range=(1, 2), stop_words=stop_words)),
   ('tfidf', TfidfTransformer(use_idf=False)),
   ('model', EnsembleModel())   
])

pipeline.fit(X, y)

pickle.dump(pipeline, open('model.pickle', 'wb'))

azure_model = Model.register(workspace=workspace,
                             model_path='model.pickle',
                             model_name=args.topics_clf_name,
                             model_framework=Model.Framework.SCIKITLEARN,
                             model_framework_version=sklearn.__version__,
                             description='Predicts the govern area of a summary',
                             tags={'area': 'area', 'type': 'classification'})
