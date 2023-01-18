import argparse
from asyncio.log import logger
from collections import Counter
from itertools import combinations, chain
import logging
import os
import pickle
import re
from time import time

from azureml.core import Dataset, Datastore, Model, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.data.datapath import DataPath
from bs4 import BeautifulSoup
from flair.data import Sentence
from flair.models import SequenceTagger
from geopy.geocoders import Nominatim
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
from pyvis.network import Network
import psycopg2
from sentence_transformers import SentenceTransformer, util
import shapefile
from shapely.geometry import Point, shape
import spacy
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
)

from ensemble_model import *

parser = argparse.ArgumentParser(description='Run models')
parser.add_argument(
    '--full', dest='full', default=False, action='store_true')
parser.add_argument(
    '--districs', dest='districs', default=False, action='store_true')
parser.add_argument(
    '--topics', dest='topics', default=False, action='store_true')
parser.add_argument(
    '--keywords', dest='keywords', default=False, action='store_true')
parser.add_argument(
    '--duplicates', dest='duplicates', default=False, action='store_true')
parser.add_argument(
    '--comments', dest='comments', default=False, action='store_true')
parser.add_argument(
    '--topic-clf-name',
    dest='topics_clf_name',
    help='Name of the model in Azure ML models.'
)
parser.add_argument(
    '--sentence-transformer-model',
    dest='sentence_transformer_model',
    action='store',
    default='hiiamsid/sentence_similarity_spanish_es',
    help='Name of the model in Azure ML models.'
)
parser.add_argument(
    '--polarity-model',
    dest='polarity_model',
    action='store',
    default='pysentimiento/robertuito-sentiment-analysis',
    help='Name of the model in Azure ML models.'
)
parser.add_argument(
    '--hate-model',
    dest='hate_model',
    action='store',
    default='pysentimiento/robertuito-hate-speech',
    help='Name of the model in Azure ML models.'
)
parser.add_argument(
    '--spacy-model',
    dest='spacy_model',
    action='store',
    default='es_core_news_lg',
    help='Name of the model in Azure ML models.'
)
parser.add_argument(
    '--tagger-model',
    dest='tagger_model',
    action='store',
    default='ner-multi',
    help='Name of the model in Azure ML models.'
)
parser.add_argument(
    '--threshold-pred-title',
    dest='threshold_pred_title',
    action='store',
    default=0.95,
    help='Name of the model in Azure ML models.'
)
parser.add_argument(
    '--threshold-pred-summary',
    dest='threshold_pred_summary',
    action='store',
    default=0.95,
    help='Name of the model in Azure ML models.'
)
parser.add_argument(
    '--datastore',
    dest='datastore',
    action='store',
    default=None,
    help='Name of the model in Azure ML models.'
)
parser.add_argument(
    '--nominatim-user-agent',
    dest='nominatim_user_agent',
    action='store',
    default=None,
    help='Name of the model in Azure ML models.'
)
parser.add_argument(
    '--created-at-ago',
    dest='created_at_ago',
    action='store',
    default=None,
    help='24 HOURS, 2 DAYS, ...'
)

args = parser.parse_args()

sentence_transformer = None
topics_clf = None
area_clf = None
hate_tokenizer = None
hate_model = None
nlp = None
polarity_tokenizer = None
polarity_model =  None
tagger = None
threshold_pred_title = None
threshold_pred_summary = None
subtopics2areas = None
subtopics2topics = None

workspace = None
datastore = None
conn = None
cur = None

distritos = shapefile.Reader("data/Distritos.zip")
transformer = pyproj.Transformer.from_crs(4326, 25830)
geolocator = None


device = "cpu"

logging.warning(f'DEVIDE: {device}')

def logging_execution_time(func):
    """
    Wrapper to log the execution time
    """
    def wrapper_function(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        logging.warning(
            f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrapper_function


@logging_execution_time
def get_areas_pred(subtopics, subtopics_pred):
    """
    Map subtupics to goverment areas
    """
    keys = subtopics if subtopics else subtopics_pred
    areas = []
    if not keys:
        return ""
    for k in keys.split(" "):
        logger.warning(k)
        try:
            areas.append(
                subtopics2areas[subtopics2areas["subtopic"] == k]["area"].iloc[0])
        except Exception as e:
            logger.warning(e)
            continue
    return remove_accents(" ".join(set(areas)).replace("'", ""))


@logging_execution_time
def get_topics_pred(subtopics, subtopics_pred):
    """
    Map subtupics to topic
    """
    keys = subtopics if subtopics else subtopics_pred
    topics = []
    if not keys:
        return ""
    for k in keys.split(" "):
        logger.warning(k)
        try:
            topics.append(
                subtopics2topics[subtopics2topics["subtopic"] == k]["topic"].iloc[0])
        except Exception as e:
            logger.warning(e)
            continue
    return remove_accents(" ".join(set(topics)).replace("'", ""))


@logging_execution_time
def get_subtopics_pred(raw_texts):
    """
    Predict the subtopic from texts using a pretrained ML model
    """
    texts = [raw_text.lower() for raw_text in raw_texts]
    pred = topics_clf.predict(texts)
    topic_list = pred.apply(lambda x: remove_accents(" ".join(list(x.index[x.astype(bool)]))), axis=1)
    return topic_list


# REFACTORIZAR
@logging_execution_time
def get_coocurrence_graph(texts):
    """
    Generate the coocurrence graph from multiple texts
    """
    try:
        if texts.empty:
            return [], []      
        tokens = texts.apply(
            lambda t: get_keywords_pred(str(t), remove_duplicates=False)
        )
        MG = nx.MultiGraph()
        MG.add_nodes_from([(
            c[0], {'count': c[1]}) 
            for c in Counter(chain(*tokens.to_list())).items()
        ])
        for c in tokens:
            MG.add_edges_from(combinations(c, 2))
        G = nx.Graph()
        G.add_nodes_from(MG)
        G.add_edges_from(
            (u, v, {'weight': value}) 
            for ((u, v), value) in Counter(MG.edges(G.nodes)).items()
        )
        nodes = set([
            k for k, _ in sorted(
                nx.pagerank_numpy(G, weight='count').items(),
                key=lambda item: item[1], reverse=True)][:50]
        )
        comm = list(nx.algorithms.community.louvain_communities(G, weight='weight'))
        comm = [nodes.intersection(c) for c in comm]
        edges = [list(combinations(c, 2)) for c in comm]
        Gd = G.edge_subgraph(chain(*edges))
        def f(n):
            d = {k:c for k, c in nx.get_edge_attributes(Gd, 'weight').items() if n in k}
            out_ = [k for k, c in d.items() if c == max(d.values())]   
            return out_
        edges = [f(n) for n in set(nodes) & set(Gd.nodes)]
        Gd = G.edge_subgraph(chain(*edges))
        nx.set_node_attributes(
            Gd, {k:np.sqrt(v) for k, v in nx.get_node_attributes(MG, 'count').items() if k in nodes}, 'size')
        nx.set_node_attributes(
            Gd, {k:f"Apariciones: {v}" for k, v in nx.get_node_attributes(MG, 'count').items() if k in nodes}, 'title')
            
        for i, c in enumerate(comm):
            nx.set_node_attributes(Gd, {k:i for k in c}, 'group')
        nt = Network()
        nt.from_nx(Gd)
    except Exception as e:
        logging.warning(e)
        return "", ""
    return str(nt.nodes).replace("'", '"'), str(nt.edges).replace("'", '"')


def get_district_name(point):
    """
    Get the district name from coordinates
    """
    transformed_point = Point(
        transformer.transform(point.latitude, point.longitude)
    )
    for s in distritos.iterShapeRecords():
        if transformed_point.within(shape(s.shape)):
            return s.record['DECIDEML']
        
    return None


def get_geocode(text, suffix=", Área metropolitana de Madrid"):
    """
    Try to get the location from a text. To do this, a request to a online
    service is made
    """
    return geolocator.geocode(text + suffix)


@logging_execution_time
def get_districts(raw_text, location=None):
    """
    Get a list of district names from a text and coordinates
    """
    if location:
        point = eval(location)
        transformed_point = Point(
            transformer.transform(point[0], point[1])
        )
        for s in distritos.iterShapeRecords():
            if transformed_point.within(shape(s.shape)):
                return s.record['DECIDEML']
    text = raw_text.lower()
    district_names = []
    try:
        if not text:
            logging.warning("{funcName}: Empty text.")
            return None
        for text_location in get_named_entities(text, tags=['LOC']):
            location = get_geocode(text_location)
            
            if location is None and 'calle' in text_location:
                location = get_geocode(
                    str(text_location).replace(
                        'calle', 'calle de').replace(
                            'Calle', 'Calle de'
                        )
                    )
            
            if location:
                district_name = get_district_name(location.point)
                
                if district_name is None:
                    logging.warning(
                        f'Point {location.point} not found within district bounds'
                    )
                else:
                    district_names.append(district_name) 
    except:
        logging.warning("{funcName}: Max retries nominatim.")
            
    result = " ".join(d.upper().replace(" ", "_") for d in district_names)
    return remove_accents(result)


@logging_execution_time
def get_named_entities(raw_text, tags=None):
    """
    Extract the name entities form a texts. IF tags is defined, this named
    entities are filtered by label
    """
    if not raw_text:
        logging.warning("{funcName}: Empty text.")
        return []
    
    doc = nlp(raw_text)
    if tags is None:
        result = [e.text.lower() for e in doc.ents if e.label != 'MISC']
    else:
        result = [e.text.lower() for e in doc.ents if e.label in tags]
    return result


@logging_execution_time
def get_keywords_pred(raw_text, remove_duplicates=True):
    """
    Extract the keywords from a text. If remove_duplicates, the duplicated 
    keywords are removed
    """
    text = raw_text.lower()
    if not text:
        logging.info("{funcName}: Empty text.")
        return []
    f = set if remove_duplicates else list
    doc = nlp(text)
    tokens = f(
        [
            l.lower() 
            for s in doc.sents 
            for t in s if t.pos_ in ["NOUN", "PROPN", "VERB"] and \
                not t.text.startswith("http") and 
                not t.is_stop and t.is_alpha
            for l in t.lemma_.split()
        ]
    )
    return tokens


CHUNK_SIZE = 128

@logging_execution_time
def get_polarity_pred(texts):
    """
    Predict the polarity of a text using a pretrained model
    """
    result = []
    for x in range(0,len(texts), CHUNK_SIZE):
        out_ = polarity_model(**polarity_tokenizer.batch_encode_plus(
            texts[x:x+CHUNK_SIZE].to_list(),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=64
        ).to(device))
        aux = out_.logits.argmax(axis=1) * .5
        result += aux.detach().to('cpu').tolist()

    return result


@logging_execution_time
def get_hate_pred(texts):
    """
    Predict the hate of a text using a pretrained model
    """
    result = []
    for x in range(0,len(texts), CHUNK_SIZE):
        out_ = hate_model(**hate_tokenizer.batch_encode_plus(
            texts[x:x+CHUNK_SIZE].to_list(),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=64
        ).to(device))
        aux = np.delete(
            out_.logits.sigmoid().detach().to('cpu').numpy(), 2, axis=1)
        result += aux.tolist()

    return result

@logging_execution_time
def init_azure():
    """
    Initialize azure connection using enviroment variables
    """
    global workspace
    global datastore


    sp = ServicePrincipalAuthentication(tenant_id=os.getenv("AZURE_TENANT_ID"), # tenantID
                                    service_principal_id=os.getenv("AZURE_SP_ID"), # clientId
                                    service_principal_password=os.getenv("AZURE_SP_PASSWORD")) # clientSecret

    workspace = Workspace(os.getenv("AZURE_SUBSCRIPTION_ID"), os.getenv("AZURE_RESOURCE_GROUP"), os.getenv("AZURE_ML_WORKSPACE"), auth=sp)
    # workspace = Workspace.from_config()
    datastore = Datastore(workspace, args.datastore)
    connect_to_db()


def connect_to_db():
    """
    Connect to db
    """
    global conn
    global cur
    conn = psycopg2.connect(f"""
        dbname='{datastore.database_name}'    
        user='{datastore.user_id}' 
        password='{datastore.user_password}' 
        host='{datastore.server_name}.postgres.database.azure.com' 
        password='{datastore.user_password}'"""
    )
    cur = conn.cursor()   


def retry_conn_once(f):
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            connect_to_db()
            return f(*args, **kwargs)
            
    return wrapper

def init_models():
    """
    Initialize models 
    """
    global topics_clf
    global geolocator
    global nlp
    global hate_model
    global hate_tokenizer
    global polarity_model
    global polarity_tokenizer
    global sentence_transformer
    global tagger
    global threshold_pred_title
    global threshold_pred_summary
    global subtopics2areas
    global subtopics2topics

    topics_clf_dir = Model(
        workspace, name=args.topics_clf_name).download(exist_ok=True)
    topics_clf = pickle.load(        
        open(topics_clf_dir, "rb")
    )
    spacy.cli.download(args.spacy_model)
    nlp = spacy.load(args.spacy_model)
    polarity_tokenizer = AutoTokenizer.from_pretrained(
        args.polarity_model
    )
    polarity_model = AutoModelForSequenceClassification.from_pretrained(
        args.polarity_model
    ).to(device)
    hate_tokenizer = AutoTokenizer.from_pretrained(
        args.hate_model
    )
    hate_model = AutoModelForSequenceClassification.from_pretrained(
        args.hate_model
    ).to(device)
    tagger = SequenceTagger.load(args.tagger_model)
    sentence_transformer = SentenceTransformer(args.sentence_transformer_model)
    threshold_pred_title = args.threshold_pred_title
    threshold_pred_summary = args.threshold_pred_summary
    subtopics2areas = pd.read_csv("subtopics2areas.csv")
    subtopics2topics = pd.read_csv("subtopics2topics.csv")
    logging.warning(subtopics2areas)    
    logging.warning("--------------------------")
    logging.warning(subtopics2topics)
    geolocator = Nominatim(user_agent=args.nominatim_user_agent)


@retry_conn_once
def fetch_table(name, **kwargs):
    """
    Fetch data from table name
    """
    query = f'SELECT * FROM {name}'
    if kwargs:
        where = " AND ".join([
            f"{k} = '{kwargs[k]}'"
            if k != "created_at" else
            f' created_at >= NOW() - INTERVAL \'{kwargs["created_at"]}\''
            for k in kwargs if kwargs[k]
        ])
        if where:
            query += f" WHERE {where}"
    query = DataPath(datastore, query)
    tabular = Dataset.Tabular.from_sql_query(query)
    return tabular

@retry_conn_once
def update_table(name, df, id_column="id"):
    """
    Update table with the dataframe
    """
    def format_query(r):
        updates = ", ".join(
            f"""{k} = '{str(v).replace("'", "''")}'""" 
            for k, v in r.drop(id_column).to_dict().items() if v
        )
        if not updates:
            return None
        query = f"""
            UPDATE {name}
            SET {updates}
            WHERE {id_column} = {r[id_column]}
        """
        return query
    def execute_query(r):
        query = format_query(r)
        logging.warning(query)
        if not query:
            return None
        try:
            cur.execute(query)
        except:
            logging.warning(f'Query error: {query}')
    df.apply(execute_query, axis=1)
    conn.commit()

@retry_conn_once
def upsert_table(table, df, constraint_name):
    """
    Upsert table with the dataframe
    """
    def format_query(r):
        names = "(" + ", ".join(r.index.to_list()) + ")"
        values = str(tuple(r.values))
        query = f"""
            INSERT INTO {table} {names} VALUES {values}
            ON CONFLICT ON CONSTRAINT {constraint_name}
            DO NOTHING;
        """
        logging.warning(query)
        return query
    def execute_query(r):
        logging.warning(f"ROW: {r}")
        query = format_query(r)
        if not query:
            return None
        cur.execute(query)
    df.apply(execute_query, axis=1)
    conn.commit()


def get_beto_vector(text):
    """
    Generate the vector representation of a text
    """
    tokens = sentence_transformer.encode(text)
    return pickle.dumps(tokens)


def remove_accents(raw_text):
    if not pd.notna(raw_text):
        return ""
    raw_text = re.sub(u"[àáâãäå]", 'a', raw_text)
    raw_text = re.sub(u"[èéêë]", 'e', raw_text)
    raw_text = re.sub(u"[ìíîï]", 'i', raw_text)
    raw_text = re.sub(u"[òóôõö]", 'o', raw_text)
    raw_text = re.sub(u"[ùúûü]", 'u', raw_text)
    raw_text = re.sub(u"[ñ]", 'n', raw_text)
    return raw_text 


def preprocess_text(df, columns=['title']):
    """
    Clean the text (remove html tags, urls, ...)
    """
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


def get_table(name='', full_text_columns=['title', ], **kwargs):
    try:
        tabular = fetch_table(name, **kwargs)
    except:
        return pd.DataFrame(columns=['id', 'full_text'])
    df = preprocess_text(
        tabular.to_pandas_dataframe(), columns=full_text_columns)
    return df


@logging_execution_time
def predict_comments():
    df = get_table(
        name="comments", full_text_columns=['description'],
        created_at=args.created_at_ago)
    if df.empty:
        return
    columns = ['id']
    if args.full or args.keywords:
        df['named_entities_pred'] = \
            df['full_text'].apply(  
                lambda t: " ".join(w.replace(" ", "_") for w in get_named_entities(t)).lower())
        df['keywords_pred'] = \
            df['full_text'].apply(
                lambda t: " ".join(w.replace(" ", "_") for w in get_keywords_pred(t)).lower())
        columns += ['named_entities_pred', 'keywords_pred',]
        update_table('comments', df[columns])
    if args.full or args.comments:
        df['polarity_pred'] = get_polarity_pred(df['full_text'])
        update_table('comments', df[columns])
        df[['hate_pred', 'offensive_pred']] = get_hate_pred(df['full_text'])
        update_table('comments', df[columns])
        columns += ['polarity_pred', 'hate_pred', 'offensive_pred']


@logging_execution_time
def predict_entities(name, type_id, full_text_columns=['title']):
    df = get_table(
        name=name, full_text_columns=full_text_columns,
        created_at=args.created_at_ago)
    if df.empty:
        return
    columns = ['id']
    if args.full or args.districs:
        if 'location' in df.columns:
            df['districts_pred'] = df[['full_text', 'location']].apply(
                lambda r: get_districts(r['full_text'], location=r['location']),
                axis=1
            )
        else:
            df['districts_pred'] = df['full_text'].apply(
                get_districts
            )
        columns += ['districts_pred', ]
        update_table(name, df[columns])            
    if args.full or args.topics:
        df['subtopics_pred'] = get_subtopics_pred(df['full_text'].values)
        df['subtopics'] = df['subtopics_pred'].apply(remove_accents)
        df['topics_pred'] = df[['subtopics', 'subtopics_pred']].apply(
            lambda r: get_topics_pred(*r), axis=1
        )
        df['topics'] = df['topics_pred']
        df['government_areas_pred'] = df[['subtopics', 'subtopics_pred']].apply(
            lambda r: get_areas_pred(*r), axis=1
        )
        df['government_areas'] = df['government_areas_pred']
        columns += ['government_areas_pred', 'topics_pred', 'government_areas', 
            'topics', 'subtopics', 'subtopics_pred']
        update_table(name, df[columns])
    if args.full or args.keywords:
        df['named_entities_pred'] = \
            df['full_text'].apply(  
                lambda t: " ".join(w.replace(" ", "_") for w in get_named_entities(t)).lower())
        df['keywords_pred'] = \
            df['full_text'].apply(
                lambda t: " ".join(w.replace(" ", "_") for w in get_keywords_pred(t)).lower())
        df[['nodes', 'edges']] = df.apply(
            lambda r: get_coocurrence_graph(
                get_table(
                    name="comments", full_text_columns=['description', ],
                    parent_id=r['id'], parent_type_id=type_id
                )['full_text']
            ), axis=1, result_type='expand'
        )
        columns += ['named_entities_pred', 'nodes', 'edges', 'keywords_pred']
        update_table(name, df[columns])

@logging_execution_time
def update_duplicates():
    def calculate_beto_vector(name, full_text_columns=[]):
        df = get_table(
            name="proposals", full_text_columns=full_text_columns,
            created_at=args.created_at_ago)
        df['beto_vector'] = df['full_text'].apply(get_beto_vector)
        columns = ['id', 'beto_vector']
        update_table(name, df[columns])
    for n in ['proposals', 'debates']:
        calculate_beto_vector(n, full_text_columns=['title', 'summary'])
    df_p = get_table(
        name="proposals", created_at=args.created_at_ago)[['id', 'beto_vector']]
    df_p['type_id'] = 'PROPUESTA'
    df_d = get_table(
        name="debates", created_at=args.created_at_ago)[['id', 'beto_vector']]
    df_d['type_id'] = 'DEBATE'
    df = pd.concat([df_p, df_d])
    df['beto_vector'] = df['beto_vector'].apply(
        lambda s: pickle.loads(eval(s)) if type(s) is str else np.nan)
    df = df.dropna() 
    duplicates = []
    for x in df[['id', 'type_id', 'beto_vector']].itertuples():
        aux = pd.DataFrame(
            df.apply(
                lambda t: (
                    t['id'], t['type_id'],
                    util.pytorch_cos_sim(x[3], t['beto_vector']).numpy()[0][0]
                ),
                axis=1
            ).tolist(),
            columns=('duplicate_id', 'duplicate_type_id', 'probability')
        ).sort_values('probability', ascending=False).iloc[1:6]
        aux[['entity_id', 'entity_type_id']] = (x[1], x[2])
        duplicates.append(aux)
    df_duplicates = pd.concat(duplicates)
    upsert_table('duplicates', df_duplicates, 'entities_duplicate_pairs')


def main():
    init_azure()
    init_models()

    predict_comments()
    predict_entities(
        'proposals', 'PROPUESTA', full_text_columns=['title', 'summary'])
    predict_entities(
        'debates', 'DEBATE', full_text_columns=['title', 'summary'])
    predict_entities(
        'participatory_budgets', 'PRESUPUESTO',
        full_text_columns=['title', 'description'])
    if args.full or args.duplicates:
        update_duplicates()


if __name__ == '__main__':
    main()
