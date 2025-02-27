import datetime
import pprint
import time
from langchain.docstore.document import Document
import chromadb
import traceback
import pandas as pd
import re
from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
chroma_client = chromadb.PersistentClient(path=dbpath)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key = gpt_emb_config['api_key'],
    api_base = gpt_emb_config['api_base'],
    api_type = gpt_emb_config['openai_type'],
    api_version = gpt_emb_config['api_version'],
    deployment_id = gpt_emb_config['deployment_name']
)
collection = chroma_client.get_or_create_collection(
    name="TRAVEL",
    metadata={"hnsw:space": "cosine"},
    embedding_function=openai_ef
)
file_name = "COA_OpenData.csv"

def generate_hw01():
    df = pd.read_csv(file_name)
    required_columns={"ID", "Name", "Type", "Address", "Tel", "CreateDate", "HostWords"}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV 缺少必要欄位！請確認 CSV 欄位名稱是否正確。")
    # print(df["HostWords"])
    city_pattern = re.compile(r"^(.*?[市縣])")
    town_pattern = re.compile(r"^(.*?[市縣].*?[區鄉鎮市])")
    documents = df["HostWords"].tolist()
    metadata_list = []
    ids = []
    for index, row in df.iterrows():
        try:
            create_timestamp = int(time.mktime(pd.to_datetime(row["CreateDate"]).timetuple()))
        except:
            create_timestamp = int(time.time())  # 若日期轉換失敗，則使用當前時間
        # address_split_list = re.split(city_pattern, row["Address"])
        city_str = re.match(city_pattern, row["Address"]).group()
        print(city_str)

        if re.match(town_pattern, row["Address"]) is not None:
            # print(re.match(town_pattern, row["Address"]).group())
            town_str = re.match(town_pattern, row["Address"]).group().split(city_str)[1]
        
        print(str(index) + " "+town_str)
        
        metadata = {
            "file_name": file_name,
            "name": row["Name"],
            "type": row["Type"],
            "address": row["Address"],
            "tel": row["Tel"],
            "city": city_str,
            "town": town_str,
            "date": create_timestamp
        }

        metadata_list.append(metadata)
        ids.append(row["ID"])  # 以 index 作為 ID
    # 將數據存入 ChromaDB
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadata_list
    )
    
    print(collection.metadata)
    return collection
    # pass
    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection


print(generate_hw01())
print("finish")