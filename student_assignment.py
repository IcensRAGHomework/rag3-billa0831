from datetime import datetime
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
    
    if collection.count() != 0:
        return collection
    # 讀取 CSV 檔案
    df = pd.read_csv("COA_OpenData.csv")

    # 將資料轉換為 ChromaDB 可以接受的格式
    documents = []
    metadatas = []
    ids = []

    # 處理每一條資料並儲存
    for idx, row in df.iterrows():
        # 擷取 Metadata
        metadata = {
            "file_name": "COA_OpenData.csv",
            "name": row["Name"],
            "type": row["Type"],
            "address": row["Address"],
            "tel": row["Tel"],
            "city": row["City"],
            "town": row["Town"],
            "date": int(datetime.strptime(row["CreateDate"], "%Y-%m-%d").timestamp())  # 將 CreateDate 轉為時間戳
        }

        # 擷取 HostWords 作為文件內容
        document = row["HostWords"]
        
        # 存入資料
        documents.append(document)
        metadatas.append(metadata)
        ids.append(str(idx))  # 每條資料的唯一 ID
    # 將資料寫入 ChromaDB
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection
def generate_hw01_rr():
    df = pd.read_csv(file_name)
    required_columns={"ID", "Name", "Type", "Address", "Tel", "CreateDate", "HostWords"}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV 缺少必要欄位！請確認 CSV 欄位名稱是否正確。")
    city_pattern = re.compile(r"^(.*?[市縣])")
    town_pattern = re.compile(r"^(.*?[市縣].*?[區鄉市])")
    town_pattern2 = re.compile(r"^(.*?[市縣].*?[鎮])")
    documents = df["HostWords"].tolist()
    metadata_list = []
    ids = []
    for index, row in df.iterrows():
        try:
            create_timestamp = int(time.mktime(pd.to_datetime(row["CreateDate"]).timetuple()))
        except:
            create_timestamp = int(time.time())  # 若日期轉換失敗，則使用當前時間
        city_str = re.match(city_pattern, row["Address"]).group()

        if re.match(town_pattern, row["Address"]) is not None:
            town_str = re.match(town_pattern, row["Address"]).group().split(city_str)[1]
        elif re.match(town_pattern2, row["Address"]) is not None:
            town_str = re.match(town_pattern2, row["Address"]).group().split(city_str)[1]
        else:
            town_str = city_str
            city_str = city_str.replace("市", "縣")
        
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
        ids.append(str(index))  # 以 index 作為 ID
    # 將數據存入 ChromaDB
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadata_list
    )
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