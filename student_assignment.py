from datetime import datetime
import datetime
import chromadb
import traceback
import pandas as pd
import re
from chromadb.utils import embedding_functions


from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_type=gpt_emb_config['openai_type'],
        api_base=gpt_emb_config['api_base'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
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
    required_columns = {"ID", "Name", "Type", "Address", "Tel", "CreateDate", "HostWords"}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV 缺少必要欄位！請確認 CSV 欄位名稱是否正確。")
    city_pattern = re.compile(r"^(.*?[市縣])")
    town_pattern = re.compile(r"^(.*?[市縣].*?[區鄉市])")
    town_pattern2 = re.compile(r"^(.*?[市縣].*?[鎮])")
    documents = df["HostWords"].tolist()
    metadata_list = []
    ids = []
    for index, row in df.iterrows():
        # try:
        create_timestamp = int(datetime.strptime(row["CreateDate"], "%Y-%m-%d").timestamp())
        # except:
        #     create_timestamp = int(time.time())  # 若日期轉換失敗，則使用當前時間
        city_str = re.match(city_pattern, row["Address"]).group()

        if re.match(town_pattern, row["Address"]) is not None:
            town_str = re.match(town_pattern, row["Address"]).group().split(city_str)[1]
        elif re.match(town_pattern2, row["Address"]) is not None:
            town_str = re.match(town_pattern2, row["Address"]).group().split(city_str)[1]
        else:
            town_str = city_str
            city_str = city_str.replace("市", "縣")

        metadata = {
            "file_name": "COA_OpenData.csv",
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
    # 初始化 ChromaDB 客戶端並獲取 TRAVEL 集合
    client = chromadb.PersistentClient(path="./")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_type=gpt_emb_config['openai_type'],
        api_base=gpt_emb_config['api_base'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    collection = client.get_collection(name="TRAVEL", embedding_function=openai_ef)

    # 構建主要過濾條件（這裡以 city 為主條件）
    where_conditions = {}
    if city and len(city) > 0:
        where_conditions = {"city": {"$in": city}}
    # 如果沒有 city，則不設置 where，查詢所有數據

    # 執行查詢
    results = collection.query(
        query_texts=[question],  # 用戶問題作為查詢文本
        n_results=10,  # 返回最多 10 筆結果
        where=where_conditions if where_conditions else None  # 只用 city 作為主要過濾
    )

    # 提取結果並手動過濾其他條件
    store_names = []
    distances = results["distances"][0]  # 距離列表
    metadatas = results["metadatas"][0]  # Metadata 列表

    for i, distance in enumerate(distances):
        similarity_score = 1 - distance  # 轉換為相似度分數
        print(similarity_score)
        metadata = metadatas[i]

        # 手動過濾 store_type
        if store_type and len(store_type) > 0 and metadata["type"] not in store_type:
            continue

        # 手動過濾日期範圍
        store_date = int(metadata["date"])
        if start_date and store_date < int(start_date.timestamp()):
            continue
        if end_date and store_date > int(end_date.timestamp()):
            continue

        # 過濾相似度 >= 0.80
        if similarity_score >= 0.80:
            store_names.append((metadata["name"], similarity_score))

    # 按相似度分數遞減排序
    store_names.sort(key=lambda x: x[1], reverse=True)

    # 只保留店家名稱
    final_result = [name for name, score in store_names]

    return final_result
    
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
# generate_hw01()
print(generate_hw02("我想要找有關茶餐點的店家", ["宜蘭縣", "新北市"], ["美食"], datetime.datetime(2024, 4, 1), datetime.datetime(2024, 5, 1)))
print("finish")