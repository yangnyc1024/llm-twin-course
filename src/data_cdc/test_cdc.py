from pymongo import MongoClient  # 从 pymongo 导入 MongoDB 客户端类


def insert_data_to_mongodb(uri, database_name, collection_name, data):
    """
    向 MongoDB 的指定集合中插入一条数据。

    参数说明:
    :param uri: MongoDB 连接字符串（包含主机、端口、副本集等信息）
    :param database_name: 数据库名称
    :param collection_name: 集合名称
    :param data: 要插入的数据（Python 字典类型）
    """
    # 1️⃣ 创建 MongoDB 客户端，连接到数据库集群
    client = MongoClient(uri)

    # 2️⃣ 获取指定数据库
    db = client[database_name]

    # 3️⃣ 获取指定集合
    collection = db[collection_name]

    try:
        # 4️⃣ 插入一条文档（data 必须是 dict）
        result = collection.insert_one(data)

        # 5️⃣ 打印插入成功的文档 ID（MongoDB 会自动生成 _id）
        print(f"Data inserted with _id: {result.inserted_id}")

    except Exception as e:
        # 6️⃣ 捕获并打印异常信息
        print(f"An error occurred: {e}")

    finally:
        # 7️⃣ 无论是否成功，都要关闭数据库连接，防止资源泄露
        client.close()


# 8️⃣ 当文件被直接执行时，运行测试插入逻辑
if __name__ == "__main__":
    # 调用函数，连接本地 MongoDB 副本集，并插入一条测试数据
    insert_data_to_mongodb(
        "mongodb://localhost:30001,localhost:30002,localhost:30003/?replicaSet=my-replica-set",  # MongoDB 副本集连接地址
        "twin",             # 数据库名称
        "posts",            # 集合名称
        {"platform": "linkedin", "content": "Test content"}  # 要插入的数据
    )
