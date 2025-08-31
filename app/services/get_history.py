from .mongo_client import get_mongo_collection

def get_history():
    collection = get_mongo_collection()
    docs = collection.find().sort("timestamp", -1)
    return list(docs)
