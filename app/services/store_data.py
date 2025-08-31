from .mongo_client import get_mongo_collection

def store_data(data):
    collection = get_mongo_collection()
    result = collection.insert_one(data)
    return str(result.inserted_id)
