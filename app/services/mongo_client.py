from pymongo import MongoClient
import os

def get_mongo_collection():
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri)
    db = client["batiknusantara"]
    return db["predictions"]
