import os
import cohere
import pymongo
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

# 1. Initialize Cohere Client using env variable
co = cohere.ClientV2(COHERE_API_KEY)

# 2. Setup MongoDB Connection using env variable
client = pymongo.MongoClient(MONGODB_URI)
db = client.sample_mflix
collection = db.movies

# 3. Embedding Generator Function
def generate_embedding(text: str, input_type: str = "search_document") -> list[float]:
    response = co.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type=input_type
    )
    return response.embeddings.float_[0]

# 4. Vector Search Query Execution
query = "imaginary characters from outer space at war"
print(f"Executing Vector Search for: '{query}'...\n")

results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": generate_embedding(query, input_type="search_query"),
            "path": "plot_embedding_hf",
            "numCandidates": 100,
            "limit": 4,
            "index": "autoembed_index"
        }
    }
])

# 5. Print Search Results
found = 0
for document in results:
    found += 1
    print(f"[{found}] Movie Name: {document.get('title')}")
    print(f"    Movie Plot: {document.get('plot')}\n")

if found == 0:
    print("No results found. Ensure your Atlas Vector Search Index 'autoembed_index' status is READY.")