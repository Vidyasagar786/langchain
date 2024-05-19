import os
import numpy as np
import faiss
os.environ['MONGODB_URI'] = 'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false'
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chat_models import AzureChatOpenAI
from pymongo import MongoClient
from scipy.spatial.distance import cosine

load_dotenv()

class LLMCache:
    def __init__(self, openai_model_args):
        self.openai_model_args = openai_model_args
        self.bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en")
        self.client = MongoClient(os.environ.get('MONGODB_URI'))
        self.db = self.client["cache_database"]
        self.collection = self.db["cache"]
        self.index_built = False
        self.index = None  

    def build_index(self):
        prompt_embeddings = []
        for document in self.collection.find():
            prompt_embeddings.append(np.array(document['prompt_embedding']).flatten())
        
        if prompt_embeddings:
            self.index = faiss.IndexFlatL2(len(prompt_embeddings[0]))
            self.index.add(np.array(prompt_embeddings, dtype=np.float32))
            self.index_built = True
        else:
            self.index_built = False


    def query_cache(self, prompt, threshold=0.9):
        print("Querying cache for prompt:", prompt)
        new_prompt_embedding = np.array(self.bge_embeddings.embed_query(prompt)).flatten()

        if not self.index_built:
            # print("Index not built. Building index...")
            self.build_index()

        if self.index_built:
            distances, indices = self.index.search(np.array([new_prompt_embedding], dtype=np.float32), 1)
            best_similarity = 1 - distances[0][0]
            # Normalizing similarity score to [0, 1]
            normalized_similarity = (best_similarity - 0) / (1 - 0)
            index_int = int(indices[0][0])
            document = self.collection.find_one({"prompt_embedding": self.index.reconstruct(index_int).tolist()})
            best_response = document['response'] if document and normalized_similarity > threshold else None
            # print("Calculated normalized similarity score:", normalized_similarity)
            # print("Threshold:", threshold)
            if best_response:
                print("Cache hit. Best response:", best_response)
            else:
                print("Cache miss, data is not in our DB.")
            # embeddings for debug
            # print("New prompt embedding:", new_prompt_embedding)
            if document:
                cached_embedding = np.array(document['prompt_embedding'])
                #print("Cached prompt embedding:", cached_embedding)
            return best_response, normalized_similarity if best_response else None
        else:
            print("Index not built. Cache miss.")
            return None, None


    def update_cache(self, prompt, response):
        prompt_embedding = self.bge_embeddings.embed_query(prompt)
        prompt_embedding = np.array([prompt_embedding])
        
        document = {
            "prompt": prompt,
            "response": response if isinstance(response, str) else str(response),
            "prompt_embedding": prompt_embedding.tolist()  # Storing as floats
        }
        try:
            self.collection.insert_one(document)
            if self.index_built:
                self.index.add(np.array(prompt_embedding, dtype=np.float32))
        except Exception as e:
            print("Error updating cache:", e)



    def call_llm(self, prompt):
        threshold = 0.90
        cached_response, similarity = self.query_cache(prompt, threshold=threshold)
        if cached_response and similarity >= threshold:
            print("Using cache for prompt:", prompt)
            print("Similarity score:", similarity)
            return cached_response, similarity
        else:
            print("Cache missed. Calling llm:")#, prompt)
            llm = AzureChatOpenAI(**self.openai_model_args)
            response = llm.invoke(prompt)
            self.update_cache(prompt, response)
            return response, None


openai_model_args = {
    "deployment_name": os.environ.get('DEPLOYMENT_NAME'),
    "model_name": os.environ.get('MODEL_NAME'),
    "temperature": float(os.environ.get('TEMPERATURE')),
    "openai_api_base": os.environ.get('OPENAI_API_BASE'),
    "openai_api_version": os.environ.get('OPENAI_API_VERSION'),
    "openai_api_key": os.environ.get('OPENAI_API_KEY'),
    "openai_api_type": os.environ.get('OPENAI_API_TYPE')
}

llm_cache = LLMCache(openai_model_args)

# Test
# prompt = "Explain the steps involved in brewing a cup of tea."
prompt="give me the steps involved in brewing the cup of tea"
response, similarity = llm_cache.call_llm(prompt)
print("Response:", response)
