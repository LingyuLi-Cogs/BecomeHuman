from transformers import AutoModel, AutoTokenizer
import torch
import pickle
import numpy as np
from tqdm import tqdm

class YearEmbeddingExtractor:
    def __init__(self, model_path, pkl_file="./year_embeddings_qwen.pkl"):
        self.pkl_file = pkl_file
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
    
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy().squeeze()
    
    def extract_year_embeddings(self, start_year=1525, end_year=2024, num_points=1000):
        years = np.linspace(start_year, end_year, num_points, dtype=int)
        data = []
        for year in tqdm(years):
            text = f"year: {year}"
            embedding = self.get_embedding(text)
            data.append({'year': int(year), 'text': text, 'embedding': embedding})
        return data
    
    def save_embeddings(self, data):
        with open(self.pkl_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {len(data)} embeddings to {self.pkl_file}")

def main():
    model_path = "Qwen--Qwen3-Embedding-8B"
    extractor = YearEmbeddingExtractor(model_path)
    year_data = extractor.extract_year_embeddings()
    extractor.save_embeddings(year_data)

if __name__ == "__main__":
    main()
