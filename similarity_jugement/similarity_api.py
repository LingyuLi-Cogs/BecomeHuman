from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pickle
import re
import argparse

class SimilarityJudgement:

    def __init__(self, args):
        self.client = OpenAI(api_key=args.api_key, base_url=args.api_url)
        self.max_workers = args.max_workers
        self.model = args.model_name
        self.type = 'year'
        self.num_control = args.num_control
        self.temperature = args.temperature
        self.batch_size = args.batch_size
        if args.num_control == False:
            self.output_path = f"./results/{self.model}_similarity_year2year.pkl"
        elif args.num_control == True:
            self.output_path = f"./results/{self.model}_similarity_num2num.pkl"
    
    def getprompt(self, num1, num2, type, item):
        prompt = f"""
        How close are the two {type} on a scale of 0 (completely dissimilar) to 1 (completely similar)? 
        
        Respond only with the rating. 
        
        {item}_1: {num1} 
        
        {item}_2: {num2}
        
        Rating: 
        """
        return prompt
    
    def getrating(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        rating = response.choices[0].message.content
        return rating
    
    def generate_databases(self):
        return list(range(1525, 2525))
    
    def extract_float(self, string):
        match = re.search(r'(\d+\.\d+)', string)
        if match:
            return float(match.group(1))
        else:
            return float(string)

    def process_pair(self, dps):
        datapoint1, datapoint2 = dps
        kind = self.type if self.num_control == False else 'number'
        item = kind.capitalize()
        prompt = self.getprompt(datapoint1, datapoint2, kind, item)

        try:
            rating_raw = self.getrating(prompt)
            rating = max(0, min(1, self.extract_float(rating_raw)))
            return rating
        except Exception as e:
            print(f"Error comparing {datapoint1} and {datapoint2}: {e}")
            return 0
    
    def similarity_judgement(self):

        kind_exp = self.type if self.num_control == False else 'number'
        prompt_showing = self.getprompt(1, 2, kind_exp, kind_exp.capitalize())
        print("Example of prompt:", prompt_showing)


        datapoints = self.generate_databases()
        datalength = len(datapoints)
        similarity_matrix = np.zeros((datalength, datalength))
        total_pairs = datalength * datalength

        all_pairs = [
            (datapoints[i], datapoints[j]) 
            for i in range(datalength) 
            for j in range(datalength)
        ]

        batches = [
            all_pairs[i:i + self.batch_size] 
            for i in range(0, len(all_pairs), self.batch_size)
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with tqdm(total=total_pairs) as pbar:
                for batch_idx, batch in enumerate(batches):
                    batch_results = list(executor.map(self.process_pair, batch))
                
                    for pair_idx, rating in enumerate(batch_results):
                        absolute_idx = batch_idx * self.batch_size + pair_idx
                        i = absolute_idx // datalength
                        j = absolute_idx % datalength
                        similarity_matrix[i, j] = rating
                        pbar.update(1)
        
        with  open(self.output_path, 'wb') as f:
            pickle.dump(similarity_matrix, f)
    
        return similarity_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--api_url", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_workers", type=int, required=True)
    parser.add_argument("--num_control", type=bool, default=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=500)

    args = parser.parse_args()

    test = SimilarityJudgement(args)
    similarity_matrix = test.similarity_judgement()
