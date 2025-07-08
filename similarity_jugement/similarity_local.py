from vllm import LLM, SamplingParams
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import pickle
import re
import argparse
from typing import List, Tuple
import glob
import os

class SimilarityJudgement:

    def __init__(self, args):
        
        self.llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        self.type = 'year'
        self.num_control = args.num_control
        self.batch_size = args.batch_size
        
        if args.num_control == False:
            self.output_path = f"./results/{args.model_name}_similarity_year2year.pkl"
        elif args.num_control == True:
            self.output_path = f"./results/{args.model_name}_similarity_num2num.pkl"
    
    def getprompt(self, num1, num2, type, item):
        prompt = f"""How close are the two {type} on a scale of 0 (completely dissimilar) to 1 (completely similar)? 

Respond only with the rating. 

{item}_1: {num1};

{item}_2: {num2};

Rating: """
        return prompt
    
    def generate_databases(self):
        return list(range(1525, 2525))
    
    def extract_float(self, string):
        
        match = re.search(r'(\d+\.\d+)', string)
        if match:
            return float(match.group(1))
        
        match = re.search(r'(\d+)', string)
        if match:
            return float(match.group(1))
        return 0.0

    def create_prompts_batch(self, pairs: List[Tuple[int, int]]) -> List[str]:
        
        prompts = []
        kind = self.type if self.num_control == False else 'number'
        item = kind.capitalize()
        
        for datapoint1, datapoint2 in pairs:
            prompt = self.getprompt(datapoint1, datapoint2, kind, item)
            prompts.append(prompt)
        
        return prompts
    
    def process_batch_responses(self, responses) -> List[float]:

        ratings = []
        for response in responses:
            try:
                output_text = response.outputs[0].text.strip()
                rating = max(0, min(1, self.extract_float(output_text)))
                ratings.append(rating)
            except Exception as e:
                print(f"Error processing response: {e}")
                ratings.append(0.0)
        
        return ratings
    
    def show_example(self):

        kind_example = self.type if self.num_control == False else 'number'
        prompt_example = self.getprompt(1, 2, kind_example, kind_example.capitalize())

        print("Prompt example:", prompt_example)
        
    
    def similarity_judgement(self):
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

        print(f"Total pairs: {total_pairs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches: {len(batches)}")

        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for batch_idx, batch in enumerate(batches):
                
                prompts = self.create_prompts_batch(batch)
                
                
                try:
                    responses = self.llm.generate(prompts, self.sampling_params)
                    ratings = self.process_batch_responses(responses)
                    
                    
                    for pair_idx, rating in enumerate(ratings):
                        absolute_idx = batch_idx * self.batch_size + pair_idx
                        i = absolute_idx // datalength
                        j = absolute_idx % datalength
                        similarity_matrix[i, j] = rating
                
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    
                    for pair_idx in range(len(batch)):
                        absolute_idx = batch_idx * self.batch_size + pair_idx
                        i = absolute_idx // datalength
                        j = absolute_idx % datalength
                        similarity_matrix[i, j] = 0.0
                
                pbar.update(1)

                if (batch_idx + 1) % 10000 == 0:
                    temp_path = self.output_path.replace('.pkl', f'_temp_batch_{batch_idx}.pkl')
                    with open(temp_path, 'wb') as f:
                        pickle.dump(similarity_matrix, f)
        
        with open(self.output_path, 'wb') as f:
            pickle.dump(similarity_matrix, f)
        


        temp_pattern = self.output_path.replace('.pkl', '_temp_batch_*.pkl')
        temp_files = glob.glob(temp_pattern)
    
        print(f"Cleaning up {len(temp_files)} temporary files...")
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"Deleted: {temp_file}")
            except Exception as e:
                print(f"Error deleting {temp_file}: {e}")
        
        print(f"Similarity matrix saved to {self.output_path}")
        return similarity_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Similarity judgement using vLLM")

    parser.add_argument("--model_path", type=str, default="meta-llama--Llama-3.1-70B", 
                       help="Path to the local model")
    parser.add_argument("--model_name", type=str, default="Llama-3.1-70B-Base",
                       help="Model name for output file naming")
    parser.add_argument("--tensor_parallel_size", type=int, default=8,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                       help="GPU memory utilization ratio")
    parser.add_argument("--max_model_len", type=int, default=4096,
                       help="Maximum model length")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Trust remote code for model loading")    
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=10,
                       help="Maximum tokens to generate")    
    parser.add_argument("--num_control", default=False,
                       help="Use number control instead of year")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing")

    args = parser.parse_args()

    test = SimilarityJudgement(args)
    test.show_example()
    similarity_matrix = test.similarity_judgement()
