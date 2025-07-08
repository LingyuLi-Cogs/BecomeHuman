import h5py
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import json
import os
import re
import warnings
warnings.filterwarnings('ignore')

def benjamini_hochberg_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR"""
    p_values = np.array(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]    
    n = len(p_values)
    corrected_p_values = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        if i == n-1:
            corrected_p_values[sorted_indices[i]] = sorted_p_values[i]
        else:
            corrected_p_values[sorted_indices[i]] = min(
                sorted_p_values[i] * n / (i + 1),
                corrected_p_values[sorted_indices[i+1]]
            )
    corrected_p_values = np.minimum(corrected_p_values, 1.0)
    
    return corrected_p_values


def extract_model_name(file_path):
    
    dir_name = os.path.basename(os.path.dirname(file_path))    
    if dir_name.endswith('_neural_activations'):
        model_name = dir_name[:-len('_neural_activations')]
    else:
        model_name = dir_name
    model_name = re.sub(r'[^\w\-_.]', '_', model_name)
    
    return model_name


def create_output_path(input_file_path, base_analysis_dir):

    model_name = extract_model_name(input_file_path)
    output_dir = os.path.join(base_analysis_dir, f"{model_name}_temporal_neurons")
    
    return output_dir


def convert_to_json_serializable(obj):    
    if isinstance(obj, np.ndarray):
        return obj.astype(float).tolist()
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    else:
        return obj


class TemporalNeuronFilter:
    def __init__(self, activation_file_temporal, activation_file_numerical, output_dir=None):
        self.temporal_data = h5py.File(activation_file_temporal, 'r')
        self.numerical_data = h5py.File(activation_file_numerical, 'r')
        self.years = self.temporal_data['years'][:]
        layer_names_raw = [key for key in self.temporal_data.keys() if key.startswith('layer_')]
        self.layer_names = sorted(layer_names_raw, key=lambda x: int(x.split('_')[1]))
        if output_dir is None:
            self.output_dir = create_output_path(activation_file_temporal)
            self.model_name = extract_model_name(activation_file_temporal)
        else:
            self.output_dir = output_dir
            self.model_name = extract_model_name(activation_file_temporal)        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Model: {self.model_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Loaded data for {len(self.years)} years and {len(self.layer_names)} layers")
        print(f"Year range: {self.years.min()} - {self.years.max()}")
    
    def calculate_neuron_statistics(self, temporal_acts, numerical_acts):
        
        temporal_acts = temporal_acts.astype(np.float64)
        numerical_acts = numerical_acts.astype(np.float64)
        mean_temporal = np.mean(temporal_acts)
        mean_numerical = np.mean(numerical_acts)
        mean_difference = mean_temporal - mean_numerical
        pooled_std = np.sqrt(((len(temporal_acts) - 1) * np.var(temporal_acts, ddof=1) + 
                             (len(numerical_acts) - 1) * np.var(numerical_acts, ddof=1)) / 
                            (len(temporal_acts) + len(numerical_acts) - 2))
        
        if pooled_std == 0:
            cohens_d = 0
        else:
            cohens_d = mean_difference / pooled_std
        try:
            t_stat, p_value = stats.ttest_rel(temporal_acts, numerical_acts)
        except:
            t_stat, p_value = 0, 1

        differences = temporal_acts - numerical_acts
        consistency = np.mean(differences > 0)
        
        return {
            'cohens_d': float(cohens_d),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'consistency': float(consistency),
            'mean_temporal': float(mean_temporal),
            'mean_numerical': float(mean_numerical),
            'mean_difference': float(mean_difference),
            'activations_temporal': temporal_acts.astype(float).tolist(),
            'activations_numerical': numerical_acts.astype(float).tolist()
        }
    
    def filter_temporal_neurons(self, fdr_threshold=0.0001, cohens_d_threshold=2.0, consistency_threshold=0.95):

        print(f"\nFiltering temporal neurons for model: {self.model_name}")
        print("Criteria:")
        print(f"  - FDR-corrected p-value < {fdr_threshold}")
        print(f"  - Cohen's d > {cohens_d_threshold}")
        print(f"  - Consistency > {consistency_threshold}")
        
        temporal_neurons = []
        layer_summary = {}
        
        for layer_name in tqdm(self.layer_names, desc="Processing layers"):
            temporal_activations = self.temporal_data[layer_name][:]
            numerical_activations = self.numerical_data[layer_name][:]
            layer_num = int(layer_name.split('_')[1])
            
            layer_neurons = []
            neuron_p_values = []
            neuron_stats_list = []
            
            for neuron_idx in tqdm(range(temporal_activations.shape[1]), 
                                 desc=f"Layer {layer_num}", leave=False):
                
                temp_acts = temporal_activations[:, neuron_idx]
                num_acts = numerical_activations[:, neuron_idx]
                
                stats_dict = self.calculate_neuron_statistics(temp_acts, num_acts)
                stats_dict['layer'] = layer_num
                stats_dict['neuron_idx'] = neuron_idx
                
                neuron_stats_list.append(stats_dict)
                neuron_p_values.append(stats_dict['p_value'])
            
            valid_p_mask = np.array([(0 < p < 1) and np.isfinite(p) for p in neuron_p_values])
            corrected_p_values = np.ones(len(neuron_p_values))  # 默认为1
            
            if valid_p_mask.sum() > 1:
                valid_p_values = np.array(neuron_p_values)[valid_p_mask]
                corrected_valid_p = benjamini_hochberg_correction(valid_p_values)
                corrected_p_values[valid_p_mask] = corrected_valid_p
                
            for i, stats_dict in enumerate(neuron_stats_list):
                stats_dict['p_value_fdr'] = float(corrected_p_values[i])
                
                if (stats_dict['p_value_fdr'] < fdr_threshold and
                    stats_dict['cohens_d'] > cohens_d_threshold and
                    stats_dict['consistency'] > consistency_threshold):
                    stats_dict['years'] = self.years.astype(int).tolist()
                    layer_neurons.append(stats_dict)
                    temporal_neurons.append(stats_dict)
                    
            layer_summary[layer_num] = {
                'total_neurons': len(neuron_stats_list),
                'temporal_neurons': len(layer_neurons),
                'percentage': len(layer_neurons) / len(neuron_stats_list) * 100
            }
            
            print(f"Layer {layer_num:2d}: {len(layer_neurons)}/{len(neuron_stats_list)} "
                  f"({layer_summary[layer_num]['percentage']:.3f}%) temporal neurons")
        
        print(f"\nTotal temporal neurons found: {len(temporal_neurons)}")
        
        return temporal_neurons, layer_summary
    
    def save_results(self, temporal_neurons, layer_summary, custom_prefix=None):
        
        if custom_prefix is None:
            filename_prefix = f"{self.model_name}_temporal_neurons"
        else:
            filename_prefix = custom_prefix
            
        print(f"Saving results to {self.output_dir}/...")
        output_data = {
            'metadata': {
                'model_name': self.model_name,
                'total_neurons': len(temporal_neurons),
                'year_range': [int(self.years.min()), int(self.years.max())],
                'total_years': len(self.years),
                'total_layers': len(self.layer_names),
                'filtering_criteria': {
                    'fdr_threshold': 0.0001,
                    'cohens_d_threshold': 2.0,
                    'consistency_threshold': 0.95
                },
                'output_directory': self.output_dir
            },
            'layer_summary': layer_summary,
            'temporal_neurons': temporal_neurons
        }
        output_data = convert_to_json_serializable(output_data)
        
        with open(f'{self.output_dir}/{filename_prefix}_full.json', 'w') as f:
            json.dump(output_data, f, indent=2)

        if temporal_neurons:
            df_data = []
            for neuron in temporal_neurons:
                df_data.append({
                    'model': self.model_name,
                    'layer': neuron['layer'],
                    'neuron_idx': neuron['neuron_idx'],
                    'cohens_d': neuron['cohens_d'],
                    'p_value_fdr': neuron['p_value_fdr'],
                    'consistency': neuron['consistency'],
                    't_statistic': neuron['t_statistic'],
                    'mean_temporal': neuron['mean_temporal'],
                    'mean_numerical': neuron['mean_numerical'],
                    'mean_difference': neuron['mean_difference']
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(f'{self.output_dir}/{filename_prefix}_summary.csv', index=False)
            
            print(f"Saved {len(temporal_neurons)} temporal neurons to:")
            print(f"  - {filename_prefix}_full.json (complete data with activations)")
            print(f"  - {filename_prefix}_summary.csv (summary statistics)")
        else:
            print("No temporal neurons found to save.")
            
        layer_df = pd.DataFrame.from_dict(layer_summary, orient='index')
        layer_df.index.name = 'layer'
        layer_df['model'] = self.model_name
        layer_df.to_csv(f'{self.output_dir}/{self.model_name}_layer_statistics.csv')
        
        print(f"  - {self.model_name}_layer_statistics.csv (layer-wise summary)")
        
        with open(f'{self.output_dir}/README.txt', 'w') as f:
            f.write(f"Temporal Neuron Analysis Results\n")
            f.write(f"================================\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            f.write(f"Files Generated:\n")
            f.write(f"- {filename_prefix}_full.json: Complete neuron data with activations\n")
            f.write(f"- {filename_prefix}_summary.csv: Summary statistics table\n")
            f.write(f"- {self.model_name}_layer_statistics.csv: Layer-wise statistics\n")
            f.write(f"- README.txt: This file\n\n")
            f.write(f"Filtering Criteria:\n")
            f.write(f"- FDR-corrected p-value < 0.0001\n")
            f.write(f"- Cohen's d > 2.0\n")
            f.write(f"- Consistency > 95%\n\n")
            f.write(f"Results Summary:\n")
            f.write(f"- Total temporal neurons found: {len(temporal_neurons)}\n")
            f.write(f"- Year range analyzed: {self.years.min()}-{self.years.max()}\n")
            f.write(f"- Layers analyzed: {len(self.layer_names)}\n")
    
    def print_summary(self, temporal_neurons, layer_summary):

        print("\n" + "="*70)
        print(f"TEMPORAL NEURON FILTERING SUMMARY - {self.model_name.upper()}")
        print("="*70)
        
        total_neurons = sum([stats['total_neurons'] for stats in layer_summary.values()])
        total_temporal = len(temporal_neurons)
        overall_percentage = total_temporal / total_neurons * 100 if total_neurons > 0 else 0
        
        print(f"Model: {self.model_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nDataset overview:")
        print(f"  - Years: {len(self.years)} ({self.years.min()}-{self.years.max()})")
        print(f"  - Layers: {len(self.layer_names)}")
        print(f"  - Total neurons: {total_neurons:,}")
        print(f"  - Temporal neurons: {total_temporal:,} ({overall_percentage:.4f}%)")
        
        print(f"\nFiltering criteria applied:")
        print(f"  - FDR-corrected p < 0.0001")
        print(f"  - Cohen's d > 2.0")
        print(f"  - Consistency > 95%")
        
        if temporal_neurons:
            print(f"\nTop temporal neurons by Cohen's d:")
            sorted_neurons = sorted(temporal_neurons, key=lambda x: x['cohens_d'], reverse=True)
            for i, neuron in enumerate(sorted_neurons[:5]):
                print(f"  {i+1}. Layer {neuron['layer']}, Neuron {neuron['neuron_idx']}: "
                      f"d={neuron['cohens_d']:.3f}, p={neuron['p_value_fdr']:.2e}, "
                      f"consistency={neuron['consistency']:.3f}")
        
        print("\n" + "="*70)
    
    def close(self):
        
        self.temporal_data.close()
        self.numerical_data.close()


def main():
    
    temporal_file = './results/qwen25_72b_neural_activations/activations_temporal_merged.h5'
    numerical_file = './results/qwen25_72b_neural_activations/activations_numerical_merged.h5'
    
    if not os.path.exists(temporal_file) or not os.path.exists(numerical_file):
        print("Error: Activation files not found")
        print(f"Expected files:")
        print(f"  {temporal_file}")
        print(f"  {numerical_file}")
        return
    model_name = extract_model_name(temporal_file)
    output_path = create_output_path(temporal_file)
    
    print(f"Detected model: {model_name}")
    print(f"Results will be saved to: {output_path}")
    
    filter_tool = TemporalNeuronFilter(
        temporal_file, 
        numerical_file
    )
    
    try:
        temporal_neurons, layer_summary = filter_tool.filter_temporal_neurons(
            fdr_threshold=0.0001,
            cohens_d_threshold=2.0,
            consistency_threshold=0.95
        )
        filter_tool.print_summary(temporal_neurons, layer_summary)
        filter_tool.save_results(temporal_neurons, layer_summary)
        print(f"Results saved in: {filter_tool.output_dir}")
        
        return temporal_neurons, layer_summary
        
    finally:
        filter_tool.close()

if __name__ == "__main__":
    main()
