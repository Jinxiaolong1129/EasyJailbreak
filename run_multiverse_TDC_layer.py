import os

os.environ['HF_HOME'] = '/data3/user/jin509/hf_cache'

import sys
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.attacker.Multiverse import *
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.gemini import GeminiModel
from easyjailbreak.models.huggingface_model import HuggingfaceModel

sys.path.append(os.getcwd())


def dataset_prepare(process_retain=False, retain_file_path=None, 
                    start_num=0, end_num=1, 
                    target_model_name=None, dataset=None,
                    layer_num=None):
    path = os.path.join('results', f'ablation_layer_{layer_num}')
    os.makedirs(path, exist_ok=True)
    if process_retain:
        file_path = retain_file_path
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data = []
        for index, line in enumerate(lines):
            record = json.loads(line)
            data.append(record)
        filtered_indices = [record["query_index"] for record in data if record["eval_results"] == [False]]

        dataset._dataset = [dataset._dataset[i] for i in filtered_indices]
            
        log_path = os.path.join(path, f'TDC_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_retain{len(filtered_indices)}.log')
        print(f"log_path: {log_path}")    
        save_dir = os.path.join(path, f'TDC_target.{target_model_name}_eval.gpt-3.5-turbo_retain{len(filtered_indices)}.jsonl')
        print(f"save: {save_dir}")
    else:
        dataset._dataset = dataset._dataset[start_num:end_num]
        log_path = os.path.join(path, f'TDC_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_{start_num}_{end_num}.log')
        print(f"log_path: {log_path}")
        save_dir = os.path.join(path, f'TDC_target.{target_model_name}_eval.gpt-3.5-turbo_{start_num}_{end_num}.jsonl')
        print(f"save: {save_dir}")
    
    return dataset, log_path, save_dir


def load_target_model(target_model_name, api_key=None, generation_config=None):
    model_paths = {
        'llama-2-7b': 'meta-llama/Llama-2-7b-chat-hf',
        'llama-2-13b': 'meta-llama/Llama-2-13b-chat-hf',
        'vicuna-7b': "lmsys/vicuna-7b-v1.5",
        'mixtral-7b': "mistralai/Mistral-7B-v0.1",
        'chatglm3-6b': "THUDM/chatglm3-6b",
        'chatglm2-6b': "THUDM/chatglm2-6b",
        "qwen-7b": "Qwen/Qwen-7B-Chat",
        "intern-7b": "internlm/internlm-7b",
        'gpt-4': 'gpt-4', 
        'gpt-3.5-turbo': 'gpt-3.5-turbo',  
        'gemini-1.5-flash': 'gemini-1.5-flash'
    }
    
    if target_model_name not in model_paths:
        raise ValueError(f"Invalid target model name: {target_model_name}")

    # Load the target model
    if target_model_name.startswith('llama') or target_model_name.startswith('vicuna') or target_model_name.startswith('mixtral'):
        model_path = model_paths[target_model_name]
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
    
    elif target_model_name == 'qwen-7b':
        model_path = model_paths[target_model_name]
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path)
    
    elif target_model_name == 'intern-7b':
        model_path = model_paths[target_model_name]
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
        
    elif target_model_name == 'chatglm3-6b' or target_model_name == 'chatglm2-6b':
        model_path = model_paths[target_model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
    
    elif target_model_name == 'gpt-4':
        target_model = OpenaiModel(model_name='gpt-4', api_keys=api_key)
    
    elif target_model_name == 'gpt-3.5-turbo':
        target_model = OpenaiModel(model_name='gpt-3.5-turbo', api_keys=api_key)
    
    elif target_model_name == 'gemini-1.5-flash':
        target_model = GeminiModel(model_name='gemini-1.5-flash')
    
    return target_model


def main(target_model_name, start_num, end_num, process_retain=False, retain_file_path=None, layer_num=None):
    generation_config = {'max_new_tokens': 500}
    
    dataset = JailbreakDataset(dataset='data/TDC_data.csv', local_file_type='csv')
    
    dataset, log_path, save_dir = dataset_prepare(process_retain=process_retain, retain_file_path=retain_file_path, 
                                                start_num=start_num, end_num=end_num, 
                                                target_model_name=target_model_name, dataset=dataset,
                                                layer_num=layer_num)

    target_model = load_target_model(target_model_name, api_key=api_key, generation_config=generation_config)
    
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    eval_model = OpenaiModel(model_name='gpt-3.5-turbo', api_keys=api_key)
    
    attacker = Multiverse(
        attack_model=None,
        target_model=target_model,
        target_model_name = target_model_name,
        eval_model=eval_model,
        jailbreak_datasets=dataset,
        layer_number=layer_num,
        save_path='results',
        log_path=log_path,
        ablation_layer = True
    )
    
    attacker.attack()
    attacker.log()
    attacker.attack_results.save_to_jsonl(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run jailbreak attacks on specified target model.")
    parser.add_argument('--target_model', type=str, default='gpt-3.5-turbo',
                        help="Specify the target model for the attack.")
    parser.add_argument('--start_num', type=int, default=0)
    parser.add_argument('--end_num', type=int, default=1)
    parser.add_argument('--retain_file_path', type=str, default=None)
    parser.add_argument('--layer_num', type=int, default=None)
    
    args = parser.parse_args()
    
    
    api_key = os.getenv('OPENAI_API_KEY')
    if args.retain_file_path is not None:
        main(args.target_model, args.start_num, args.end_num, process_retain=True, retain_file_path=args.retain_file_path)
    else:
        for layer_num in range(1, 7):
            print('===========================')
            print(f"Layer number: {layer_num}")
            main(args.target_model, args.start_num, args.end_num, process_retain=False, layer_num=layer_num)
        