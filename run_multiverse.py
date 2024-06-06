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

def main_llama(target_model_name, start_num, end_num):
    generation_config = {'max_new_tokens': 500}
    
    dataset_name = 'AdvBench'
    
    dataset = JailbreakDataset(dataset_name)
    
    file_path = 'results/advbench_target.llama-2-7b_eval.gpt-3.5-turbo-retain174.jsonl'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for index, line in enumerate(lines):
        record = json.loads(line)
        data.append(record)
    filtered_indices = [record["query_index"] for record in data if record["eval_results"] == [False]]

    dataset._dataset = [dataset._dataset[i] for i in filtered_indices]
    
    log_path = os.path.join('results', f'advbench_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_retain{len(filtered_indices)}.log')
    print(f"log_path: {log_path}")
    
    save_dir = os.path.join('results', f'advbench_target.{target_model_name}_eval.gpt-3.5-turbo_retain{len(filtered_indices)}.jsonl')
    print(f"Running jailbreak attack on target model: {save_dir}")
    
    
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
    }
    
    if target_model_name not in model_paths:
        raise ValueError(f"Invalid target model name: {target_model_name}")

    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

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
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
    
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
        
        
    eval_model = OpenaiModel(model_name='gpt-3.5-turbo', api_keys=api_key)
    

    attacker = Multiverse(
        attack_model=None,
        target_model=target_model,
        target_model_name = target_model_name,
        eval_model=eval_model,
        jailbreak_datasets=dataset,
        scene='dream',
        character_number=5,
        layer_number=5,
        save_path='results',
        log_path=log_path
    )
    
    attacker.attack()
    attacker.log()
    attacker.attack_results.save_to_jsonl(save_dir)


def main_chatglm3_6b(target_model_name, start_num, end_num):
    generation_config = {'max_new_tokens': 500}
    
    dataset_name = 'AdvBench'
    
    dataset = JailbreakDataset(dataset_name)
    
    # file_path = 'results/advbench_target.chatglm3-6b_eval.gpt-3.5-turbo.jsonl'
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()

    # data = []
    # for index, line in enumerate(lines):
    #     record = json.loads(line)
    #     data.append(record)
    # filtered_indices = [record["query_index"] for record in data if record["eval_results"] == [False]]

    # dataset._dataset = [dataset._dataset[i] for i in filtered_indices]
    
    dataset._dataset = dataset._dataset[start_num:end_num]
    log_path = os.path.join('results', f'advbench_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_start{start_num}_end{end_num}.log')
    print(f"log_path: {log_path}")
    save_dir = os.path.join('results', f'advbench_target.{target_model_name}_eval.gpt-3.5-turbo_start{start_num}_end{end_num}.jsonl')
    print(f"Running jailbreak attack on target model: {save_dir}")
    
    
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
    }
    
    if target_model_name not in model_paths:
        raise ValueError(f"Invalid target model name: {target_model_name}")


    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

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
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
    
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
        
        
    eval_model = OpenaiModel(model_name='gpt-3.5-turbo', api_keys=api_key)
    

    attacker = Multiverse(
        attack_model=None,
        target_model=target_model,
        target_model_name = target_model_name,
        eval_model=eval_model,
        jailbreak_datasets=dataset,
        scene='dream',
        character_number=5,
        layer_number=5,
        save_path='results',
        log_path=log_path
    )
    
    attacker.attack()
    attacker.log()
    attacker.attack_results.save_to_jsonl(save_dir)


def main_chatglm2_6b(target_model_name, start_num, end_num):
    generation_config = {'max_new_tokens': 500}
    
    dataset_name = 'AdvBench'
    
    dataset = JailbreakDataset(dataset_name)
    
    # file_path = 'results/advbench_target.chatglm3-6b_eval.gpt-3.5-turbo.jsonl'
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()

    # data = []
    # for index, line in enumerate(lines):
    #     record = json.loads(line)
    #     data.append(record)
    # filtered_indices = [record["query_index"] for record in data if record["eval_results"] == [False]]

    # dataset._dataset = [dataset._dataset[i] for i in filtered_indices]    
    # log_path = os.path.join('results', f'advbench_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_retain{len(filtered_indices)}.log')
    # print(f"log_path: {log_path}")
    
    # save_dir = os.path.join('results', f'advbench_target.{target_model_name}_eval.gpt-3.5-turbo_retain{len(filtered_indices)}.jsonl')
    # print(f"Running jailbreak attack on target model: {save_dir}")
    
    
    
    dataset._dataset = dataset._dataset[start_num:end_num]
    log_path = os.path.join('results', f'advbench_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_start{start_num}_end{end_num}.log')
    print(f"log_path: {log_path}")
    save_dir = os.path.join('results', f'advbench_target.{target_model_name}_eval.gpt-3.5-turbo_start{start_num}_end{end_num}.jsonl')
    print(f"Running jailbreak attack on target model: {save_dir}")
    
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
    }
    
    if target_model_name not in model_paths:
        raise ValueError(f"Invalid target model name: {target_model_name}")


    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

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
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
    
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
        
        
    eval_model = OpenaiModel(model_name='gpt-3.5-turbo', api_keys=api_key)
    

    attacker = Multiverse(
        attack_model=None,
        target_model=target_model,
        target_model_name = target_model_name,
        eval_model=eval_model,
        jailbreak_datasets=dataset,
        scene='dream',
        character_number=5,
        layer_number=5,
        save_path='results',
        log_path=log_path
    )
    
    attacker.attack()
    attacker.log()
    attacker.attack_results.save_to_jsonl(save_dir)


def main_vicuna_7b(target_model_name, start_num, end_num):
    generation_config = {'max_new_tokens': 500}
    
    dataset_name = 'AdvBench'
    
    dataset = JailbreakDataset(dataset_name)
    
    
    # dataset._dataset = dataset._dataset[start_num:end_num]
    # log_path = os.path.join('results', f'advbench_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_start{start_num}_end{end_num}.log')
    # print(f"log_path: {log_path}")
    # save_dir = os.path.join('results', f'advbench_target.{target_model_name}_eval.gpt-3.5-turbo.jsonl')
    # print(f"Running jailbreak attack on target model: {save_dir}")
    
    file_path = 'results/advbench_target.vicuna-7b_eval.gpt-3.5-turbo.jsonl'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for index, line in enumerate(lines):
        record = json.loads(line)
        data.append(record)
    filtered_indices = [record["query_index"] for record in data if record["eval_results"] == [False]]
    dataset._dataset = [dataset._dataset[i] for i in filtered_indices]

    log_path = os.path.join('results', f'advbench_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_retain{len(filtered_indices)}.log')
    print(f"log_path: {log_path}")
    save_dir = os.path.join('results', f'advbench_target.{target_model_name}_eval.gpt-3.5-turbo_retain{len(filtered_indices)}.jsonl')
    print(f"Running jailbreak attack on target model: {save_dir}")
    
    
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
    }
    
    if target_model_name not in model_paths:
        raise ValueError(f"Invalid target model name: {target_model_name}")


    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

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
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
    
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
        
        
    eval_model = OpenaiModel(model_name='gpt-3.5-turbo', api_keys=api_key)
    

    attacker = Multiverse(
        attack_model=None,
        target_model=target_model,
        target_model_name = target_model_name,
        eval_model=eval_model,
        jailbreak_datasets=dataset,
        scene='dream',
        character_number=5,
        layer_number=5,
        save_path='results',
        log_path=log_path
    )
    
    attacker.attack()
    attacker.log()
    attacker.attack_results.save_to_jsonl(save_dir)


def main_gpt_4(target_model_name, start_num, end_num):
    generation_config = {'max_new_tokens': 500}
    
    dataset_name = 'AdvBench'
    
    dataset = JailbreakDataset(dataset_name)
    
    # file_path = 'results/advbench_target.gpt-4_eval.gpt-3.5-turbo-retain130.jsonl'
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()

    # data = []
    # for index, line in enumerate(lines):
    #     record = json.loads(line)
    #     data.append(record)
    # filtered_indices = [record["query_index"] for record in data if record["eval_results"] == [False]]

    # dataset._dataset = [dataset._dataset[i] for i in filtered_indices]
    # log_path = os.path.join('results', f'advbench_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_retain{len(filtered_indices)}.log')
    # print(f"log_path: {log_path}")
    # save_dir = os.path.join('results', f'advbench_target.{target_model_name}_eval.gpt-3.5-turbo_retain{len(filtered_indices)}.jsonl')
    # print(f"Running jailbreak attack on target model: {save_dir}")
    
    
    dataset._dataset = dataset._dataset[start_num:end_num]
    log_path = os.path.join('results', f'advbench_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_start{start_num}_end{end_num}.log')
    print(f"log_path: {log_path}")
    save_dir = os.path.join('results', f'advbench_target.{target_model_name}_eval.gpt-3.5-turbo_start{start_num}_end{end_num}.jsonl')
    print(f"Running jailbreak attack on target model: {save_dir}")
    
    
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
    }
    
    if target_model_name not in model_paths:
        raise ValueError(f"Invalid target model name: {target_model_name}")


    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

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
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
    
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
        
        
    eval_model = OpenaiModel(model_name='gpt-3.5-turbo', api_keys=api_key)
    

    attacker = Multiverse(
        attack_model=None,
        target_model=target_model,
        target_model_name = target_model_name,
        eval_model=eval_model,
        jailbreak_datasets=dataset,
        scene='dream',
        character_number=5,
        layer_number=5,
        save_path='results',
        log_path=log_path
    )
    
    attacker.attack()
    attacker.log()
    attacker.attack_results.save_to_jsonl(save_dir)


def main_gpt_3_turbo(target_model_name, start_num, end_num):
    generation_config = {'max_new_tokens': 500}
    
    dataset_name = 'AdvBench'
    
    dataset = JailbreakDataset(dataset_name)
    
    # file_path = 'results/advbench_target.gpt-4_eval.gpt-3.5-turbo-retain130.jsonl'
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()

    # data = []
    # for index, line in enumerate(lines):
    #     record = json.loads(line)
    #     data.append(record)
    # filtered_indices = [record["query_index"] for record in data if record["eval_results"] == [False]]

    # dataset._dataset = [dataset._dataset[i] for i in filtered_indices]
    
    dataset._dataset = dataset._dataset[start_num:end_num]
    
    log_path = os.path.join('results', f'advbench_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_{start_num}_{end_num}.log')
    print(f"log_path: {log_path}")
    save_dir = os.path.join('results', f'advbench_target.{target_model_name}_eval.gpt-3.5-turbo_{start_num}_{end_num}.jsonl')
    
    print(f"Running jailbreak attack on target model: {save_dir}")
    
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
    }
    
    if target_model_name not in model_paths:
        raise ValueError(f"Invalid target model name: {target_model_name}")


    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

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
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
    
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
        
        
    eval_model = OpenaiModel(model_name='gpt-3.5-turbo', api_keys=api_key)
    

    attacker = Multiverse(
        attack_model=None,
        target_model=target_model,
        target_model_name = target_model_name,
        eval_model=eval_model,
        jailbreak_datasets=dataset,
        scene='dream',
        character_number=5,
        layer_number=5,
        save_path='results',
        log_path=log_path
    )
    
    attacker.attack()
    attacker.log()
    attacker.attack_results.save_to_jsonl(save_dir)


def main_gemini(target_model_name, start_num, end_num):
    generation_config = {'max_new_tokens': 500}
    
    dataset_name = 'AdvBench'
    
    dataset = JailbreakDataset(dataset_name)
    
    dataset._dataset = dataset._dataset[start_num:end_num]
    
    # log_path = os.path.join('results', f'advbench_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_{start_num}_{end_num}.log')
    # print(f"log_path: {log_path}")
    # save_dir = os.path.join('results', f'advbench_target.{target_model_name}_eval.gpt-3.5-turbo_{start_num}_{end_num}.jsonl')
    # print(f"Running jailbreak attack on target model: {save_dir}")
    
    
    file_path = 'results/advbench_target.gemini-1.5-flash_eval.gpt-3.5-turbo_0_520.jsonl'
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    for index, line in enumerate(lines):
        record = json.loads(line)
        data.append(record)
    filtered_indices = [record["query_index"] for record in data if record["eval_results"] == [False]]
    dataset._dataset = [dataset._dataset[i] for i in filtered_indices]
    
    
    log_path = os.path.join('results', f'advbench_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_retain{len(filtered_indices)}.log')
    print(f"log_path: {log_path}")
    save_dir = os.path.join('results', f'advbench_target.{target_model_name}_eval.gpt-3.5-turbo_retain{len(filtered_indices)}.jsonl')
    print(f"Running jailbreak attack on target model: {save_dir}")
    
    
    
    
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
        'gemini-1.5-flash': 'gemini-1.5-flash',
    }
    
    if target_model_name not in model_paths:
        raise ValueError(f"Invalid target model name: {target_model_name}")


    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

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
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
    
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
        
    eval_model = OpenaiModel(model_name='gpt-3.5-turbo', api_keys=api_key)
    

    attacker = Multiverse(
        attack_model=None,
        target_model=target_model,
        target_model_name = target_model_name,
        eval_model=eval_model,
        jailbreak_datasets=dataset,
        scene='dream',
        character_number=5,
        layer_number=5,
        save_path='results',
        log_path=log_path
    )
    
    attacker.attack()
    attacker.log()
    attacker.attack_results.save_to_jsonl(save_dir)



def main_qwen_7b(target_model_name, start_num, end_num):
    generation_config = {'max_new_tokens': 500}
    
    dataset_name = 'AdvBench'
    
    dataset = JailbreakDataset(dataset_name)
    
    file_path = '/home/jin509/EasyJailbreak/results/advbench_target.qwen-7b_eval.gpt-3.5-turbo-back.jsonl'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for index, line in enumerate(lines):
        record = json.loads(line)
        data.append(record)
    filtered_indices = [record["query_index"] for record in data if record["eval_results"] == [False]]

    dataset._dataset = [dataset._dataset[i] for i in filtered_indices]
    
    log_path = os.path.join('results', f'advbench_jailbreak_target.{target_model_name}_eval.gpt-3.5-turbo_retain{len(filtered_indices)}.log')
    print(f"log_path: {log_path}")
    
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
    }
    
    if target_model_name not in model_paths:
        raise ValueError(f"Invalid target model name: {target_model_name}")

    save_dir = os.path.join('results', f'advbench_target.{target_model_name}_eval.gpt-3.5-turbo.jsonl')
    print(f"Running jailbreak attack on target model: {save_dir}")
    
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

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
        target_model = HuggingfaceModel(model=model, tokenizer=tokenizer, model_name=model_path, generation_config=generation_config)
    
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
        
        
    eval_model = OpenaiModel(model_name='gpt-3.5-turbo', api_keys=api_key)
    

    attacker = Multiverse(
        attack_model=None,
        target_model=target_model,
        target_model_name = target_model_name,
        eval_model=eval_model,
        jailbreak_datasets=dataset,
        scene='dream',
        character_number=5,
        layer_number=5,
        save_path='results',
        log_path=log_path
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
    
    args = parser.parse_args()
    
    api_key = os.getenv('OPEN_API_KEY')
    
    if args.target_model == 'llama-2-7b':
        main_llama(args.target_model, args.start_num, args.end_num)
    elif args.target_model == 'chatglm3-6b':
        main_chatglm3_6b(args.target_model, args.start_num, args.end_num)
    elif args.target_model == 'chatglm2-6b':
        main_chatglm2_6b(args.target_model, args.start_num, args.end_num)
    elif args.target_model == 'vicuna-7b':
        main_vicuna_7b(args.target_model, args.start_num, args.end_num)
    elif args.target_model == 'gpt-4':
        main_gpt_4(args.target_model, args.start_num, args.end_num)
    elif args.target_model == 'qwen-7b':
        main_qwen_7b(args.target_model, args.start_num, args.end_num)
    elif args.target_model == 'gpt-3.5-turbo':
        main_gpt_3_turbo(args.target_model, args.start_num, args.end_num)
    elif args.target_model == 'gemini-1.5-flash':
        main_gemini(args.target_model, args.start_num, args.end_num)