"""
DeepInception Class
============================================
This class can easily hypnotize LLM to be a jailbreaker and unlock its 
misusing risks.

Paper title: DeepInception: Hypnotize Large Language Model to Be Jailbreaker
arXiv Link: https://arxiv.org/pdf/2311.03191.pdf
Source repository:  https://github.com/tmlr-group/DeepInception
"""

import logging
logging.basicConfig(level=logging.INFO)
from easyjailbreak.metrics.Evaluator import EvaluatorGenerativeJudge
from easyjailbreak.attacker import AttackerBase
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.mutation.rule import Multiverse_mutation
import os
__all__ = ['Multiverse']

class Multiverse(AttackerBase):
    def __init__(self, attack_model, target_model, target_model_name, eval_model, jailbreak_datasets: JailbreakDataset, layer_number=None, save_path=None, log_path=None, ablation_layer=False):
        r"""
        Initialize the DeepInception attack instance.
        :param attack_model: In this case, the attack_model should be set as None.
        :param target_model: The target language model to be attacked.
        :param eval_model: The evaluation model to evaluate the attack results.
        :param jailbreak_datasets: The dataset to be attacked.
        :param template_file: The file path of the template.
        :param scene: The scene of the deepinception prompt (The default value is 'science fiction').
        :param character_number: The number of characters in the deepinception prompt (The default value is 4).
        :param layer_number: The number of layers in the deepinception prompt (The default value is 5).
        """
        super().__init__(attack_model, target_model, eval_model, jailbreak_datasets)
        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0

        if not ablation_layer:
            self.layer_number = layer_number
            self.evaluator = EvaluatorGenerativeJudge(eval_model)
            self.mutation = Multiverse_mutation(attr_name='query',target_model_name= target_model_name)
        
        else:
            self.layer_number = layer_number
            self.evaluator = EvaluatorGenerativeJudge(eval_model)
            self.mutation = Multiverse_mutation(attr_name='query',target_model_name= target_model_name, 
                                                ablation_layer=ablation_layer, layer_number=layer_number)
        
        self.save_path = save_path
        self.log_path = log_path
        self.configure_logger()


    def configure_logger(self):
        logging.basicConfig(level=logging.INFO)  # Set the logging level
        os.makedirs(self.save_path, exist_ok=True)  
        handler = logging.FileHandler(self.log_path)  # Specify the log file path
        handler.setLevel(logging.INFO)  # Set the handler logging level
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.logger.addHandler(handler)
        
        
        
    def single_attack(self, instance: Instance) -> JailbreakDataset:
        r"""
        single_attack is a method for conducting jailbreak attacks on language models.
        """
        new_instance_list = []
        
        instance_ds = JailbreakDataset([instance])
        new_instance = self.mutation(instance_ds)[-1]
        system_prompt = new_instance.jailbreak_prompt.format(query = new_instance.query)
        
        new_instance.jailbreak_prompt = system_prompt
        answer = self.target_model.generate(system_prompt.format(query = new_instance.query))
        new_instance.target_responses.append(answer)
        new_instance_list.append(new_instance)
        
        return JailbreakDataset(new_instance_list)
    
    def attack(self):
        r"""
        Execute the attack process using provided prompts.
        """
        logging.info("Jailbreak started!")
        self.attack_results = JailbreakDataset([])
        try:
            for i, Instance in enumerate(self.jailbreak_datasets):
                logging.info(f"Processing | {i+1} / {len(self.jailbreak_datasets)}")
                results = self.single_attack(Instance)    

                for new_instance in results:
                    self.attack_results.add(new_instance)
                    
                    
        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")
        self.evaluator(self.attack_results)
        self.update(self.attack_results)
        logging.info("Jailbreak finished!")

    def update(self, Dataset: JailbreakDataset):
        r"""
        Update the state of the Jailbroken based on the evaluation results of Datasets.
        
        :param Dataset: The Dataset that is attacked.
        """
        for prompt_node in Dataset:
            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject

    def log(self):
        r"""
        Report the attack results.
        """
        logging.info("======Jailbreak report:======")
        logging.info(f"Total queries: {self.current_query}")
        logging.info(f"Total jailbreak: {self.current_jailbreak}")
        logging.info(f"Total reject: {self.current_reject}")
        logging.info("========Report End===========")
