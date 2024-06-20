"""Ensemble all existed policy"""

import logging
import time
from typing import List

from alpa_serve.profiling import ParallelConfig
from alpa_serve.placement_policy.base_policy import (
    BasePlacementPolicy, ModelPlacement, ModelData, ClusterEnv,
    PlacementEvaluator, gen_train_workload, ModelPlacementWithReplacement,
    replica_placement_fast_greedy, replica_placement_beam_search)
from alpa_serve.simulator.workload import Workload
from alpa_serve.util import eps, inf, to_str_round

# import other policies
# sr-greedy, sr-replace, mp-search
from alpa_serve.placement_policy.selective_replication import SelectiveReplicationGreedy, SelectiveReplicationReplacement
from alpa_serve.placement_policy.model_parallelism import ModelParallelismSearch

import random
import ray
import json
import os
        
class TreeNode:
    def __init__(self) -> None:
        self.policy = None

        self.p90 = None
        self.goodput = None
        self.timecost = None
        
        self.device_per_model = None
        self.mem_per_model = None
        
    def to_dict(self):
        return {
            "policy": self.policy,
            "p90": self.p90,
            "device_per_model": self.device_per_model,
            "mem_per_model": self.mem_per_model,
            "goodput": self.goodput,
            "timecost": self.timecost
        }
        
    @classmethod
    def from_dict(cls, dict_data):
        node = cls()
        node.assign_best_performance(
            dict_data["policy"],
            dict_data["p90"],
            dict_data["device_per_model"],
            dict_data["mem_per_model"],
            dict_data["goodput"],
            dict_data["timecost"]
        )
        return node

    def assign_best_performance(self, policy, p90, device_per_model, mem_per_model, goodput, timecost):
        self.policy = policy
        # request detail
        self.device_per_model = device_per_model
        self.mem_per_model = mem_per_model
        # performance
        self.p90 = p90
        self.goodput = goodput
        self.timecost = timecost

    def get_time_value(self):
        return TreeNode.cal_time_value(self.p90, self.goodput, self.timecost)
    
    @staticmethod
    def cal_time_value(p90, goodput, timecost):
        # Define the weight of each necessary value
        alpha = 0.03
        beta = 100
        gamma = 0.5
        return alpha * p90 + beta * goodput + gamma * timecost

          
class DTree:
    def __init__(self) -> None:
        self.close_feature = 0.3
        # Store all existed branches
        self.available_policies = []
        # policy time cost weight dict
        self.time_cost_weight = {}
        # Branch dict
        self.branches = {}
        
        # All necessary layers
        self.arrival_process_model_type = []
        # self.model_type = []
        self.device_per_model = []
        self.mem_per_model = []
        
    def save_status(self, filename: str = "fs.txt"):
        for key in self.branches.keys():
            self.branches[key] = self.branches[key].to_dict()
        with open(filename, "w+") as f:
            json.dump(self.branches, f)
            
    def load_status(self, filename: str = 'fs.txt'):
        if os.path.exists(filename):
            with open(filename, 'r+') as f:
                self.branches = json.load(f) 
            for key in self.branches.keys():
                self.branches[key] = TreeNode.from_dict(self.branches[key])
        

    def init_from_file(self, filename: str):
        with open(filename, 'r+') as f:
            lines = f.readlines()
        records = []
        for line in lines:
            records.append(eval(line))
        
        for record in records:
            self.add_new_record(record)
            
    def search_close_node(self, curr_data, prev_data_list):
        has_node = False
        prev_suit_data = None
        min_distance = 9999999
        
        for prev_data in prev_data_list:
            if curr_data < (1 - self.close_feature) * prev_data and curr_data < (1 + self.close_feature) * prev_data:
                curr_distance = abs(curr_data - prev_data)
                if curr_distance < min_distance:
                    min_distance = curr_distance
                    has_node = True
                    prev_suit_data = prev_data
            if curr_data > (1 + self.close_feature) * prev_data:
                break
        return has_node, prev_suit_data
    
    def add_new_node(self, arrival_process_model_type, device_per_model, mem_per_model, p90, goodput, curr_policy):
        self.arrival_process_model_type.append(arrival_process_model_type)
        # self.model_type.append(model_type)
        self.device_per_model.append(device_per_model)
        self.device_per_model.sort()
        self.mem_per_model.append(mem_per_model)
        self.mem_per_model.sort()

        node = TreeNode()
        node.assign_best_performance(curr_policy, p90=p90, goodput=goodput, timecost=self.time_cost_weight[curr_policy], device_per_model=device_per_model, mem_per_model=mem_per_model)
        self.branches[f'{arrival_process_model_type}-{device_per_model}-{mem_per_model}'] = node
        

    def add_new_record(self, record):
        # curr_policy, prev_policy, load, p99, avg
        # curr_policy, arrival_process, model_type, num_devices, mum_budget, num_models, p90, goodput
        curr_policy = record[0]
        arrival_process_model_type = f"{record[1]}-{record[2]}"
        # model_type = record[2]
        num_devices = record[3]
        num_budget = record[4]
        num_models = record[5]
        
        device_per_model = num_models / num_devices
        mem_per_model = num_budget / num_devices
        
        p90 = record[6]
        goodput = record[7]
        # print(load)
        if len(self.arrival_process_model_type) == 0 or arrival_process_model_type not in self.arrival_process_model_type:
            """
            Init the first node of decision tree
            """
            
            self.add_new_node(arrival_process_model_type,
                              device_per_model,
                              mem_per_model,
                              p90, goodput,
                              curr_policy)

        else:
            # If there has been multiple nodes in the tree
            has_node = False
            prev_suit_model = None
            prev_suit_mem = None
            """
            Start to determine if there are existed suitable model node 
            """
            has_node, prev_suit_model = self.search_close_node(device_per_model, self.device_per_model)
            if has_node:
                """
                Start to determine if there are existed suitable mem node 
                """
                has_node = False
                has_node, prev_suit_mem = self.search_close_node(mem_per_model, self.mem_per_model)
                if has_node:
                    # There are existed node
                    existed_branch_name = f'{arrival_process_model_type}-{prev_suit_model}-{prev_suit_mem}'
                    prev_value = self.branches[existed_branch_name].get_time_value()
                    new_value = TreeNode.cal_time_value(p90, goodput, self.time_cost_weight[curr_policy])
                    if prev_value > new_value:
                        self.add_new_node(arrival_process_model_type,
                                          device_per_model,
                                          mem_per_model,
                                          p90, goodput,
                                          curr_policy)
                else:
                    self.add_new_node(arrival_process_model_type,
                                          device_per_model,
                                          mem_per_model,
                                          p90, goodput,
                                          curr_policy)
            else:
                self.add_new_node(arrival_process_model_type,
                                          device_per_model,
                                          mem_per_model,
                                          p90, goodput,
                                          curr_policy)
                    

    def get_best_policy(self, load, prev_policy):
        has_node = False
        prev_suit_load = None
        min_distance = 9999999
        for prev_load in self.load_layer:
            if prev_load > (1 - self.close_feature) * load and prev_load < (1 + self.close_feature) * load:
                if abs(load - prev_load) < min_distance:
                    min_distance = abs(load - prev_load)
                    has_node = True
                    prev_suit_load = prev_load
            if prev_load > load * (1 + self.close_feature):
                break
        if has_node:
            if f'{prev_suit_load}-{prev_policy}' in self.branches.keys():
                prev_node = self.branches[f'{prev_suit_load}-{prev_policy}']
                return prev_node.policy
            else:
                return None

@ray.remote
class Scheduler_Actor:
    def __init__(self):
        self.policies = {}
        self.time_cost_weight = {}
        
        self.tree = DTree()
        
        self.test_cache = []
        
    def save_status(self, filename: str = "fs.txt"):
        self.tree.save_status(filename)
            
    def load_status(self, filename: str = 'fs.txt'):
        self.tree.load_status(filename)
        
    def init_mp_search(self, policy_name: str):
        # use_evo_search = "evo" in policy_name
        self.tree.time_cost_weight[policy_name] = 21.27
        self.policies[policy_name] = policy_name
        
    def init_sr_greedy(self, policy_name: str):
        self.tree.time_cost_weight[policy_name] = 4.2
        self.policies[policy_name] = policy_name
    
    def init_sr_replace(self, policy_name: str):
        self.tree.time_cost_weight[policy_name] = 26.32
        pn = '-'.join(policy_name.split('-')[:2])
        self.policies[pn] = policy_name
        # interval = int(policy_name.split("-")[2])
        # self.policies[policy_name] = SelectiveReplicationReplacement(verbose=3, replacement_interval=interval)
        
    def print_policies(self):
        print(f"Activated Policies: {self.policies}")
    
    def solve_placement(self,
                        num_models: int,
                        num_devices: int,
                        mem_budget: float):
        # with open("/home/rongyuan/WorkSpace/DLInference/mms/osdi23_artifact/log.txt", 'a') as f:
        #     f.write(f"model datas: {model_datas}\n")
        #     f.write(f"cluster env: budget: {cluster_env.mem_budget}; num_devices: {cluster_env.num_devices}, num_devices_per_node : {cluster_env.num_devices_per_node}\n")
        # Current policy name
        # Placement data from child policies.
        # placement, debug_info = None
        """
        Start to find the current best policy
        """
        device_per_model = num_devices / num_models
        mem_per_model = mem_budget / num_models
        epsilon = 0.3
        if random.random() < epsilon:
            # explore
            random_idx = random.randint(0, len(self.policies.keys()) - 1)
            choice_policy = list(self.policies.keys())[random_idx]
            policy = self.policies[choice_policy]
        else:
            # Using the best policy
            has_best_policy = False
            best_policy_name = None
            if "test-test" in self.tree.arrival_process_model_type:
                for prev_device in self.tree.device_per_model:
                    if device_per_model > prev_device * (1 - self.tree.close_feature) and device_per_model < prev_device * (1 + self.tree.close_feature):
                        for prev_mem in self.tree.mem_per_model:
                            if mem_per_model > prev_mem * (1 - self.tree.close_feature) and mem_per_model < prev_mem * (1 + self.tree.close_feature):
                                key = f"test-test-{prev_device}-{prev_mem}"
                                if key in self.tree.branches.keys():
                                    has_best_policy = True
                                    best_policy_name = self.tree.branches[key].policy
                            elif mem_per_model > prev_mem * (1 + self.tree.close_feature):
                                break
                    elif device_per_model > prev_device * (1 + self.tree.close_feature):
                        break
            if has_best_policy:
                if "sr-replace" in best_policy_name:
                    best_policy_name = "sr-replace"
                policy = self.policies[best_policy_name]
            else:
                random_idx = random.randint(0, len(self.policies.keys()) - 1)
                choice_policy = list(self.policies.keys())[random_idx]
                policy = self.policies[choice_policy]
                        
        return policy
        
        # random_idx = random.randint(0, len(self.policies.keys()) - 1)
        # choice_policy = list(self.policies.keys())[random_idx]
        # policy = self.policies[choice_policy]
        (placement, debug_info) = policy.solve_placement(model_datas, cluster_env, train_workload)
        if debug_info is None:
            debug_info = {}
        debug_info['policy'] = choice_policy
        
        # RY TODO:  The return value of this function should be:
        #           (placement, debug_info)
        #           Moreover, the current policy name should be added
        #           to debug_info for update the tree structure.
        return placement, debug_info
    
    def update_data(self, data:list):
        self.tree.add_new_record(data)
        print(f"decision tree: {self.tree.branches}")
    
    
class DSScheduler(BasePlacementPolicy):
    def __init__(self, verbose: int = 3):
        self.policy_actor = Scheduler_Actor.remote()
        super().__init__(verbose=verbose)
        self.policies = {}
        
        
    def init_mp_search(self, policy_name: str):
        use_evo_search = "evo" in policy_name
        self.policies[policy_name] = ModelParallelismSearch(use_evo_search=use_evo_search, verbose=3)
        ray.get(self.policy_actor.init_mp_search.remote(policy_name))
        
    def init_sr_greedy(self, policy_name: str):
        self.policies[policy_name] = SelectiveReplicationGreedy(verbose=3)
        ray.get(self.policy_actor.init_sr_greedy.remote(policy_name))
    
    def init_sr_replace(self, policy_name: str):
        interval = int(policy_name.split("-")[2])
        self.policies[policy_name] = SelectiveReplicationReplacement(verbose=3, replacement_interval=interval)
        ray.get(self.policy_actor.init_sr_replace.remote(policy_name))
        
    def add_new_policy(self, policy_name: str):
        """
        Init policy dict and corresponding weight dict based on policy name
        params:
            policy_name:    name of policy
        """
        if "sr-greedy" in policy_name:
            self.init_sr_greedy(policy_name)
            return
        if "sr-replace" in policy_name:
            self.init_sr_replace(policy_name=policy_name)
            return
        if "mp-search" in policy_name:
            self.init_mp_search(policy_name=policy_name)
            
    
    
    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        start_time = time.time()
        policy_name = ray.get(self.policy_actor.solve_placement.remote(len(model_datas), cluster_env.num_devices, cluster_env.mem_budget))
        end_time = time.time()
        print(f"Ray communication time: {end_time - start_time}")
        placement, debug_info = self.policies[policy_name].solve_placement(model_datas, cluster_env, train_workload)
        if debug_info is None:
            debug_info = {}
        debug_info['policy'] = policy_name
        return placement, debug_info
    
    
    def update_data(self, data: list):
        ray.get(self.policy_actor.update_data.remote(data))
