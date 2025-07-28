import multiprocessing as mp
import time
import numpy as np
import torch
import copy

from rl4co.envs.common.base import RL4COEnvBase
from tensordict import TensorDict
from rl4co.utils.ops import gather_by_index, unbatchify, batchify
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
)

import numba as nb
from numba import njit, prange
from numba.core.config import NUMBA_NUM_THREADS
import os
CPU_CORES = os.cpu_count()
EVOLUTION_THREADS = min(64, CPU_CORES // 3)
NUMBA_NUM_THREADS = min(12, CPU_CORES // 10)

nb.set_num_threads(NUMBA_NUM_THREADS)

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

def evolution_worker(actions, _td, ea, env):
    """
    Normal evolution worker without any multi-process or multi-thread
    
    Args:
        actions: initial actions / population
        _td: tensor dict used to represent environments
        ea: evolution algorithm runner
        decode_type: decoding type
        env: environment
    """
    batch_size = _td["locs"].shape[0]
    if ea.method != "am":
        multi_start = True
    else:
        multi_start = False
    with torch.no_grad():
        init_td = _td.clone()
        td = _td.clone()
        
        n_start = env.get_num_starts(td)
        
        actions = actions.cpu()
        td = td.cpu()
        if not multi_start:
            assert ea.env_name != "ffsp", "FFSP environment does not support single start evolution"
            # use data augmentation
            actions = actions.unsqueeze(1).cpu().numpy().astype(np.int64) # [batch_size, 1, seq_len]
            if ea.env_name == "tsp":
                env_code = 1
            elif ea.env_name == "cvrp" or ea.env_name == "pctsp":
                env_code = 2
            elif ea.env_name == "op":
                env_code = 3
                
            if np.any(np.all(actions == 0, axis=2)):
                print("Warning: actions contains rows with all zeros.")
            actions = generate_batch_population(actions, env_code, ea.mutation_rate)
        else:
            # use original actions
            actions = unbatchify(actions, n_start).cpu().numpy().astype(np.int64)
        
        new_actions = np.zeros_like(actions)
    
        if ea.env_name == "tsp":
            keys = ['locs']
        elif ea.env_name == "cvrp":
            keys = ['locs', 'demand', 'vehicle_capacity']
        elif ea.env_name == "pctsp":
            keys = ['locs', 'real_prize', 'penalty']
        elif ea.env_name == "op":
            keys = ['locs', 'prize', 'max_length']
        elif ea.env_name == "ffsp":
            keys = ['job_duration', 'schedule', 'job_location', 'run_time']
            
        def process_batch(b):
            batch_actions = actions[b]
            
            env_td = {}
            for key in keys:
                if key in td:
                    env_td[key] = td[key][b:b+1].cpu()
            env_td = TensorDict(env_td, batch_size=[1])
            
            verbose = (b == 0)
                        
            evolved_actions, _ = ea.run(batch_actions, env_td, verbose)
            return b, evolved_actions
        
        with ThreadPoolExecutor(max_workers=min(batch_size, EVOLUTION_THREADS)) as executor:
            futures = [executor.submit(process_batch, b) for b in range(batch_size)]
            for future in concurrent.futures.as_completed(futures):
                b, evolved = future.result()
                evolved_np = np.array(evolved, dtype=np.int64)
                new_actions[b] = torch.from_numpy(evolved_np)
            
        new_actions = torch.as_tensor(new_actions, dtype=torch.int64)
        if multi_start:
            new_actions = new_actions.permute(1, 0, 2)
            new_actions = new_actions.reshape(-1, new_actions.shape[-1])[:, 1:]
        else:
            new_actions = new_actions[:, 0].squeeze(1)
            
        return new_actions, init_td
            
class EA():
    """
    Evolutionary Algorithm, super slow implementation
    """
    def __init__(self,
                 env: RL4COEnvBase,
                 kwargs: dict,
                 ):
        self.env = env
        
        self.num_generations = kwargs.get("num_generations")
        self.mutation_rate = kwargs.get("mutation_rate")
        self.crossover_rate = kwargs.get("crossover_rate")
        self.selection_rate = kwargs.get("selection_rate")
        
        self.method = kwargs.get("method", None)
        
        self.env_name = env.name
        if self.env_name == "tsp":
            self.select_fn = elitism_selection
            self.crossover_fn = order_crossover_tsp
            self.mutate_fn = inverse_mutate_tsp
        elif self.env_name == "cvrp":
            self.select_fn = elitism_selection
            self.crossover_fn = order_crossover_cvrp
            self.mutate_fn = inverse_mutate_cvrp
        elif self.env_name == "pctsp":
            self.select_fn = elitism_selection
            self.crossover_fn = cycle_crossover_pctsp
            self.mutate_fn = inverse_mutate_pctsp
        elif self.env_name == "op":
            self.select_fn = elitism_selection
            self.crossover_fn = order_crossover_op
            self.mutate_fn = inverse_mutate_op
        elif self.env_name == "ffsp":
            self.select_fn = elitism_selection
            self.crossover_fn = multi_point_crossover_ffsp
            self.mutate_fn = swap_mutate_ffsp
        
        assert self.num_generations is not None, "Number of generations must be specified"
        assert self.mutation_rate is not None, "Mutation rate must be specified"
        assert self.crossover_rate is not None, "Crossover rate must be specified"
        assert self.selection_rate is not None, "Selection rate must be specified"
        
        self.reward_fn = self.env.get_reward
        self.cost_fn = self.get_cost
        self.fitness_fn = self.get_fitness
        
    def get_cost(self, pop, td, verbose=False):
        """
        cost = reward * -1
        """
        device = td.device
        
        pop_copy = copy.deepcopy(pop)
        tensor_pop = torch.tensor(pop_copy, device=device, dtype=torch.int64) if not isinstance(pop, torch.Tensor) else pop.to(device=device, dtype=torch.int64)

        pop_size = tensor_pop.shape[0]
        if td.batch_size[0] == 1 and pop_size > 1:
            expanded_td = TensorDict({}, batch_size=[pop_size])
            for key, value in td.items():
                expanded_td[key] = value.expand(pop_size, *value.shape[1:])
                
            assert expanded_td.device == tensor_pop.device, "Devices do not match"
            
            result = -self.reward_fn(expanded_td, tensor_pop).cpu().numpy().astype(np.float32)
            
            return result
        else:
            if td.device != tensor_pop.device:
                tensor_pop = tensor_pop.to(td.device)
        
        result = -self.reward_fn(td, tensor_pop).cpu().numpy().astype(np.float32)
        
        return result

    def get_fitness(self, pop, td, verbose=False):
        safe_pop = copy.deepcopy(pop)
        problem_size = pop.shape[1]
        costs = self.get_cost(safe_pop, td, verbose)
        if self.env_name == "tsp":
            fitness = calculate_fitness_tsp(costs, problem_size)
        elif self.env_name == "cvrp":
            fitness = calculate_fitness_cvrp(costs, problem_size)
        elif self.env_name == "pctsp":
            fitness = calculate_fitness_cvrp(costs, problem_size)
        elif self.env_name == "op":
            fitness = calculate_fitness_op(costs, problem_size)
        elif self.env_name == "ffsp":
            fitness = calculate_fitness_ffsp(costs, problem_size)
        return fitness
        
    def select(self, pop, fitness):
        selected_pop, _ = self.select_fn(pop, fitness, self.selection_rate)
        return selected_pop
    
    def crossover(self, parents, td, dist_mat = None, verbose=False):
        if self.env_name == "tsp":
            return self.crossover_fn(parents, self.crossover_rate)
        elif self.env_name == "cvrp":
            demand = td["demand"].squeeze(-1).squeeze().numpy()
            vehicle_capacity = float(td["vehicle_capacity"].squeeze().numpy())
            offspring = self.crossover_fn(parents, self.crossover_rate, demand, vehicle_capacity)
        
        elif self.env_name == "pctsp":
            penalty = td["penalty"].squeeze(-1).squeeze().numpy()
            prize = td["real_prize"].squeeze(-1).squeeze().numpy()
            offspring = self.crossover_fn(parents, self.crossover_rate, prize, penalty)
            
        elif self.env_name == "op":
            prize = td["prize"].squeeze(-1).squeeze().numpy()
            max_length = td["max_length"].squeeze().numpy()
            offspring = self.crossover_fn(parents, self.crossover_rate, prize, dist_mat, max_length)
          
          
        assert not (offspring == -1).any(), "Invalid offspring generated"
        assert offspring.shape[1] == parents.shape[1], "Offspring shape mismatch"  
        return offspring
    
    def mutate(self, offspring, td, dist_mat = None):
        if self.env_name == "op":
            assert dist_mat is not None, "Distance matrix is required for OP mutation"
            
            prize = td["prize"].squeeze(-1).squeeze().numpy()
            max_length = td["max_length"].squeeze().numpy()
            return self.mutate_fn(offspring, self.mutation_rate, prize, dist_mat, max_length)
        return self.mutate_fn(offspring, self.mutation_rate)
    
    def run(self, init_pop, td, verbose=False):
        def calculate_distance_matrix(locs):
            """
            计算所有节点之间的欧几里得距离矩阵。

            参数:
                locs (numpy.ndarray): 节点坐标数组，形状为 [101, 2]。

            返回:
                numpy.ndarray: 距离矩阵，形状为 [101, 101]。
            """
            locs = locs.squeeze(0)
            diff = locs[:, np.newaxis, :] - locs[np.newaxis, :, :]
            dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))
            return dist_mat
        dist_mat = None
        pop = copy.deepcopy(init_pop)
        
        if self.env_name == "op":
            dist_mat = calculate_distance_matrix(td["locs"].cpu().numpy())
            pop = self.mutate(pop, td, dist_mat)
        elif self.env_name == "cvrp" or self.env_name == "pctsp" or self.env_name == "ffsp":
            pop = self.mutate(pop, td)
        # before_update = copy.deepcopy(pop)
        fitness = self.fitness_fn(pop, td, verbose)
        
        initial_first_nodes = copy.deepcopy(init_pop[:, 0])
        node_to_position = {node: idx for idx, node in enumerate(initial_first_nodes)}
        
        select_time, crossover_time, mutate_time, fitness_time = 0, 0, 0, 0
        
        for i in range(self.num_generations):
            t0 = time.time()
            if pop.shape[0] <= 2:
                selected = pop.copy()
            else:
                selected = self.select(pop, fitness)    
            select_time += time.time() - t0
            
            t0 = time.time()
            # assert not np.any(np.all(selected == 0, axis=1)), f"选择的种群中存在全0的染色体, {i}"
            offspring = self.crossover(selected, td, dist_mat, verbose)
            # assert not np.any(np.all(offspring == 0, axis=1)), f"子代种群中存在全0的染色体, {i}"
            crossover_time += time.time() - t0
            
            t0 = time.time()
            offspring = self.mutate(offspring, td, dist_mat)
            mutate_time += time.time() - t0
            
            if len(offspring) > 0:
                t0 = time.time()
                mutated_fitness = self.fitness_fn(offspring, td, verbose)
                fitness_time += time.time() - t0
                
                pop_cols = pop.shape[1]
                off_cols = offspring.shape[1]
                
                if pop_cols < off_cols:
                    pad_pop = np.zeros((pop.shape[0], off_cols), dtype=pop.dtype)
                    pad_pop[:, :pop_cols] = pop
                    combined_pop = np.vstack([pad_pop, offspring])
                elif pop_cols > off_cols:
                    pad_off = np.zeros((offspring.shape[0], pop_cols), dtype=offspring.dtype)
                    pad_off[:, :off_cols] = offspring
                    combined_pop = np.vstack([pop, pad_off])
                else:
                    combined_pop = np.vstack([pop, offspring])
                combined_fitness = np.concatenate([fitness, mutated_fitness])
                
                if self.method == "am" and (self.env_name == "cvrp" or self.env_name == "pctsp" or self.env_name == "op"):
                    sorted_indices = np.argsort(combined_fitness)[::-1]
                    sorted_pop = combined_pop[sorted_indices]
                    sorted_fitness = combined_fitness[sorted_indices]
                    
                    new_pop = sorted_pop[:pop.shape[0]]
                    new_fitness = sorted_fitness[:pop.shape[0]]
                else:
                    pop_by_first_node = {}
                    for idx, ind in enumerate(combined_pop):
                        first_node = ind[0]
                        if first_node not in pop_by_first_node:
                            pop_by_first_node[first_node] = []
                        pop_by_first_node[first_node].append((idx, combined_fitness[idx]))
                    
                    if pop_cols < off_cols:
                        new_pop = np.zeros((pop.shape[0], off_cols), dtype=pop.dtype)
                    else:
                        new_pop = np.zeros_like(pop)
                    new_fitness = np.zeros_like(fitness)
                    
                    for first_node, candidates in pop_by_first_node.items():
                        if first_node in node_to_position:
                            pos = node_to_position[first_node]
                            candidates.sort(key=lambda x: x[1], reverse=True)
                            best_idx = candidates[0][0]
                            new_pop[pos] = combined_pop[best_idx]
                            new_fitness[pos] = combined_fitness[best_idx]
        
                pop = new_pop
                fitness = new_fitness
                
                assert pop.shape[1] == init_pop.shape[1], "Population shape[1] mismatch"

        return pop, fitness
    
@nb.njit(nb.float32[:](nb.float32[:], nb.int64), nogil=True)
def calculate_fitness_tsp(costs, problem_size):
    result = np.empty_like(costs)
    worst_cost = np.float32(1.5 * problem_size)
    for i in range(len(costs)):
        result[i] = worst_cost - costs[i]
    return result

@nb.njit(nb.float32[:](nb.float32[:], nb.int64), nogil=True)
def calculate_fitness_cvrp(costs, problem_size):
    result = np.empty_like(costs)
    worst_cost = np.float32(2.5 * problem_size)
    for i in range(len(costs)):
        result[i] = worst_cost - costs[i]
    return result
     
@nb.njit(nb.float32[:](nb.float32[:], nb.int64), nogil=True)
def calculate_fitness_op(costs, problem_size):
    result = np.empty_like(costs)
    worst_cost = np.float32(0.0)
    for i in range(len(costs)):
        result[i] = worst_cost - costs[i]
    return result

@nb.njit(nb.float32[:](nb.float32[:], nb.int64), nogil=True)
def calculate_fitness_ffsp(costs, problem_size):
    """
    针对FFSP问题的适应度计算，假设成本是负的
    """
    MAX_COST = 1e5
    result = np.empty_like(costs)
    worst_cost = np.float32(MAX_COST)
    for i in range(len(costs)):
        result[i] = worst_cost - costs[i]
    return result

@nb.njit(nb.int64[:,:](nb.int64[:,:], nb.float32), parallel=True, nogil=True)
def order_crossover_tsp(parents, crossover_rate):
    """
    顺序交叉算法，为numba优化
    
    Args:
        parents: 父代种群
        crossover_rate: 交叉率
    
    Returns:
        后代种群
    """
    pop_size, chrom_length = parents.shape
    
    if pop_size % 2 != 0:
        pop_size -= 1
        
    offspring = np.zeros((pop_size, chrom_length), dtype=nb.int64)
    num_pairs = pop_size // 2
    
    adjusted_rate = crossover_rate
    if num_pairs > 1:
        adjusted_rate = max(0.0, min(1.0, (num_pairs * crossover_rate - 1.0) / (num_pairs - 1)))
    
    cross_rand = np.random.random(num_pairs)
    
    if num_pairs > 0:
        cross_rand[0] = 0.0
    
    for pair_idx in prange(num_pairs):
        p1_idx = pair_idx * 2
        p2_idx = pair_idx * 2 + 1
        
        parent1 = parents[p1_idx]
        parent2 = parents[p2_idx]
        
        current_rate = crossover_rate if pair_idx == 0 else adjusted_rate
        
        if cross_rand[pair_idx] < current_rate:
            # 选择交叉点
            idx1 = np.random.randint(1, chrom_length)
            idx2 = np.random.randint(1, chrom_length)
            
            start = min(idx1, idx2)
            end = max(idx1, idx2)
            
            # 为子代预分配内存
            o1 = np.full(chrom_length, -1, dtype=nb.int32)
            o2 = np.full(chrom_length, -1, dtype=nb.int32)
            
            # 保留首位不变
            o1[0] = parent1[0]
            o2[0] = parent2[0]
            
            # 复制交叉段
            for i in range(start, end):
                o1[i] = parent1[i]
                o2[i] = parent2[i]
            
            # 使用布尔数组跟踪已使用节点
            used1 = np.zeros(chrom_length*2, dtype=nb.boolean)
            used2 = np.zeros(chrom_length*2, dtype=nb.boolean)
            
            # 标记已使用节点
            used1[parent1[0]] = True
            used2[parent2[0]] = True
            
            for i in range(start, end):
                used1[parent1[i]] = True
                used2[parent2[i]] = True
            
            # 从另一个父代填充剩余位置
            pos1 = end % chrom_length
            pos2 = end % chrom_length
            
            for _ in range(chrom_length):
                # 处理第一个子代
                if pos1 != 0 and o1[pos1] == -1: # 跳过首位
                    for j in range(chrom_length):
                        node = parent2[j]
                        if not used1[node]:
                            o1[pos1] = node
                            used1[node] = True
                            break
                
                # 处理第二个子代
                if pos2 != 0 and o2[pos2] == -1: # 跳过首位
                    for j in range(chrom_length):
                        node = parent1[j]
                        if not used2[node]:
                            o2[pos2] = node
                            used2[node] = True
                            break
                            
                pos1 = (pos1 + 1) % chrom_length
                pos2 = (pos2 + 1) % chrom_length
            
            # 转换回无符号整型
            for i in range(chrom_length):
                offspring[p1_idx, i] = o1[i]
                offspring[p2_idx, i] = o2[i]
        else:
            # 直接复制
            for i in range(chrom_length):
                offspring[p1_idx, i] = parent1[i]
                offspring[p2_idx, i] = parent2[i]
    
    return offspring

@nb.njit(nb.int64[:,:](nb.int64[:,:], nb.float64), parallel=True, nogil=True)
def inverse_mutate_tsp(pop, mutation_rate):
    """
    倒置变异算法，根据numba优化
    
    Args:
        pop: 种群矩阵
        mutation_rate: 变异率
    
    Returns:
        变异后的种群
    """
    pop_size, chrom_length = pop.shape
    mutated_pop = pop.copy()
    
    random_vals = np.random.random(pop_size)
    
    for i in prange(pop_size):
        if random_vals[i] < mutation_rate:
            idx1 = np.random.randint(1, chrom_length)
            idx2 = np.random.randint(1, chrom_length)
            
            start = min(idx1, idx2)
            end = max(idx1, idx2)
            
            if start < end:
                segment = np.zeros(end-start, dtype=nb.int64)
                for j in range(end-start):
                    segment[j] = mutated_pop[i, start+j]
                
                for j in range(end-start):
                    mutated_pop[i, start+j] = segment[end-start-j-1]
            elif start < chrom_length - 1:
                temp = mutated_pop[i, start]
                mutated_pop[i, start] = mutated_pop[i, start+1]
                mutated_pop[i, start+1] = temp
    
    return mutated_pop

@nb.njit(nb.int64[:,:](nb.int64[:,:], nb.float64), parallel=True, nogil=True)
def inverse_mutate_cvrp(pop, mutation_rate):
    """
    针对CVRP问题的倒置变异算法，经过numba优化
    
    Args:
        pop: 种群矩阵，包含多个染色体，每个染色体表示一条路径，0表示depot
        mutation_rate: 变异率
    
    Returns:
        变异后的种群
    """
    pop_size, chrom_length = pop.shape
    mutated_pop = pop.copy()
    
    random_vals = np.random.random(pop_size)
    
    for i in prange(pop_size):
        if random_vals[i] < mutation_rate:
            depot_positions = np.zeros(chrom_length, dtype=nb.int64)
            depot_count = 0
            
            for j in range(chrom_length):
                if mutated_pop[i, j] == 0:
                    depot_positions[depot_count] = j
                    depot_count += 1
            
            if depot_count > 1:
                route_idx = np.random.randint(0, depot_count-1)
                start = depot_positions[route_idx] + 1
                end = depot_positions[route_idx+1] - 1
                
                if end - start > 1:
                    seg_start = np.random.randint(start, end)
                    seg_end = np.random.randint(seg_start+1, end+1)
                    
                    if seg_start < seg_end:
                        segment = np.zeros(seg_end-seg_start, dtype=nb.int64)
                        for j in range(seg_end-seg_start):
                            segment[j] = mutated_pop[i, seg_start+j]
                        
                        for j in range(seg_end-seg_start):
                            mutated_pop[i, seg_start+j] = segment[seg_end-seg_start-j-1]
    
    return mutated_pop

@nb.njit(nb.int64[:,:](nb.int64[:,:], nb.float64), parallel=True, nogil=True)
def inverse_mutate_pctsp(pop, mutation_rate):
    """
    针对CVRP问题的倒置变异算法，经过numba优化
    
    Args:
        pop: 种群矩阵，包含多个染色体，每个染色体表示一条路径，0表示depot
        mutation_rate: 变异率
    
    Returns:
        变异后的种群
    """
    pop_size, chrom_length = pop.shape
    mutated_pop = pop.copy()
    
    random_vals = np.random.random(pop_size)
    
    for i in prange(pop_size):
        if random_vals[i] < mutation_rate:
            valid_len = np.max(np.where(mutated_pop[i] != 0)[0])
            
            idx1 = np.random.randint(1, valid_len)
            idx2 = np.random.randint(1, valid_len)
            
            start = min(idx1, idx2)
            end = max(idx1, idx2)
            
            if start < end:
                segment = np.zeros(end-start, dtype=nb.int64)
                for j in range(end-start):
                    segment[j] = mutated_pop[i, start+j]
                
                for j in range(end-start):
                    mutated_pop[i, start+j] = segment[end-start-j-1]
            elif start < chrom_length - 1:
                temp = mutated_pop[i, start]
                mutated_pop[i, start] = mutated_pop[i, start+1]
                mutated_pop[i, start+1] = temp
    return mutated_pop

@nb.njit(nb.int64[:,:](nb.int64[:,:], nb.float64, nb.float32[:], nb.float64), parallel=True, nogil=True)
def order_crossover_cvrp(parents, crossover_rate, demand, vehicle_capacity):
    """
    针对CVRP问题的顺序交叉算子，考虑容量约束，并保持每个染色体的第一个节点不变
    
    Args:
        parents: 父代种群矩阵，每行是一个染色体，0表示depot
        crossover_rate: 交叉率
        demand: 每个客户的需求量数组（不包含仓库节点）
        vehicle_capacity: 车辆容量限制
    
    Returns:
        后代种群
    """
    pop_size, chrom_length = parents.shape
    
    # 创建包含仓库需求的新数组（Numba兼容方式）
    num_customers = len(demand)
    full_demand = np.zeros(num_customers+1, dtype=np.float32)
    for i in range(num_customers):
        full_demand[i+1] = demand[i]
    # full_demand[0] = 0.0 默认已经是0
    
    if pop_size % 2 != 0:
        pop_size -= 1
        
    offspring = np.zeros((pop_size, chrom_length), dtype=nb.int64)
    num_pairs = pop_size // 2
    
    adjusted_rate = crossover_rate
    if num_pairs > 1:
        adjusted_rate = max(0.0, min(1.0, (num_pairs * crossover_rate - 1.0) / (num_pairs - 1)))
    
    cross_rand = np.random.random(num_pairs)
    
    if num_pairs > 0:
        cross_rand[0] = 0.0
        
    for pair_idx in prange(num_pairs):
        p1_idx = pair_idx * 2
        p2_idx = pair_idx * 2 + 1
        
        parent1 = parents[p1_idx]
        parent2 = parents[p2_idx]
        
        current_rate = crossover_rate if pair_idx == 0 else adjusted_rate
        
        if cross_rand[pair_idx] < current_rate:
            # 找到有效的父代路径长度
            p1_valid_end = np.max(np.where(parent1 != 0)[0]) + 1 if np.any(parent1 != 0) else 1
            p2_valid_end = np.max(np.where(parent2 != 0)[0]) + 1 if np.any(parent2 != 0) else 1
        
            # 计算路径数量
            p1_route_num = np.count_nonzero(parent1[:p1_valid_end] == 0)
            p2_route_num = np.count_nonzero(parent2[:p2_valid_end] == 0)
            min_route_num = min(p1_route_num, p2_route_num)
            
            # 随机选择交叉点
            end = np.random.randint(1, min_route_num) if min_route_num > 1 else 0
            
            # 找到对应的索引位置
            end_idx1 = np.where(parent1 == 0)[0][end] if end > 0 else 0
            end_idx2 = np.where(parent2 == 0)[0][end] if end > 0 else 0
            
            # 初始化子代
            o1 = np.full(chrom_length*2, -1, dtype=nb.int64)
            o2 = np.full(chrom_length*2, -1, dtype=nb.int64)
            
            # 从父代复制部分路径
            for j in range(end_idx1):
                o1[j] = parent1[j]
            for j in range(end_idx2):
                o2[j] = parent2[j]
            
            # 记录已使用节点
            used1 = np.zeros(num_customers+1, dtype=nb.boolean)
            used2 = np.zeros(num_customers+1, dtype=nb.boolean)
            
            # 仓库节点不需要标记为已使用（可重复）
            for j in range(end_idx1):
                if o1[j] > 0:  # 只标记非depot节点
                    used1[o1[j]] = True
            for j in range(end_idx2):
                if o2[j] > 0:  # 只标记非depot节点
                    used2[o2[j]] = True
            
            pos1 = end_idx1
            pos2 = end_idx2
            
            # 确保从仓库开始新路径
            if pos1 > 0 and o1[pos1-1] != 0:
                o1[pos1] = 0
                pos1 += 1
                
            if pos2 > 0 and o2[pos2-1] != 0:
                o2[pos2] = 0
                pos2 += 1
            
            o1_current_load = 0.0
            o2_current_load = 0.0
            
            # 创建剩余未使用节点列表 - 使用预分配的numpy数组
            remaining_nodes1 = np.zeros(num_customers, dtype=nb.int64)
            remaining_nodes2 = np.zeros(num_customers, dtype=nb.int64)
            count1 = 0
            count2 = 0
            
            for i in range(1, num_customers+1):
                if not used1[i]:
                    remaining_nodes1[count1] = i
                    count1 += 1
                if not used2[i]:
                    remaining_nodes2[count2] = i
                    count2 += 1
            
            # 填充第一个子代
            for i in range(count1):
                node = remaining_nodes1[i]
                if pos1 >= chrom_length*2 - 1:
                    break
                
                # 检查容量约束
                if o1_current_load + full_demand[node] > vehicle_capacity:
                    # 添加仓库节点，开始新路径
                    # 但要确保不会在还有节点未访问时连续访问仓库
                    if pos1 > 0 and o1[pos1-1] == 0 and i < count1-1:
                        # 如果上一个是仓库且还有节点未访问，则不添加仓库节点
                        # 改为添加当前节点，并在下一轮重新考虑容量约束
                        continue
                    
                    o1[pos1] = 0
                    pos1 += 1
                    o1_current_load = 0.0
                    
                    if pos1 >= chrom_length*2 - 1:
                        break
                
                # 添加客户节点
                o1[pos1] = node
                o1_current_load += full_demand[node]
                pos1 += 1
            
            # 确保路径以仓库结束，但只有在所有节点都被访问后才能添加最后的仓库节点
            if pos1 < chrom_length*2 and o1[pos1-1] != 0:
                # 检查是否所有客户节点都已访问完毕
                all_visited1 = True
                for i in range(1, num_customers+1):
                    if not used1[i] and i < count1 and remaining_nodes1[i] > 0:
                        all_visited1 = False
                        break
                
                if all_visited1:
                    o1[pos1] = 0
                    pos1 += 1
            
            # 填充第二个子代
            for i in range(count2):
                node = remaining_nodes2[i]
                if pos2 >= chrom_length*2 - 1:
                    break
                
                # 检查容量约束
                if o2_current_load + full_demand[node] > vehicle_capacity:
                    # 添加仓库节点，开始新路径
                    # 但要确保不会在还有节点未访问时连续访问仓库
                    if pos2 > 0 and o2[pos2-1] == 0 and i < count2-1:
                        # 如果上一个是仓库且还有节点未访问，则不添加仓库节点
                        # 改为添加当前节点，并在下一轮重新考虑容量约束
                        continue
                    
                    o2[pos2] = 0
                    pos2 += 1
                    o2_current_load = 0.0
                    
                    if pos2 >= chrom_length*2 - 1:
                        break
                
                # 添加客户节点
                o2[pos2] = node
                o2_current_load += full_demand[node]
                pos2 += 1
            
            # 确保路径以仓库结束，但只有在所有节点都被访问后才能添加最后的仓库节点
            if pos2 < chrom_length*2 and o2[pos2-1] != 0:
                # 检查是否所有客户节点都已访问完毕
                all_visited2 = True
                for i in range(1, num_customers+1):
                    if not used2[i] and i < count2 and remaining_nodes2[i] > 0:
                        all_visited2 = False
                        break
                
                if all_visited2:
                    o2[pos2] = 0
                    pos2 += 1
            
            # 验证子代是否符合约束：不存在连续的仓库访问且存在未访问的客户节点
            has_invalid1 = False
            has_invalid2 = False
            
            # 检查o1是否有连续仓库访问
            for j in range(1, pos1):
                if o1[j] == 0 and o1[j-1] == 0:
                    # 检查是否还有未访问的节点
                    for i in range(count1):
                        if remaining_nodes1[i] > 0 and not used1[remaining_nodes1[i]]:
                            has_invalid1 = True
                            break
                    if has_invalid1:
                        break
                    
            # 检查o2是否有连续仓库访问
            for j in range(1, pos2):
                if o2[j] == 0 and o2[j-1] == 0:
                    # 检查是否还有未访问的节点
                    for i in range(count2):
                        if remaining_nodes2[i] > 0 and not used2[remaining_nodes2[i]]:
                            has_invalid2 = True
                            break
                    if has_invalid2:
                        break
            
            # 如果有无效解，则使用原父代
            if has_invalid1:
                for j in range(chrom_length):
                    offspring[p1_idx, j] = parent1[j]
            else:
                o1_last_valid_idx = np.max(np.where(o1 != -1)[0])
                if o1_last_valid_idx >= chrom_length:
                    for j in range(chrom_length):
                        offspring[p1_idx, j] = parent1[j]
                else:
                    for i in range(chrom_length):
                        offspring[p1_idx, i] = o1[i] if o1[i] != -1 else 0
            
            if has_invalid2:
                for j in range(chrom_length):
                    offspring[p2_idx, j] = parent2[j]
            else:
                o2_last_valid_idx = np.max(np.where(o2 != -1)[0])
                if o2_last_valid_idx >= chrom_length:
                    for j in range(chrom_length):
                        offspring[p2_idx, j] = parent2[j]
                else:
                    for i in range(chrom_length):
                        offspring[p2_idx, i] = o2[i] if o2[i] != -1 else 0
            
        else:
            for j in range(chrom_length):
                offspring[p1_idx, j] = parent1[j]
                offspring[p2_idx, j] = parent2[j]
                
    return offspring

@nb.njit(nb.int64[:,:](nb.int64[:,:], nb.float64, nb.float32[:], nb.float32[:]), parallel=True, nogil=True)
def order_crossover_pctsp(parents, crossover_rate, prize, penalty):
    """
    针对PCTSP问题的顺序交叉算子，考虑容量和奖赏约束，并保持每个染色体的第一个节点不变
    
    Args:
        parents: 父代种群矩阵，每行是一个染色体，0表示depot
        crossover_rate: 交叉率
        prize: 每个客户的奖赏值数组[num_customer + 1]
        penalty: 惩罚数值[num_customer + 1]
    
    Returns:
        后代种群
    """
    pop_size, chrom_length = parents.shape
    
    # 创建包含仓库需求和奖赏的新数组（Numba兼容方式）
    num_customers = len(prize) - 1
    # prize_penalty_ratio = np.zeros(len(prize), dtype = nb.float64)
    # for i in range(1, num_customers+1):
    #     prize_penalty_ratio = prize[i] / penalty[i]
    
    
    if pop_size % 2 != 0:
        pop_size -= 1
        
    offspring = np.zeros((pop_size, chrom_length), dtype=nb.int64)
    num_pairs = pop_size // 2
    
    adjusted_rate = crossover_rate
    if num_pairs > 1:
        adjusted_rate = max(0.0, min(1.0, (num_pairs * crossover_rate - 1.0) / (num_pairs - 1)))
    
    cross_rand = np.random.random(num_pairs)
    
    if num_pairs > 0:
        cross_rand[0] = 0.0
        
    for pair_idx in prange(num_pairs):
        p1_idx = pair_idx * 2
        p2_idx = pair_idx * 2 + 1
        
        parent1 = parents[p1_idx]
        parent2 = parents[p2_idx]
        
        current_rate = crossover_rate if pair_idx == 0 else adjusted_rate
        
        if cross_rand[pair_idx] < current_rate:
            end = np.random.randint(1, chrom_length)
            
            # 初始化子代
            o1 = np.full(chrom_length*2, -1, dtype=nb.int64)
            o2 = np.full(chrom_length*2, -1, dtype=nb.int64)
            
            # 从父代复制部分路径
            for j in range(0, end):
                o1[j] = parent1[j]
                o2[j] = parent2[j]
            
            # 记录已使用节点
            used1 = np.zeros(num_customers+1, dtype=nb.boolean)
            used2 = np.zeros(num_customers+1, dtype=nb.boolean)
            
            # 仓库节点不需要标记为已使用（可重复）
            for j in range(end):
                if o1[j] > 0:  # 只标记非depot节点
                    used1[o1[j]] = True
                if o2[j] > 0:  # 只标记非depot节点
                    used2[o2[j]] = True
            
            pos1 = end
            pos2 = end
            
            o1_current_prize = 0.0
            o2_current_prize = 0.0
            
            # 创建剩余未使用节点列表 - 使用预分配的numpy数组
            remaining_nodes1 = np.zeros(num_customers, dtype=nb.int64)
            remaining_nodes2 = np.zeros(num_customers, dtype=nb.int64)
            count1 = 0
            count2 = 0
            
            for i in range(1, num_customers+1):
                if not used1[i]:
                    remaining_nodes1[count1] = i
                    count1 += 1
                if not used2[i]:
                    remaining_nodes2[count2] = i
                    count2 += 1
            
            # 填充第一个子代
            for i in range(count1):
                node = remaining_nodes1[i]
                if pos1 >= chrom_length*2 - 1:
                    break
                
                # 添加客户节点
                o1[pos1] = node
                o1_current_prize += prize[node]
                pos1 += 1
                
                # 如果奖赏已满足要求，可以提前结束
                if o1_current_prize >= 1 - 1e-5:
                    break
            
            for i in range(count2):
                node = remaining_nodes2[i]
                if pos2 >= chrom_length*2 - 1:
                    break
                
                o2[pos2] = node
                o2_current_prize += prize[node]
                pos2 += 1
                
                if o2_current_prize >= 1 - 1e-5:
                    break

            o1_last_valid_idx = np.max(np.where(o1 != -1)[0])
            if o1_last_valid_idx >= chrom_length:
                for j in range(chrom_length):
                    offspring[p1_idx, j] = parent1[j]
            else:
                for i in range(chrom_length):
                    offspring[p1_idx, i] = o1[i] if o1[i] != -1 else 0
            
            o2_last_valid_idx = np.max(np.where(o2 != -1)[0])
            if o2_last_valid_idx >= chrom_length:
                for j in range(chrom_length):
                    offspring[p2_idx, j] = parent2[j]
            else:
                for i in range(chrom_length):
                    offspring[p2_idx, i] = o2[i] if o2[i] != -1 else 0
        else:
            for j in range(chrom_length):
                offspring[p1_idx, j] = parent1[j]
                offspring[p2_idx, j] = parent2[j]
                
    return offspring

@nb.njit(nb.int64[:,:](nb.int64[:,:], nb.float64, nb.float32[:], nb.float32[:]), parallel=True, nogil=True)
def cycle_crossover_pctsp(parents, crossover_rate, prize, penalty):
    """
    针对PCTSP问题的循环交叉算子，考虑奖赏约束
    
    Args:
        parents: 父代种群矩阵，每行是一个染色体，0表示depot（但不应出现在起始位置）
        crossover_rate: 交叉率
        prize: 每个客户的奖赏值数组[num_customer + 1]，索引0对应depot
        penalty: 惩罚数值[num_customer + 1]，索引0对应depot
    
    Returns:
        后代种群
    """
    pop_size, chrom_length = parents.shape
    
    # 创建包含仓库需求和奖赏的新数组
    num_customers = len(prize) - 1
    
    if pop_size % 2 != 0:
        pop_size -= 1
        
    offspring = np.zeros((pop_size, chrom_length), dtype=nb.int64)
    num_pairs = pop_size // 2
    
    adjusted_rate = crossover_rate
    if num_pairs > 1:
        adjusted_rate = max(0.0, min(1.0, (num_pairs * crossover_rate - 1.0) / (num_pairs - 1)))
    
    cross_rand = np.random.random(num_pairs)
    
    if num_pairs > 0:
        cross_rand[0] = 0.0
        
    for pair_idx in prange(num_pairs):
        p1_idx = pair_idx * 2
        p2_idx = pair_idx * 2 + 1
        
        parent1 = parents[p1_idx]
        parent2 = parents[p2_idx]
        
        current_rate = crossover_rate if pair_idx == 0 else adjusted_rate
        
        if cross_rand[pair_idx] < current_rate:
            # 初始化子代
            o1 = np.full(chrom_length, 0, dtype=nb.int64)  # 用0填充而不是-1
            o2 = np.full(chrom_length, 0, dtype=nb.int64)
            
            # 初始化已访问节点集合
            used_nodes1 = np.zeros(num_customers+1, dtype=nb.boolean)
            used_nodes2 = np.zeros(num_customers+1, dtype=nb.boolean)
            
            # 找到有效的父代长度（排除末尾的填充0）
            p1_valid_end = 0
            for i in range(chrom_length-1, -1, -1):
                if parent1[i] != 0:
                    p1_valid_end = i + 1
                    break
                    
            p2_valid_end = 0
            for i in range(chrom_length-1, -1, -1):
                if parent2[i] != 0:
                    p2_valid_end = i + 1
                    break
            
            # 收集父代中的非零节点
            p1_nodes = []
            for i in range(p1_valid_end):
                if parent1[i] > 0:  # 只收集非depot节点
                    p1_nodes.append(parent1[i])
            
            p2_nodes = []
            for i in range(p2_valid_end):
                if parent2[i] > 0:
                    p2_nodes.append(parent2[i])
                    
            # 构建映射
            node_to_pos1 = {}
            for i, node in enumerate(p1_nodes):
                if node > 0:  # 只映射非depot节点
                    node_to_pos1[node] = i
                    
            node_to_pos2 = {}
            for i, node in enumerate(p2_nodes):
                if node > 0:
                    node_to_pos2[node] = i
            
            # 执行循环交叉
            cycles = []  # 存储找到的所有循环
            remaining_nodes = set(p1_nodes)
            
            while remaining_nodes:
                cycle = []
                start_node = next(iter(remaining_nodes))
                node = start_node
                
                while True:
                    cycle.append(node)
                    remaining_nodes.remove(node)
                    
                    # 找到这个节点在第二个父代中的位置
                    if node not in node_to_pos2:
                        break
                    pos2 = node_to_pos2[node]
                    
                    # 找到这个位置在第一个父代中的节点
                    if pos2 >= len(p1_nodes):
                        break
                    node = p1_nodes[pos2]
                    
                    if node == start_node or node not in remaining_nodes:
                        break
                
                cycles.append(cycle)
            
            # 根据循环交替填充子代
            o1_nodes = []
            o2_nodes = []
            
            for i, cycle in enumerate(cycles):
                if i % 2 == 0:  # 偶数循环
                    o1_nodes.extend(cycle)
                    # 在o2中找到对应位置的节点
                    for node in cycle:
                        if node in node_to_pos2:
                            pos = node_to_pos2[node]
                            if pos < len(p2_nodes):
                                o2_nodes.append(p2_nodes[pos])
                else:  # 奇数循环
                    o2_nodes.extend(cycle)
                    # 在o1中找到对应位置的节点
                    for node in cycle:
                        if node in node_to_pos1:
                            pos = node_to_pos1[node]
                            if pos < len(p1_nodes):
                                o1_nodes.append(p1_nodes[pos])
            
            # 计算奖励和移除重复节点
            o1_prize = 0.0
            o1_unique = []
            used_nodes1 = np.zeros(num_customers+1, dtype=nb.boolean)
            
            for node in o1_nodes:
                if node > 0 and not used_nodes1[node]:
                    o1_unique.append(node)
                    used_nodes1[node] = True
                    o1_prize += prize[node]
            
            o2_prize = 0.0
            o2_unique = []
            used_nodes2 = np.zeros(num_customers+1, dtype=nb.boolean)
            
            for node in o2_nodes:
                if node > 0 and not used_nodes2[node]:
                    o2_unique.append(node)
                    used_nodes2[node] = True
                    o2_prize += prize[node]
            
            # 检查并添加额外节点以满足奖励约束
            if o1_prize < 1.0 - 1e-5:
                # 计算非访问节点的奖励/惩罚比
                prize_ratios = np.zeros(num_customers+1, dtype=nb.float32)
                
                for i in range(1, num_customers+1):
                    if not used_nodes1[i]:
                        prize_ratios[i] = prize[i] / (penalty[i] + 1e-10)
                
                # 按奖励/惩罚比添加节点
                while o1_prize < 1.0 - 1e-5:
                    best_node = 0
                    best_ratio = -1.0
                    
                    for i in range(1, num_customers+1):
                        if not used_nodes1[i] and prize_ratios[i] > best_ratio:
                            best_ratio = prize_ratios[i]
                            best_node = i
                    
                    if best_node == 0:
                        break
                    
                    o1_unique.append(best_node)
                    used_nodes1[best_node] = True
                    o1_prize += prize[best_node]
            
            # 同样处理o2
            if o2_prize < 1.0 - 1e-5:
                prize_ratios = np.zeros(num_customers+1, dtype=nb.float32)
                
                for i in range(1, num_customers+1):
                    if not used_nodes2[i]:
                        prize_ratios[i] = prize[i] / (penalty[i] + 1e-10)
                
                while o2_prize < 1.0 - 1e-5:
                    best_node = 0
                    best_ratio = -1.0
                    
                    for i in range(1, num_customers+1):
                        if not used_nodes2[i] and prize_ratios[i] > best_ratio:
                            best_ratio = prize_ratios[i]
                            best_node = i
                    
                    if best_node == 0:
                        break
                    
                    o2_unique.append(best_node)
                    used_nodes2[best_node] = True
                    o2_prize += prize[best_node]
            
            # 将唯一节点复制到子代染色体
            for i, node in enumerate(o1_unique):
                if i < chrom_length:
                    o1[i] = node
            
            for i, node in enumerate(o2_unique):
                if i < chrom_length:
                    o2[i] = node
            
            # 最后，检查是否已经满足奖励约束，只有满足约束才能添加depot(0)作为终点
            # 在PCTSP中，路径隐含从depot出发，最后一个节点后会自动返回depot
            
            # 复制到输出数组
            for j in range(chrom_length):
                offspring[p1_idx, j] = o1[j]
                offspring[p2_idx, j] = o2[j]
                
        else:
            # 如果不交叉，直接复制父代
            for j in range(chrom_length):
                offspring[p1_idx, j] = parent1[j]
                offspring[p2_idx, j] = parent2[j]
                
    return offspring

@nb.njit(nb.types.Tuple((nb.int64[:,:], nb.float32[:]))(nb.int64[:,:], nb.float32[:], nb.float64), nogil=True)
def elitism_selection(pop, fitness, selection_rate):
    num_elites = int(selection_rate * pop.shape[0])
    idx = np.argsort(fitness)
    elite_idx = idx[-num_elites:]
    return pop[elite_idx], fitness[elite_idx]

@nb.njit(nb.int64[:,:](nb.int64[:,:], nb.float64, nb.float32[:], nb.float32[:,:], nb.float32[:]), parallel=True, nogil=True)
def order_crossover_op(parents, crossover_rate, prize, dist_matrix, max_distances):
    """
    针对定向越野问题(OP, Orienteering Problem)的顺序交叉算子的NumPy版本
    考虑全局距离约束，不限制起点，但终点必须是depot(0)
    
    Args:
        parents: 父代种群矩阵，每行是一个染色体，最后一个非0节点应该是depot(0)
        crossover_rate: 交叉率
        prize: 每个节点的奖励值数组[num_nodes]
        dist_matrix: 距离矩阵[num_nodes, num_nodes]
        max_distances: 全局最大距离约束[num_nodes]，其中max_distances[0]是原始约束
    
    Returns:
        后代种群
    """
    pop_size, chrom_length = parents.shape
    
    # 节点数量
    num_nodes = len(prize)
    
    if pop_size % 2 != 0:
        pop_size -= 1
        
    offspring = np.zeros((pop_size, chrom_length), dtype=nb.int64)
    num_pairs = pop_size // 2
    
    # 为节点奖励/距离比率创建数组
    # ratio1 = np.zeros(num_nodes, dtype=nb.float32)
    # ratio2 = np.zeros(num_nodes, dtype=nb.float32)
    
    # 使用全局最大距离约束(max_distances[0])
    global_max_dist = max_distances[0]
    
    # 添加安全裕度，确保不会超出环境的距离约束
    safe_max_dist = global_max_dist - 0.1  # 减去0.5作为足够大的安全裕度
    
    # 调整交叉率，确保至少一对父代会进行交叉
    adjusted_rate = crossover_rate
    if num_pairs > 1:
        adjusted_rate = max(0.0, min(1.0, (num_pairs * crossover_rate - 1.0) / (num_pairs - 1)))
    
    cross_rand = np.random.random(num_pairs)
    
    if num_pairs > 0:
        cross_rand[0] = 0.0  # 确保至少有一对父代进行交叉
        
    for pair_idx in range(num_pairs):
        p1_idx = pair_idx * 2
        p2_idx = pair_idx * 2 + 1
        
        parent1 = parents[p1_idx].copy()
        parent2 = parents[p2_idx].copy()
        
        current_rate = crossover_rate if pair_idx == 0 else adjusted_rate
        
        if cross_rand[pair_idx] < current_rate:
            # 找到有效的父代路径长度（排除末尾可能的0）
            p1_valid_end = len(parent1)
            for j in range(len(parent1)-1, 0, -1):
                if parent1[j] != 0:
                    p1_valid_end = j + 1
                    break
                    
            p2_valid_end = len(parent2)
            for j in range(len(parent2)-1, 0, -1):
                if parent2[j] != 0:
                    p2_valid_end = j + 1
                    break
            
            # 确保父代路径是有效的（以depot结束）
            if p1_valid_end == 0 or p2_valid_end == 0 or parent1[p1_valid_end-1] != 0 or parent2[p2_valid_end-1] != 0:
                # 如果父代路径无效，则直接复制
                offspring[p1_idx] = parent1
                offspring[p2_idx] = parent2
                continue
                
            # 设置终点为depot(0)
            p1_end_node = 0
            p2_end_node = 0
            
            # 随机选择交叉点（不包括起点和终点）
            # 确保交叉点有效
            max_cross_point = min(p1_valid_end-1, p2_valid_end-1, chrom_length-1)
            if max_cross_point <= 1:
                # 如果没有足够的节点进行交叉，则直接复制
                offspring[p1_idx] = parent1
                offspring[p2_idx] = parent2
                continue
                
            end = np.random.randint(1, max_cross_point)
            
            # 初始化子代，使用大一些的数组以防溢出
            o1 = np.full(chrom_length*2, -1, dtype=nb.int64)
            o2 = np.full(chrom_length*2, -1, dtype=nb.int64)
            
            # 从父代复制部分路径（包括起点）
            for j in range(0, end):
                o1[j] = parent1[j]
                o2[j] = parent2[j]
            
            # 记录已使用节点
            used1 = np.zeros(num_nodes, dtype=nb.boolean)
            used2 = np.zeros(num_nodes, dtype=nb.boolean)
            
            # 标记已使用的节点
            for j in range(end):
                if o1[j] != 0:  # 排除depot，因为depot可以多次访问（作为终点）
                    used1[o1[j]] = True
                if o2[j] != 0:
                    used2[o2[j]] = True
            
            # 当前位置和当前距离
            pos1 = end
            pos2 = end
            
            o1_current_dist = 0.0
            o2_current_dist = 0.0
            
            # 计算已用路径的距离
            for j in range(1, end):
                o1_current_dist += dist_matrix[o1[j-1], o1[j]]
                o2_current_dist += dist_matrix[o2[j-1], o2[j]]
            o1_current_dist += dist_matrix[0, o1[0]]  # depot到第一个节点的距离
            o2_current_dist += dist_matrix[0, o2[0]]
            
            # 初始化剩余节点的奖励/距离比率
            # for i in range(num_nodes):
            #     if not used1[i] and i != 0:  # 排除depot和已使用节点
            #         # 计算从当前位置到该节点的距离
            #         dist_to_node = dist_matrix[o1[pos1-1], i]
            #         # 计算从该节点到depot的距离
            #         dist_to_end = dist_matrix[i, 0]
            #         # 如果加入该节点后总路径长度会超出预算，则设置比率为-1
            #         if o1_current_dist + dist_to_node + dist_to_end > safe_max_dist:
            #             ratio1[i] = -1
            #         else:
            #             # 计算奖励/距离比率
            #             ratio1[i] = prize[i] / dist_to_node if dist_to_node > 0 else prize[i] * 1000
            #     else:
            #         ratio1[i] = -1
                    
            #     if not used2[i] and i != 0:  # 排除depot和已使用节点
            #         dist_to_node = dist_matrix[o2[pos2-1], i]
            #         dist_to_end = dist_matrix[i, 0]
            #         if o2_current_dist + dist_to_node + dist_to_end > safe_max_dist:
            #             ratio2[i] = -1
            #         else:
            #             ratio2[i] = prize[i] / dist_to_node if dist_to_node > 0 else prize[i] * 1000
            #     else:
            #         ratio2[i] = -1
            
            remaining_nodes1 = np.zeros(chrom_length, dtype=nb.int64)
            remaining_nodes2 = np.zeros(chrom_length, dtype=nb.int64)
            count1 = 0
            count2 = 0
            
            for i in range(1, chrom_length+1):
                if not used1[i]:
                    remaining_nodes1[count1] = i
                    count1 += 1
                if not used2[i]:
                    remaining_nodes2[count2] = i
                    count2 += 1
            
            # 贪婪地选择剩余节点
            # while True:
            for j in range(count1):
                # 找到最佳的下一个节点
                # best_node1 = np.argmax(ratio1)
                best_node1 = remaining_nodes1[j]
                
                # 如果所有节点都被使用或者没有可行节点，结束循环
                # if ratio1[best_node1] < 0:
                #     break
                    
                # 计算加入该节点后的总距离
                next_dist1 = dist_matrix[o1[pos1-1], best_node1]
                dist_to_end1 = dist_matrix[best_node1, 0]
                
                # 再次检查是否仍在预算内 - 使用安全裕度
                # 这里做双重检查，避免浮点误差
                if o1_current_dist + next_dist1 + dist_to_end1 <= safe_max_dist:
                    o1[pos1] = best_node1
                    o1_current_dist += next_dist1
                    used1[best_node1] = True
                    pos1 += 1
                    
                    # 更新其他节点的比率
                #     for i in range(num_nodes):
                #         if not used1[i] and i != 0:
                #             dist_to_node = dist_matrix[best_node1, i]
                #             dist_to_end = dist_matrix[i, 0]
                #             if o1_current_dist + dist_to_node + dist_to_end > safe_max_dist:
                #                 ratio1[i] = -1
                #             else:
                #                 ratio1[i] = prize[i] / dist_to_node if dist_to_node > 0 else prize[i] * 1000
                # else:
                #     # 标记该节点为不可用
                #     ratio1[best_node1] = -1
                
                # 如果已经达到染色体长度限制，结束循环
                if pos1 >= chrom_length*2 - 2:
                    break
            
            # 为第二个子代做同样的事情
            # while True:
            for j in range(count2):
                # best_node2 = np.argmax(ratio2)
                best_node2 = remaining_nodes2[j]
                
                # if ratio2[best_node2] < 0:
                #     break
                    
                next_dist2 = dist_matrix[o2[pos2-1], best_node2]
                dist_to_end2 = dist_matrix[best_node2, 0]
                
                # 再次检查是否仍在预算内 - 使用安全裕度
                if o2_current_dist + next_dist2 + dist_to_end2 <= safe_max_dist:
                    o2[pos2] = best_node2
                    o2_current_dist += next_dist2
                    used2[best_node2] = True
                    pos2 += 1
                    
                #     for i in range(num_nodes):
                #         if not used2[i] and i != 0:
                #             dist_to_node = dist_matrix[best_node2, i]
                #             dist_to_end = dist_matrix[i, 0]
                #             if o2_current_dist + dist_to_node + dist_to_end > safe_max_dist:
                #                 ratio2[i] = -1
                #             else:
                #                 ratio2[i] = prize[i] / dist_to_node if dist_to_node > 0 else prize[i] * 1000
                # else:
                #     ratio2[best_node2] = -1
                
                if pos2 >= chrom_length*2 - 2:
                    break
            
            # 添加终点节点（固定为depot）
            o1[pos1] = 0  # depot
            o2[pos2] = 0  # depot
            pos1 += 1
            pos2 += 1
            
            # 处理第一个子代
            # 找到有效索引范围
            valid_indices1 = np.where(o1 != -1)[0]
            if len(valid_indices1) > 0:
                o1_last_valid_idx = np.max(valid_indices1)
                
                # 创建临时解
                if o1_last_valid_idx < chrom_length:
                    temp_o1 = np.zeros(chrom_length, dtype=nb.int64)
                    for i in range(chrom_length):
                        temp_o1[i] = o1[i] if i <= o1_last_valid_idx and o1[i] != -1 else 0
                    
                    # 内联验证解是否满足约束
                    # 1. 计算路径总距离
                    total_distance = 0.0
                    # 2. 检查是否有重复访问的节点
                    has_duplicate = False
                    visited = np.zeros(num_nodes, dtype=nb.boolean)
                    
                    # 寻找有效路径长度
                    valid_end_temp = chrom_length
                    for j in range(chrom_length-1, 0, -1):
                        if temp_o1[j] != 0:
                            valid_end_temp = j + 1
                            break
                    
                    # 确保路径以depot结束
                    if temp_o1[valid_end_temp-1] != 0:
                        if valid_end_temp < chrom_length:
                            temp_o1[valid_end_temp] = 0
                            valid_end_temp += 1
                        else:
                            # 如果已经达到染色体最大长度，则替换最后一个节点为depot
                            temp_o1[valid_end_temp-1] = 0
                    
                    # 重新计算完整的路径距离，确保准确
                    total_distance = 0.0
                    for j in range(1, valid_end_temp):
                        # 计算距离
                        total_distance += dist_matrix[temp_o1[j-1], temp_o1[j]]
                        
                        # 检查重复访问（排除depot）
                        if temp_o1[j] != 0:
                            if visited[temp_o1[j]]:
                                has_duplicate = True
                                break
                            visited[temp_o1[j]] = True
                    
                    # 确保路径总长度不超过安全距离，更保守一些
                    distance_ok = total_distance <= global_max_dist - 1e-5
                    
                    # 只有当解满足约束时才接受
                    if distance_ok and not has_duplicate:
                        offspring[p1_idx] = temp_o1
                    else:
                        # 如果不满足约束，使用原父代（假设父代是可行的）
                        offspring[p1_idx] = parent1
                else:
                    # 如果生成的解太长，使用原父代
                    offspring[p1_idx] = parent1
            else:
                # 如果没有有效索引，使用原父代
                offspring[p1_idx] = parent1
            
            # 处理第二个子代
            valid_indices2 = np.where(o2 != -1)[0]
            if len(valid_indices2) > 0:
                o2_last_valid_idx = np.max(valid_indices2)
                
                # 创建临时解
                if o2_last_valid_idx < chrom_length:
                    temp_o2 = np.zeros(chrom_length, dtype=nb.int64)
                    for i in range(chrom_length):
                        temp_o2[i] = o2[i] if i <= o2_last_valid_idx and o2[i] != -1 else 0
                    
                    # 内联验证解是否满足约束
                    # 1. 计算路径总距离
                    total_distance = 0.0
                    # 2. 检查是否有重复访问的节点
                    has_duplicate = False
                    visited = np.zeros(num_nodes, dtype=nb.boolean)
                    
                    # 寻找有效路径长度
                    valid_end_temp = chrom_length
                    for j in range(chrom_length-1, 0, -1):
                        if temp_o2[j] != 0:
                            valid_end_temp = j + 1
                            break
                    
                    # 确保路径以depot结束
                    if temp_o2[valid_end_temp-1] != 0:
                        if valid_end_temp < chrom_length:
                            temp_o2[valid_end_temp] = 0
                            valid_end_temp += 1
                        else:
                            # 如果已经达到染色体最大长度，则替换最后一个节点为depot
                            temp_o2[valid_end_temp-1] = 0
                    
                    # 重新计算完整的路径距离，确保准确
                    total_distance = 0.0
                    for j in range(1, valid_end_temp):
                        # 计算距离
                        total_distance += dist_matrix[temp_o2[j-1], temp_o2[j]]
                        
                        # 检查重复访问（排除depot）
                        if temp_o2[j] != 0:
                            if visited[temp_o2[j]]:
                                has_duplicate = True
                                break
                            visited[temp_o2[j]] = True
                    
                    # 确保路径总长度不超过安全距离，更保守一些
                    distance_ok = total_distance <= global_max_dist - 1e-5
                    
                    # 只有当解满足约束时才接受
                    if distance_ok and not has_duplicate:
                        offspring[p2_idx] = temp_o2
                    else:
                        # 如果不满足约束，使用原父代
                        offspring[p2_idx] = parent2
                else:
                    # 如果生成的解太长，使用原父代
                    offspring[p2_idx] = parent2
            else:
                # 如果没有有效索引，使用原父代
                offspring[p2_idx] = parent2
        else:
            # 如果不交叉，直接复制父代
            offspring[p1_idx] = parent1
            offspring[p2_idx] = parent2
                
    return offspring

@nb.njit(nb.int64[:,:](nb.int64[:,:], nb.float64, nb.float32[:], nb.float32[:,:], nb.float32[:]), parallel=True, nogil=True)
def node_replacement_mutate_op(pop, mutation_rate, prize, dist_matrix, max_distances):
    """
    针对OP问题的节点替换变异算法，尝试通过替换节点来提高总奖励值
    
    Args:
        pop: 种群矩阵，每行是一个染色体，终点应该是depot(0)
        mutation_rate: 变异率
        prize: 每个节点的奖励值数组[num_nodes]
        dist_matrix: 距离矩阵[num_nodes, num_nodes]
        max_distances: 全局最大距离约束[num_nodes]，其中max_distances[0]是原始约束
    
    Returns:
        变异后的种群
    """
    pop_size, chrom_length = pop.shape
    mutated_pop = pop.copy()
    num_nodes = len(prize)
    
    # 使用全局最大距离约束(max_distances[0])
    global_max_dist = max_distances[0]
    
    # 添加安全裕度，确保不会超出环境的距离约束
    safe_max_dist = global_max_dist - 1e-5  # 减去0.1作为安全裕度
    
    random_vals = np.random.random(pop_size)
    
    for i in range(pop_size):
        if random_vals[i] < mutation_rate:
            # 找到有效的路径长度（排除末尾可能的0）
            valid_end = chrom_length
            for j in range(chrom_length-1, 0, -1):
                if pop[i, j] != 0:
                    valid_end = j + 1
                    break
            
            # 如果路径太短无法变异，则跳过
            if valid_end <= 3:  # 起点、一个点、终点
                continue
                
            # 确保路径以depot结束
            if pop[i, valid_end-1] != 0:
                # 如果最后一个节点不是depot，添加depot作为终点
                if valid_end < chrom_length:
                    mutated_pop[i, valid_end] = 0
                    valid_end += 1
                else:
                    # 如果已经达到染色体最大长度，则替换最后一个节点为depot
                    mutated_pop[i, valid_end-1] = 0
                
            # 创建当前路径中节点集合
            current_nodes = np.zeros(num_nodes, dtype=nb.boolean)
            for j in range(valid_end - 1):  # 排除最后的depot
                if pop[i, j] != 0:  # 不是填充的0
                    current_nodes[pop[i, j]] = True
            
            # 计算当前路径的总距离和总奖励
            current_distance = 0.0
            current_prize = 0.0
            for j in range(1, valid_end):
                current_distance += dist_matrix[pop[i, j-1], pop[i, j]]
                if pop[i, j] != 0:  # 排除depot节点
                    current_prize += prize[pop[i, j]]
                    
            current_distance += dist_matrix[pop[i, valid_end-2], 0]  # depot到最后一个节点的距离
            current_distance += dist_matrix[0, pop[i, 0]]  # depot到第一个节点的距离
            
            # 随机选择要替换的节点（不包括起点和终点）
            replace_candidates = []
            for j in range(1, valid_end-1):  # 排除起点和终点
                if pop[i, j] != 0:  # 排除填充的0和depot
                    replace_candidates.append(j)
            
            if len(replace_candidates) == 0:
                continue
                
            # 随机选择要替换的位置
            replace_pos = replace_candidates[np.random.randint(0, len(replace_candidates))]
            node_to_replace = pop[i, replace_pos]
            
            # 获取替换位置的前后节点
            prev_node = pop[i, replace_pos-1]
            next_node_pos = replace_pos + 1
            while next_node_pos < valid_end and pop[i, next_node_pos] == 0 and next_node_pos < valid_end-1:
                next_node_pos += 1
            next_node = pop[i, next_node_pos]
                
            # 计算替换该节点后释放的距离
            freed_distance = dist_matrix[prev_node, node_to_replace] + dist_matrix[node_to_replace, next_node] - dist_matrix[prev_node, next_node]
            
            # 生成潜在的替换节点列表（当前未在路径中的节点）
            potential_replacements = []
            for j in range(1, num_nodes):  # 从1开始，排除depot
                # 跳过已在路径中的节点
                if not current_nodes[j] and j != 0:  # 排除depot(0)
                    potential_replacements.append(j)
            
            if len(potential_replacements) == 0:
                continue
                
            # 计算每个潜在替换节点的奖励/距离比
            best_node = -1
            best_ratio = -1.0
            
            for node in potential_replacements:
                # 计算插入该节点后增加的距离
                added_distance = dist_matrix[prev_node, node] + dist_matrix[node, next_node] - dist_matrix[prev_node, next_node]
                
                # 检查是否仍在预算内，使用安全裕度
                if current_distance - freed_distance + added_distance <= safe_max_dist:
                    # 比较奖励值
                    if prize[node] > prize[node_to_replace]:
                        ratio = prize[node] / (added_distance + 0.1)  # 避免除以0
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_node = node
            
            # 如果找到更好的节点，进行替换后内联验证约束
            if best_node != -1:
                # 创建临时解
                temp_solution = mutated_pop[i].copy()
                temp_solution[replace_pos] = best_node
                
                # 内联验证约束
                # 1. 计算路径总距离
                total_distance = 0.0
                # 2. 检查是否有重复访问的节点
                has_duplicate = False
                visited = np.zeros(num_nodes, dtype=nb.boolean)
                
                # 寻找有效路径长度
                valid_end_temp = chrom_length
                for j in range(chrom_length-1, 0, -1):
                    if temp_solution[j] != 0:
                        valid_end_temp = j + 1
                        break
                
                # 确保路径以depot结束
                if valid_end_temp < chrom_length and temp_solution[valid_end_temp-1] != 0:
                    temp_solution[valid_end_temp] = 0
                    valid_end_temp += 1
                
                # 重新计算完整路径距离，确保准确性
                total_distance = 0.0
                for j in range(1, valid_end_temp):
                    # 计算距离
                    total_distance += dist_matrix[temp_solution[j-1], temp_solution[j]]
                    
                    # 检查重复访问（排除depot）
                    if temp_solution[j] != 0:
                        if visited[temp_solution[j]]:
                            has_duplicate = True
                            break
                        visited[temp_solution[j]] = True
                
                # 确保路径总长度不超过安全距离
                distance_ok = total_distance <= global_max_dist - 1e-5
                
                # 只有当解满足约束时才应用变异
                if distance_ok and not has_duplicate:
                    mutated_pop[i, replace_pos] = best_node
            
    return mutated_pop

@nb.njit(nb.int64[:,:](nb.int64[:,:], nb.float64, nb.float32[:], nb.float32[:,:], nb.float32[:]), parallel=True, nogil=True)
def inverse_mutate_op(pop, mutation_rate, prize, dist_matrix, max_distances):
    """
    针对OP问题的路径逆转变异算法，在保持路径长度约束的前提下逆转部分路径
    
    Args:
        pop: 种群矩阵，每行是一个染色体，第一个节点是第一个访问的客户节点（不是仓库）
        mutation_rate: 变异率
        prize: 每个节点的奖励值数组[num_nodes]
        dist_matrix: 距离矩阵[num_nodes, num_nodes]
        max_distances: 全局最大距离约束[num_nodes]，其中max_distances[0]是原始约束
    
    Returns:
        变异后的种群
    """
    pop_size, chrom_length = pop.shape
    mutated_pop = pop.copy()
    num_nodes = len(prize)
    
    # 使用全局最大距离约束(max_distances[0])
    global_max_dist = max_distances[0]
    
    # 添加安全裕度，确保不会超出环境的距离约束
    safe_max_dist = global_max_dist - 1e-5  # 减去微小值作为安全裕度
    
    random_vals = np.random.random(pop_size)
    
    for i in range(pop_size):
        if random_vals[i] < mutation_rate:
            # 找到有效的路径长度（排除末尾可能的0）
            valid_end = chrom_length
            for j in range(chrom_length-1, 0, -1):
                if pop[i, j] != 0:
                    valid_end = j + 1
                    break
            
            # 如果路径太短无法变异，则跳过
            if valid_end <= 3:  # 起点、一个点、终点
                continue
                
            # 确保路径以depot结束
            if pop[i, valid_end-1] != 0:
                # 如果最后一个节点不是depot，添加depot作为终点
                if valid_end < chrom_length:
                    mutated_pop[i, valid_end] = 0
                    valid_end += 1
                else:
                    # 如果已经达到染色体最大长度，则替换最后一个节点为depot
                    mutated_pop[i, valid_end-1] = 0
            
            # 计算当前路径的总距离
            current_distance = dist_matrix[0, mutated_pop[i, 0]]  # 从仓库到第一个节点的距离
            
            # 添加路径中所有相邻节点之间的距离
            for j in range(1, valid_end):
                current_distance += dist_matrix[mutated_pop[i, j-1], mutated_pop[i, j]]
            
            # 尝试多次找到合适的路径段进行逆转
            max_attempts = 1
            success = False
            
            for attempt in range(max_attempts):
                # 随机选择一段路径进行逆转（不包括起点和终点）
                start_idx = np.random.randint(1, valid_end-2) if valid_end > 3 else 1
                end_idx = np.random.randint(start_idx+1, valid_end-1)
                
                if start_idx >= end_idx:
                    continue
                
                # 计算逆转前的子路径距离
                old_sub_distance = 0.0
                for j in range(start_idx, end_idx):
                    old_sub_distance += dist_matrix[mutated_pop[i, j], mutated_pop[i, j+1]]
                
                # 考虑与前后节点的连接距离
                old_connection_distance = 0.0
                if start_idx > 0:
                    old_connection_distance += dist_matrix[mutated_pop[i, start_idx-1], mutated_pop[i, start_idx]]
                if end_idx < valid_end-1:
                    old_connection_distance += dist_matrix[mutated_pop[i, end_idx], mutated_pop[i, end_idx+1]]
                
                # 创建临时解进行路径逆转
                temp_solution = mutated_pop[i].copy()
                
                # 逆转选定的路径段
                segment = np.zeros(end_idx-start_idx+1, dtype=nb.int64)
                for j in range(end_idx-start_idx+1):
                    segment[j] = temp_solution[start_idx+j]
                
                for j in range(end_idx-start_idx+1):
                    temp_solution[start_idx+j] = segment[end_idx-start_idx-j]
                
                # 计算逆转后的子路径距离
                new_sub_distance = 0.0
                for j in range(start_idx, end_idx):
                    new_sub_distance += dist_matrix[temp_solution[j], temp_solution[j+1]]
                
                # 计算新的连接距离
                new_connection_distance = 0.0
                if start_idx > 0:
                    new_connection_distance += dist_matrix[temp_solution[start_idx-1], temp_solution[start_idx]]
                if end_idx < valid_end-1:
                    new_connection_distance += dist_matrix[temp_solution[end_idx], temp_solution[end_idx+1]]
                
                # 计算距离变化
                distance_change = (new_sub_distance + new_connection_distance) - (old_sub_distance + old_connection_distance)
                new_total_distance = current_distance + distance_change
                
                # 检查变异后的解是否仍然满足约束
                if new_total_distance <= safe_max_dist:
                    # 内联验证约束
                    # 1. 检查是否有重复访问的节点
                    has_duplicate = False
                    visited = np.zeros(num_nodes, dtype=nb.boolean)
                    
                    # 检查每个节点是否重复
                    for j in range(valid_end):
                        if temp_solution[j] != 0:  # 排除depot节点
                            if visited[temp_solution[j]]:
                                has_duplicate = True
                                break
                            visited[temp_solution[j]] = True
                    
                    # 2. 重新计算总距离，确保准确性
                    total_distance = dist_matrix[0, temp_solution[0]]  # 从仓库到第一个节点的距离
                    for j in range(1, valid_end):
                        total_distance += dist_matrix[temp_solution[j-1], temp_solution[j]]
                    
                    # 3. 再次确认距离约束
                    distance_ok = total_distance <= safe_max_dist
                    
                    # 如果没有违反任何约束，接受变异
                    if distance_ok and not has_duplicate:
                        # 应用变异
                        for j in range(start_idx, end_idx+1):
                            mutated_pop[i, j] = temp_solution[j]
                        success = True
                        break  # 找到合适的变异，跳出尝试循环
            
            # 如果没有找到合适的变异，保持原始路径
            if not success:
                for j in range(chrom_length):
                    mutated_pop[i, j] = pop[i, j]
            
    return mutated_pop

@nb.njit(nb.int64[:,:](nb.int64[:,:], nb.float32), parallel=True, nogil=True)
def multi_point_crossover_ffsp(parents, crossover_rate):
    """
    多点交叉算法
    
    Args:
        parents: 父代种群
        crossover_rate: 交叉率
    
    Returns:
        后代种群
    """
    pop_size, chrom_length = parents.shape
    offspring = np.zeros((pop_size, chrom_length), dtype=nb.int64)
    
    for i in prange(pop_size // 2):
        p1_idx = i * 2
        p2_idx = i * 2 + 1
        
        parent1 = parents[p1_idx]
        parent2 = parents[p2_idx]
        
        if np.random.random() < crossover_rate:
            # 随机选择多个交叉点
            crossover_points = np.sort(np.random.choice(chrom_length, 2, replace=False))
            start, end = crossover_points
            offspring[p1_idx, start:end] = parent2[start:end]
            offspring[p2_idx, start:end] = parent1[start:end]
            
            # 交换其余部分
            offspring[p1_idx, :start] = parent1[:start]
            offspring[p1_idx, end:] = parent1[end:]
            offspring[p2_idx, :start] = parent2[:start]
            offspring[p2_idx, end:] = parent2[end:]
        else:
            offspring[p1_idx] = parent1
            offspring[p2_idx] = parent2
            
    return offspring

@nb.njit(nb.int64[:,:](nb.int64[:,:], nb.float64), parallel=True, nogil=True)
def swap_mutate_ffsp(pop, mutation_rate):
    """
    交换变异算法
    
    Args:
        pop: 种群矩阵
        mutation_rate: 变异率
    
    Returns:
        变异后的种群
    """
    pop_size, chrom_length = pop.shape
    mutated_pop = pop.copy()
    
    random_vals = np.random.random(pop_size)
    
    for i in prange(pop_size):
        if random_vals[i] < mutation_rate:
            # 随机选择两个基因进行交换
            idx1 = np.random.randint(0, chrom_length)
            idx2 = np.random.randint(0, chrom_length)
            
            mutated_pop[i, idx1], mutated_pop[i, idx2] = mutated_pop[i, idx2], mutated_pop[i, idx1]
    
    return mutated_pop

@nb.njit(nb.int64[:,:](nb.int64[:], nb.int64, nb.int64, nb.float64, nb.bool), parallel=True, nogil=True)
def generate_population(route, pop_size, env_code, mutate_rate, verbose=False):
    """Generate population from a single route"""
    n = len(route) - 1
    pop = np.zeros((pop_size, n+1), dtype=nb.int64)
    
    if env_code == 1:
        # env_name == "tsp"
        for j in prange(n+1):
            pop[0, j] = route[j]
        
        for i in prange(1, pop_size):
            start_idx = i % (n+1)
            if start_idx == 0:
                start_idx = 1
            
            pop[i, 0] = route[start_idx]
            
            pos = 1
            for j in range(1, n+1):
                orig_idx = (start_idx + j) % (n+1)
                pop[i, pos] = route[orig_idx]
                pos += 1
    else:
        for i in prange(pop_size):
            pop[i] = route
            
    return pop

@nb.njit(nb.int64[:,:,:](nb.int64[:,:,:], nb.int64, nb.float64), parallel=True, nogil=True)
def generate_batch_population(batch_routes, env_code, mutate_rate):
    """
    Generate population from a batch of single route
    
    Args:
        route: single route, shape: [batch_size, 1, seq_len]
        
    Returns:
        population, shape: [batch_size, pop_size(= seq_len), seq_len]
    """
    batch_size, _, seq_len = batch_routes.shape
    pop_size = nb.int64(50)
    # pop_size = seq_len
    population = np.zeros((batch_size, pop_size, seq_len), dtype=nb.int64)
    
    for b in prange(batch_size):
        route = batch_routes[b, 0, :]
        verbose = (b == 0)
        pop = generate_population(route, pop_size, env_code, 0.0, verbose)
        population[b] = pop
    
    return population