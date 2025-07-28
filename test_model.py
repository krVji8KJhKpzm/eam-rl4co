
# from rl4co.models import SharedBufferManager, evolution_worker, ThreadBuffer, evolution_thread_worker
from rl4co.utils import RL4COTrainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from rl4co.models import EAM, AttentionModelPolicy, SymNCOPolicy, SymNCO, SymEAM, POMO, AttentionModel
import argparse
import multiprocessing as mp

import threading
import os
os.environ["WANDB_MODE"] = "offline"

def main():
    problem_size = 100
    device = [4]
    # model = 'eam'
    model_name = 'eam_pomo'
    debug = False
    # device = [0,1,2,3,4,5,6]
    num_augment = 0
    env_name = 'kp'
    
    from rl4co.envs.routing import TSPEnv, TSPGenerator, CVRPEnv, CVRPGenerator, KnapsackEnv, KnapsackGenerator
    if env_name == 'tsp':
        generator = TSPGenerator(num_loc=problem_size, loc_distribution="uniform")
        env = TSPEnv(generator)
    elif env_name == 'cvrp':
        generator = CVRPGenerator(num_loc=problem_size, loc_distribution="uniform", num_depots=1)
        env = CVRPEnv(generator)
    elif env_name == 'kp':
        generator = KnapsackGenerator(num_items=problem_size, 
                                      weight_distribution="uniform", 
                                      value_distribution="uniform")
        env = KnapsackEnv(generator)
    
    model = POMO.load_from_checkpoint("eam_pomo_4.ckpt", env=env)

    trainer = RL4COTrainer(max_epochs=1, 
                            accelerator="gpu",
                            precision=32,
                            devices= device)
    trainer.test(model)
        
if __name__ == "__main__":
    main()