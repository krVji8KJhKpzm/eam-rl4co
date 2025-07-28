
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
"""
eam_pomo_cvrp100 - cuda:0
eam_pomo_cvrp50 - cuda:1
eam_am_cvrp100 - cuda:2
eam_am_cvrp50 - cuda:3
eam_symnco_cvrp100 - cuda:4
eam_symnco_cvrp50 - cuda:5
"""
def main(opt):
    problem_size = opt.problem_size
    device = [opt.cuda]
    # ...existing code...
    if opt.epoch is not None:
        if problem_size == 50:
            max_epoch = 100
        elif problem_size == 100:
            max_epoch = 200
    else:
        max_epoch = opt.epoch  # 支持自定义epoch

    # model = 'eam'
    model_name = 'eam_pomo'
    debug = False
    # device = [0,1,2,3,4,5,6]
    num_augment = 0
    env_name = 'kp'
    config = {
        'num_generations': 3,
        'mutation_rate': 0.1,
        'crossover_rate': 0.6,
        'selection_rate': 0.2,
        'batch_size': 64,
        'ea_batch_size': 64,
        'alpha': 0.5,
        'beta': 3,
        'ea_prob': 0.01,
        'ea_epoch': 700,
    }
    
    metric = {
        'train': ['reward', 'loss'],
        'val': ['reward', 'max_reward', 'max_aug_reward'],
        'test': ['reward', 'max_reward', 'max_aug_reward'],
    }
    
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
    policy = AttentionModelPolicy(
                env_name=env.name, 
                embed_dim=128,
                num_encoder_layers=6,
                num_heads=8,
                normalization="instance",
                use_graph_context=False,
                )
    if opt.model == "pomo":
        model = POMO(env, policy, 
                    batch_size=config['batch_size'], 
                    optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
                    lr_scheduler = "MultiStepLR", 
                    train_data_size = 100_000,
                    test_data_size = 10_000,
                    val_data_size = 10_000,
                    num_augment = num_augment,
                    lr_scheduler_kwargs = {"milestones": [160, 190], "gamma": 0.1},
                    baseline = "shared", # pomo use shared baseline
                    metrics = metric,)
    elif opt.model == "am":
        model = AttentionModel(
                env, 
                policy, 
                batch_size=config['batch_size'], 
                optimizer_kwargs={"lr": 1e-4,},
                lr_scheduler = "MultiStepLR", 
                train_data_size = 100_000,
                test_data_size = 10_000,
                val_data_size = 10_000,
                lr_scheduler_kwargs = {"milestones": [160, 190], "gamma": 0.1},
                metrics = metric,
            )
    if debug:
        logger = None
        trainer = RL4COTrainer(max_epochs=10, 
                                accelerator="gpu",
                                precision=32,
                                devices = device,
                                logger=logger,
                                enable_checkpointing=False)
    else:
        # checkpoint_callback = ModelCheckpoint(dirpath="eam_tsp50_checkpoints", # save to checkpoints/
        #                                 filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
        #                                 save_top_k=1, # save only the best model
        #                                 every_n_epochs=1, # save every epoch
        #                                 save_last=True, # save the last model
        #                                 monitor="val/reward", # monitor validation reward
        #                                 mode="max") # maximize validation reward

        # Print model summary
        # rich_model_summary = RichModelSummary(max_depth=3)

        # Callbacks list
        # callbacks = [checkpoint_callback, rich_model_summary]
        logger = WandbLogger(project="RL4CO_debug=" + str(debug), name=f"{model_name}_{env_name}{problem_size}")
        
        trainer = RL4COTrainer(max_epochs=max_epoch, 
                                accelerator="gpu",
                                precision=32,
                                devices= device,
                                logger=logger,)
    trainer.fit(model)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=None, help='训练轮数')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA序号')
    parser.add_argument('--problem_size', type=int, default=50, help='问题规模')
    parser.add_argument('--model', type=str, default='am', help='模型名称')
    opt = parser.parse_args()
    main(opt)