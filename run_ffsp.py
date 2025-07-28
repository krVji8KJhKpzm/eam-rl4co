
# from rl4co.models import SharedBufferManager, evolution_worker, ThreadBuffer, evolution_thread_worker
import os
os.environ["WANDB_MODE"] = "offline"
from rl4co.utils import RL4COTrainer
from lightning.pytorch.loggers import WandbLogger
from rl4co.models import EAM, AttentionModelPolicy, POMO, MatNet, MatNetPolicy

import multiprocessing as mp

import threading

def main():
    
    os.environ["WANDB_MODE"] = "offline"
    
    problem_size = 100
    debug = True
    # device = [0,1,2,3,4,5,6]
    device = [4]
    num_augment = 128
    
    from rl4co.envs.routing import TSPEnv, TSPGenerator, CVRPEnv, CVRPGenerator, PCTSPEnv, PCTSPGenerator
    from rl4co.envs.scheduling import FFSPEnv, FFSPGenerator
    generator = FFSPGenerator(num_stage = 3, 
                              num_machine = 4,
                              num_job=problem_size)
    env = FFSPEnv(generator)
    # td = env.reset(batch_size=1)
    # print(td)
    # exit()
    policy = MatNetPolicy(
                env_name=env.name, 
                embed_dim=256,
                num_encoder_layers=3,
                num_heads=16,
                normalization="instance",
                use_graph_context=False,
                )
    model = MatNet(env, policy,
                batch_size= 50,
                optimizer_kwargs={"lr": 1e-4},
                lr_scheduler = "MultiStepLR", 
                train_data_size = 1_000,
                test_data_size = 1_000,
                val_data_size = 1_000,
                num_augment = num_augment,
                ) # alpha controls the weight of the reward
    # model = EAM.load_from_checkpoint("epoch=69-step=175000.ckpt", env=env, policy=policy)
    logger = WandbLogger(project="RL4CO_debug=" + str(debug), name=f"eam_d{len(device)}_aug{num_augment}_step_{problem_size}",
                         log_model="all",
                         offline=False,)

    if debug:
        logger = None
        trainer = RL4COTrainer(max_epochs=10, 
                                accelerator="cpu",
                                precision=32,
                                # devices = device,
                                logger=logger,
                                enable_checkpointing=False)
    else:
        trainer = RL4COTrainer(max_epochs=200, 
                                accelerator="gpu",
                                precision=32,
                                devices= device,
                                # devices=[0,1,2,3],
                                logger=logger,)
    trainer.fit(model)
        
if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    main()