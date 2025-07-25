from rl4co.envs.routing import TSPEnv, TSPGenerator
from rl4co.models import AttentionModelPolicy, POMO, GLOP, EAM, GLOPPolicy
from rl4co.utils import RL4COTrainer

# Instantiate generator and environment
generator = TSPGenerator(num_loc=5000, loc_distribution="uniform")
env = TSPEnv(generator)

config = {
        'num_generations': 3,
        'mutation_rate': 0.1,
        'crossover_rate': 0.6,
        'selection_rate': 0.2,
        'batch_size': 64,
        'ea_batch_size': 64,
        'ea_prob': 0.01,
        'ea_epoch': 700,
    }
metric = {
        'train': ['reward', 'loss'],
        'val': ['reward', 'max_reward', 'max_aug_reward'],
        'test': ['reward', 'max_reward', 'max_aug_reward'],
    }
# eam = EAM(env, 
#         policy=AttentionModelPolicy(
#                 env_name=env.name, 
#                 embed_dim=128,
#                 num_encoder_layers=3,
#                 num_heads=8,
#                 normalization="instance",
#                 use_graph_context=False,
#             ), 
#         batch_size=config['batch_size'],
#         optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
#         lr_scheduler = "MultiStepLR",
#         train_data_size = 160_000, # 160_000,
#         test_data_size = 10_000, # 10_000,
#         val_data_size = 10_000, # 10_000,
#         ea_kwargs = config,
#         baseline = 'rollout', # am use rollout baseline
#         metrics = metric,
#         )
eam = EAM.load_from_checkpoint("eam_pomo_tsp100_epoch=1500.ckpt")

policy = GLOPPolicy(
    env_name=env.name,
    subprob_solver = eam,  # Use EAM as subproblem solver
)
model = GLOP(
    env=env,
    policy=policy,
    baseline="mean",
    baseline_kwargs={"reward_scale": "log"},
)

# Instantiate Trainer and fit
trainer = RL4COTrainer(max_epochs=10, accelerator="gpu", precision="32", devices = [0])
trainer.fit(model)