from rl4co.envs.routing import TSPEnv, TSPGenerator, CVRPEnv, CVRPGenerator, KnapsackEnv, KnapsackGenerator

problem_size = 100  # Example problem size, can be adjusted

generator = KnapsackGenerator(num_items=problem_size, 
                                      weight_distribution="uniform", 
                                      value_distribution="uniform")
env = KnapsackEnv(generator)

td = env.reset(batch_size = 10000)

mean_values_optimal = env.get_optimal_solutions(td)
print(f"Mean optimal solution value: {mean_values_optimal}")

mean_values_greedy = env.get_greedy_solutions(td)
print(f"Mean greedy solution value: {mean_values_greedy}")