# import supersuit
import wandb
import torch as th
from wandb.integration.sb3 import WandbCallback
import os
from stable_baselines3 import PPO, A2C, TD3, SAC, DQN, HerReplayBuffer, HER, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import TQC, MaskablePPO, TRPO, ARS, RecurrentPPO, QRDQN
# pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
from uav_env import parallel_env
from uav_centralized_env import DronesEnv as single_agent_env
from result_buffer import ResultBuffer
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from input_config import InputConfig

os.environ["WANDB_SILENT"] = "True"

project_name = "multi-agent-uav-offloading-FITCE"

print("starting {} project".format(project_name))

eval_episodes = 20  # 250

p = 1.75
o = 1

m_algo = [DQN, QRDQN, A2C, PPO, TRPO, ARS]  # multi agent
s_algo = [A2C, PPO, TRPO]
n_uav = [4, 6, 8]
for n_uavs in n_uav:
    for single_agents in [True, False]:
        if single_agents:
            algo = s_algo
        else:
            algo = m_algo
        for alg in algo:
            res_buffer = ResultBuffer(min_n_drone=n_uavs, max_n_drone=n_uavs, min_mu=p, max_mu=p, step_mu=0.1,
                                      net_slice=1, change_processing=True, alg=alg.__name__)

            config = {"algo": alg.__name__,
                      "n_cpus": 1,
                      "uavs": n_uavs,
                      "frame_stack": 4,
                      "processing_rate": p,
                      "offloading_rate": o,  # 2.5
                      "transition_probability_low": 1 / 180,
                      "transition_probability_high": 1 / 60,
                      "shifting_probs": True,  # True gives better learning curve, while False gives nice results quicker
                      "lambda_low": 1.3,
                      "lambda_high": 2.3,
                      "policy_type": "MlpPolicy",
                      "total_timesteps": 500000,  # 500000  # 1000000
                      "single_agent": single_agents,
                      "training": True,
                      }

            input_config = InputConfig(uavs=config["uavs"],
                                       frame_stack=config["frame_stack"],
                                       processing_rate=config["processing_rate"],
                                       offloading_rate=config["offloading_rate"],
                                       lmbda=[config["lambda_low"], config["lambda_high"]],
                                       prob_trans=[config["transition_probability_low"],
                                                   config["transition_probability_high"]],
                                       shifting_probs=config["shifting_probs"],
                                       algorithm=alg,
                                       )
            input_config.print_settings()

            if single_agents:
                uav_env = single_agent_env(input_c=input_config)
                uav_env = concat_vec_envs_v1(uav_env, 1, num_cpus=config["n_cpus"],
                                             base_class='stable_baselines3')
                res_buffer.set_save_runs(n_drones=config["uavs"], mu=config["processing_rate"])

                eval_env = single_agent_env(input_c=input_config, result_buffer=res_buffer)
                eval_env = concat_vec_envs_v1(eval_env, 1, num_cpus=config["n_cpus"],
                                              base_class='stable_baselines3')
            else:
                uav_env = parallel_env(input_c=input_config)
                uav_env = pettingzoo_env_to_vec_env_v1(uav_env)

                uav_env = concat_vec_envs_v1(uav_env, 1, num_cpus=config["n_cpus"],
                                             base_class='stable_baselines3')

                res_buffer.set_save_runs(n_drones=config["uavs"], mu=config["processing_rate"])

                eval_env = parallel_env(input_c=input_config, result_buffer=res_buffer)
                eval_env = pettingzoo_env_to_vec_env_v1(eval_env)
                eval_env = concat_vec_envs_v1(eval_env, 1, num_cpus=config["n_cpus"],
                                              base_class='stable_baselines3')

                # uav_env = supersuit.normalize_obs_v0(uav_env, env_min=0, env_max=1)
                # uav_env = supersuit.frame_stack_v1(uav_env, 3)

            # need to initialize it even for static policies to use evaluate_policy from sb3
            if alg != ARS:
                model = alg(config["policy_type"], uav_env, verbose=0, gamma=0.95, tensorboard_log=f"runs")
            else:
                model = alg(config["policy_type"], uav_env, verbose=0, tensorboard_log=f"runs")

            for i in range(3):
                config["training"] = True
                run = wandb.init(
                    project=project_name,
                    tags=["n {}".format(config["uavs"]),
                          "pr {}".format(config["processing_rate"]),
                          "or {}".format(config["offloading_rate"]),
                          "lmbda_l {:.2f}, lmbda_h {:.2f}".format(config["lambda_low"], config["lambda_high"]),
                          "prob_l {:.2f}, prob_h {:.2f}".format(config["transition_probability_low"],
                                                                config["transition_probability_high"]),
                          "alg {}".format(alg.__name__),
                          "single-agent:{}".format(single_agents)
                          ],
                    entity="xraulz",
                    reinit=True,
                    sync_tensorboard=True,
                    config=config,
                )

                wandb.run.name = wandb.run.name + "-{}-tr".format(alg.__name__)
                wandb.run.save()

                if os.name == 'nt':
                    model.learn(
                        total_timesteps=config["total_timesteps"],
                    )
                else:
                    model.learn(
                        total_timesteps=config["total_timesteps"],
                        callback=WandbCallback(
                            gradient_save_freq=100,
                            model_save_path=f"models/{project_name}/{run.name}",
                            verbose=2,
                        ),
                    )
                # model is already saved via model_save_path inside model.learn
                # model.save("policy {}-{}".format(run.name, i))
                run.finish()

                print("initializing evaluation...")
                config["training"] = False
                run = wandb.init(
                    project=project_name,
                    tags=["n {}".format(config["uavs"]),
                          "pr {}".format(config["processing_rate"]),
                          "or {}".format(config["offloading_rate"]),
                          "lmbda_l {:.2f}, lmbda_h {:.2f}".format(config["lambda_low"], config["lambda_high"]),
                          "prob_l {:.2f}, prob_h {:.2f}".format(config["transition_probability_low"],
                                                                config["transition_probability_high"]),
                          "evaluation",
                          "alg {}".format(alg.__name__),
                          ],
                    entity="xraulz",
                    reinit=True,
                    sync_tensorboard=True,
                    config=config,
                )

                wandb.run.name = wandb.run.name + "-{}-eval".format(alg.__name__)
                wandb.run.save()
                ev_episodes = eval_episodes if single_agents else eval_episodes * n_uavs
                mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=ev_episodes, deterministic=True)
                run.finish()
