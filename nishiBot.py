import sys
from stable_baselines3 import PPO 
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from game2048 import Game2048Env
from matplotlib import pyplot as plt
from stable_baselines3.common.env_util import make_vec_env


env = Game2048Env()#render_mode="human")
#env = make_vec_env(Game2048Env, n_envs=4)
model = PPO("MlpPolicy", env, verbose=1, batch_size=64, 
            learning_rate=lambda p: 0.0001 + 0.0004*p)
#model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000, log_interval=4)

model.save("PPO_.5M_R=0.9")
env.plot_learning_curve()

avg = 0
times = 0
obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, _, info = env.step(action)

    if done:
        print("SCORE REACHED", info)
        avg += info['score'] 
        times += 1
        if times == 100:
            break
        obs, info = env.reset()

print("AVRG SCORE", avg/times)