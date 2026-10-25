import random
from chompenv import ChompEnv

env = ChompEnv(build=None, time_scale=1.0)   # waits for you to press Play in Unity
obs = env.reset()

print("obs shape:", obs.shape)              # expect (6, 31, 28)
print("walls:", int(obs[0].sum()))          # expect several hundred
print("pellets:", int(obs[1].sum()), "| power pellets:", int(obs[2].sum()))
print("pacman cells:", int(obs[3].sum()), "| ghost cells:", int(obs[4].sum()))

total = 0.0
for i in range(400):
    obs, r, done, trunc = env.step(random.randrange(5))
    total += r
    if done:
        print(f"episode ended at step {i}, total reward so far {total:.2f}, truncated={trunc}")
        obs = env.reset()

env.close()
print("done")