import sys
import numpy as np
import torch
from dqn import QNet, dev
from chompenv import ChompEnv

def evaluate(name, checkpoint, build, episodes=20):
    env = ChompEnv(build, time_scale=1, no_graphics=True)
    q = QNet().to(dev)
    q.load_state_dict(torch.load(checkpoint, map_location=dev, weights_only=True))
    q.eval()
    rewards = []

    for ep in range(episodes):
        obs, total, done = env.reset(), 0, False
        while not done:
            with torch.no_grad():
                x = torch.from_numpy(obs.copy()).unsqueeze(0).to(dev)
                action = q(x).argmax(1).item()
            obs, reward, done, truncated = env.step(action)
            total += reward
            if truncated:
                break
        rewards.append(total)
        print(f"{name}: episode {ep+1}/{episodes}, reward={total:.1f}")

    env.close()
    print(f"{name} average reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

if __name__ == "__main__":
    build = sys.argv[1] if len(sys.argv) > 1 else "../ChompBot.app"
    evaluate("DQN", "dqn_150000.pt", build)
    evaluate("DDQN", "ddqn_150000.pt", build)
