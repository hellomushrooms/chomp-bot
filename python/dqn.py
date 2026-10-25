import random, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from chompenv import ChompEnv

dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class QNet(nn.Module):
    def __init__(self, n_actions=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU())   # 31x28 -> 16x14
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16 * 14, 256), nn.ReLU(),
                                nn.Linear(256, n_actions))

    def forward(self, x):
        return self.fc(self.conv(x.float()))


class Replay:
    def __init__(self, cap):
        self.cap, self.i, self.n = cap, 0, 0
        self.s = np.zeros((cap, 6, 31, 28), np.uint8)
        self.s2 = np.zeros_like(self.s)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)

    def add(self, s, a, r, s2, d):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        idx = np.random.randint(0, self.n, b)
        t = lambda x: torch.from_numpy(x[idx]).to(dev)
        return t(self.s), t(self.a), t(self.r), t(self.s2), t(self.d)


def train(name, double, total_steps=300_000, build=None, run_id=10, seed=0, **reward_params):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = ChompEnv(build, run_id=run_id, time_scale=20,
                   no_graphics=build is not None, **reward_params)
    q, qt = QNet().to(dev), QNet().to(dev)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-4)
    buf = Replay(50_000)                      # about 0.5 GB of RAM
    writer = SummaryWriter(f"runs/{name}")

    obs, ep_r, episodes = env.reset(), 0.0, 0
    for t in range(1, total_steps + 1):
        eps = max(0.05, 1.0 - t / 100_000)    # linear decay over the first 100k steps
        if random.random() < eps:
            a = random.randrange(5)
        else:
            with torch.no_grad():
                a = int(q(torch.from_numpy(obs).unsqueeze(0).to(dev)).argmax(1))

        obs2, r, done, trunc = env.step(a)
        buf.add(obs, a, r, obs2, float(done and not trunc))   # timeouts still bootstrap
        obs, ep_r = obs2, ep_r + r
        if done:
            episodes += 1
            writer.add_scalar("episode/reward", ep_r, t)
            print(f"[{name}] step {t}  episode {episodes}  reward {ep_r:.1f}  eps {eps:.2f}")
            obs, ep_r = env.reset(), 0.0

        if t > 5_000 and t % 4 == 0:
            s, a_, r_, s2, d = buf.sample(64)
            with torch.no_grad():
                if double:   # Double DQN: online net picks the action, target net scores it
                    nxt = qt(s2).gather(1, q(s2).argmax(1, keepdim=True)).squeeze(1)
                else:        # plain DQN
                    nxt = qt(s2).max(1)[0]
                y = r_ + 0.99 * (1 - d) * nxt
            pred = q(s).gather(1, a_.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(pred, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10)
            opt.step()
            if t % 1_000 == 0:
                writer.add_scalar("train/loss", loss.item(), t)

        if t % 2_000 == 0:
            qt.load_state_dict(q.state_dict())
        if t % 50_000 == 0:
            torch.save(q.state_dict(), f"{name}_{t}.pt")

    torch.save(q.state_dict(), f"{name}.pt")
    env.close()


if __name__ == "__main__":
    algo = sys.argv[1]                       # "dqn" or "ddqn"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
    build = sys.argv[3] if len(sys.argv) > 3 else None
    train(algo, double=(algo == "ddqn"), total_steps=steps, build=build,
          run_id=10 if algo == "dqn" else 11)