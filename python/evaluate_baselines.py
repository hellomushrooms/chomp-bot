import sys
from chompenv import ChompEnv
from agents import random_agent, rule_based_agent


def evaluate(policy, run_id, n=20, build=None):
    env = ChompEnv(build, run_id=run_id, time_scale=20, no_graphics=build is not None)
    for ep in range(n):
        env.set_param("seed", 1000 + ep)        # same seeds for every agent
        obs, done = env.reset(), False
        while not done:
            obs, r, done, trunc = env.step(policy(obs))
        print(f"run {run_id}: episode {ep + 1}/{n} finished")
    env.close()


if __name__ == "__main__":
    which = sys.argv[1]                          # "random" or "rule"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    build = sys.argv[3] if len(sys.argv) > 3 else None
    if which == "random":
        evaluate(random_agent, run_id=1, n=n, build=build)
    else:
        evaluate(rule_based_agent, run_id=2, n=n, build=build)