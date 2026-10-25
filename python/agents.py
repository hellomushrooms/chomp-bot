import random
from collections import deque
import numpy as np

# action -> (dx, dy). Up = y+1 because the grid origin is the lower-left corner.
DIRS = [(0, 1), (0, -1), (-1, 0), (1, 0)]   # 0 Up, 1 Down, 2 Left, 3 Right; 4 = Stay


def random_agent(obs):
    return random.randrange(5)


def rule_based_agent(obs, danger_radius=3):
    walls = obs[0] > 0
    food = (obs[1] + obs[2]) > 0
    pac = np.argwhere(obs[3] > 0)
    if len(pac) == 0:
        return 4
    py, px = pac[0]
    ghosts = np.argwhere(obs[4] > 0)            # only non-frightened (dangerous) ghosts
    H, W = walls.shape

    danger = np.zeros_like(walls)
    for gy, gx in ghosts:
        for y in range(max(0, gy - danger_radius), min(H, gy + danger_radius + 1)):
            for x in range(max(0, gx - danger_radius), min(W, gx + danger_radius + 1)):
                if abs(y - gy) + abs(x - gx) <= danger_radius:
                    danger[y, x] = True

    # BFS to the nearest pellet, avoiding walls and cells near ghosts
    seen = np.zeros_like(walls)
    seen[py, px] = True
    q = deque()
    for a, (dx, dy) in enumerate(DIRS):
        x, y = px + dx, py + dy
        if 0 <= x < W and 0 <= y < H and not walls[y, x] and not danger[y, x]:
            seen[y, x] = True
            q.append((x, y, a))                  # remember the FIRST action taken
    while q:
        x, y, a = q.popleft()
        if food[y, x]:
            return a
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not seen[ny, nx] \
               and not walls[ny, nx] and not danger[ny, nx]:
                seen[ny, nx] = True
                q.append((nx, ny, a))

    # No safe path to a pellet: step to the open neighbour farthest from the nearest ghost
    best, best_d = 4, -1
    for a, (dx, dy) in enumerate(DIRS):
        x, y = px + dx, py + dy
        if 0 <= x < W and 0 <= y < H and not walls[y, x]:
            d = min((abs(y - gy) + abs(x - gx) for gy, gx in ghosts), default=99)
            if d > best_d:
                best, best_d = a, d
    return best