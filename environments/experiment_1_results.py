from experiment_1_env import Env
import numpy as np
from multiprocessing import Pool
import time

from lib import dynamics as dyn

BOUND = 1e5
SECTOR = np.pi/32


def one_run(y):
    env = Env()
    state = env.reset()
    done = False
    is_halfway = False
    capture_flag = 0
    return_flag = 0
    fuel_total = 0
    turns = 0
    while not done:
        turns += 1
        random_act = np.random.uniform(0, np.sqrt(2), 2)
        fuel_total += dyn.norm(random_act[0], random_act[1])
        state, reward, done, info = env.step(random_act)
        if not is_halfway and (dyn.angle_diff(state[0], state[1], state[4], state[5]) <= SECTOR) and abs(dyn.norm(state[0], state[1])-dyn.GEO) < BOUND:
            is_halfway = True
            capture_flag = turns
        if is_halfway and (dyn.angle_diff(state[0], state[1], state[8], state[9]) <= SECTOR) and abs(dyn.norm(state[0], state[1])-dyn.GEO) < BOUND:
            done = True
            return_flag = turns
    return (capture_flag, return_flag, fuel_total)


def main():
    # TODO: Make faster? Parallelize
    for i in range(30000):
        print(i)
        print(one_run(0))


def main2():
    # TODO: Make faster? Parallelize
    x1 = time.time()
    with Pool(16) as p:
        xs = p.map(one_run, list(range(10000)))
        print(xs)
    x2 = time.time()
    print(x2-x1)
    tot_1 = 0
    tot_2 = 0
    for x in xs:
        if x[0] > 0:
            tot_1 += 1
        if x[1] > 0:
            tot_2 += 1
    print(tot_1, tot_2)


if __name__ == "__main__":
    main2()
