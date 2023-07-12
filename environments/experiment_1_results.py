from experiment_1_env import Env
import numpy as np
from multiprocessing import Pool
import time

from lib import dynamics as dyn

BOUND = 20e6

def one_run(y):
    env = Env()
    state = env.reset()
    done = False
    is_halfway = False
    success_counter1 = False
    success_counter2 = False
    while not done:
        state, reward, done, info = env.step(np.random.uniform(0, 1, 2))
        if not is_halfway and (dyn.angle_diff(state[0], state[1], state[4], state[5]) <= np.pi / 16) and abs(dyn.norm(state[0], state[1])-dyn.GEO) < BOUND:
            is_halfway = True
        if is_halfway and (dyn.angle_diff(state[0], state[1], state[4], state[5]) <= np.pi / 16) and abs(dyn.norm(state[0], state[1])-dyn.GEO) < BOUND:
            done = True
            success_counter2 = True
    return (success_counter1, success_counter2)


def main():
    # TODO: Make faster? Parallelize
    for i in range(30000):
        print(i)
        print(one_run())


def main2():
    # TODO: Make faster? Parallelize
    x1 = time.time()
    with Pool(16) as p:
        xs = p.map(one_run, list(range(1000)))
        print(xs)
    x2 = time.time()
    print(x2-x1)
    tot_1 = 0
    tot_2 = 0
    for x in xs:
        if x[0]:
            tot_1 += 1
        if x[1]:
            tot_2 += 1
    print(tot_1, tot_2)

if __name__ == "__main__":
    main2()
