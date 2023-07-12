from experiment_1_env import Env
import numpy as np

from lib import dynamics as dyn


def one_run():
    env = Env()
    state = env.reset()
    done = False
    is_halfway = False
    success_counter1 = False
    success_counter2 = True
    while not done:
        state, reward, done, info = env.step(np.random.uniform(0, 1, 2))
        if not is_halfway and (dyn.angle_diff(state[0], state[1], state[4], state[5]) <= np.pi / 16):
            is_halfway = True
        if is_halfway and (dyn.angle_diff(state[0], state[1], state[4], state[5]) <= np.pi / 16):
            done = True
            success_counter2 = True
    return (success_counter1, success_counter2)

def main():
    # TODO: Make faster? Parallelize
    for i in range(30000):
        print(i)
        print(one_run())



if __name__ == "__main__":
    main()