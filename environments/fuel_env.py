import numpy as np


class Environment:
    def __init__(self):
        self.state = None
        self.LOW_ORBIT = 50  # TODO refactor to include these fields
        self.HIGH_ORBIT = 60

    def reset(self, loc=(55, 5/360*2*np.pi)):
        self.state = loc
        return self.det_state()

    def step(self, action):
        # action = self.action_decoder(action)
        r, theta = self.state
        old_x, old_y = r * np.cos(theta), r * np.sin(theta)
        new_x = old_x + action[0] + np.random.normal(0.01)
        new_y = old_y + action[1] + np.random.normal(0.01)
        new_r = np.sqrt(new_x**2 + new_y**2)
        new_theta = np.arctan2(new_y, new_x) % (2 * np.pi)
        self.state = new_r, new_theta
        is_win =  (350/360 * 2 * np.pi < new_theta < 2 * np.pi) and (50 < new_r < 60) and not\
            (0 < theta < 10/360 * 2 * np.pi)
        is_backwards = (350/360 * 2 * np.pi < new_theta < 2 * np.pi) and (0 < theta < 10/360 * 2 * np.pi)
        is_oob = not (50 < new_r < 60)
        done = is_win or is_backwards or is_oob
        rew = self.det_rew(theta, new_theta, is_win, is_backwards, is_oob)
        return self.state, rew, done, {}

    def det_state(self):
        r, theta = self.state
        return np.array([(r-55)/10, theta])

    def det_rew(self, old_theta, new_theta, win, backwards, oob):
        delta_theta = new_theta - old_theta
        if win:
            return 10
        elif backwards or oob:
            return -10
        else:
            return 0 #delta_theta

    def action_decoder(self, action):
        if action == 0:
            return 1, 1
        elif action == 1:
            return 1, 0
        elif action == 2:
            return 1, -1
        elif action == 3:
            return 0, 1
        elif action == 4:
            return 0, -1
        elif action == 5:
            return -1, 1
        elif action == 6:
            return -1, 0
        elif action == 7:
            return -1, -1
        elif action == 8:
            return 0, 0