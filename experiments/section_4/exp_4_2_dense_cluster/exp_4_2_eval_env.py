# TODO: First start in the regular start state - look for captures
# TODO: Fill the state buffer with states in the beginning of capture trajectories
# Also try with evaluating with the default start state and using a prefilled capture buffer
# RecurrentPPO policy
# Potentially decrease the learning rate (default 3e-4)
import random
import pickle

import numpy as np

from exp_4_2_env import Env
from lib import dynamics as dyn
class EvalEnv(Env):
    def __init__(self): # Start with the base start state
        super().__init__()
        self.unproc_state = []
        self.state_buffer = []
        self.NUM_STATES = 250
        self.fill_state_buffer_with_default()
        #self.fill_state_buffer_with_captures()

    def init2(self): # __init__
        super().__init__()
        self.NUM_STATES = 250
        self.state_buffer = []
        self.unproc_state = []
        print("FILLING EVAL ENV STATE BUFFER")
        self.fill_state_buffer()
        print("DONE FILLING")

    def det_obs(self) -> np.ndarray:
        """Returns observation per Gym standard."""
        # distance from GEO, velocity vector, angle from enemy base
        # rotate observations so the enemy base is always at zero
        # TODO: Rotate everything backwards by the angle between enemy position and the normal
        #       Save actions with a capture trajectory and try those actions in another environment
        state = super().det_obs()
        self.unproc_state = state
        mobile_pos = (state[0:2] - dyn.GEO) / 5e6
        mobile_vel = (state[2:4]) / dyn.BASE_VEL_Y
        enemy_pos  = (state[8:10] - dyn.GEO) / 5e6
        return np.concatenate((mobile_pos, mobile_vel, enemy_pos, [state[12]], [state[13]]))

    def choose_action(self, state):
        # if np.random.uniform(0, 1) < 0.0 or state[13] > self.MAX_FUEL:
        #    rand_act = np.array([0.0, 0.0])
        for u in [5.5, 6, 6.5, 7, 7.5]:
            if (((self.MAX_FUEL - state[7]) / self.MAX_FUEL) < ((8 - u) * 0.125) + 0.02) and (
                    dyn.abs_angle_diff(state[0:2], state[4:6]) > (u * np.pi / 8)):
                return np.array([0.0, 0.0])
        thrust = 10
        rand_thrust = np.random.uniform(0, thrust)
        rand_angle = np.random.uniform(0, 2 * np.pi)
        rand_act = np.array([np.cos(rand_angle), np.sin(rand_angle)]) * rand_thrust
        return rand_act

    def fill_state_buffer_with_default(self):
        env = Env()
        while len(self.state_buffer) < self.NUM_STATES:
            default_state, _ = env.reset()
            self.state_buffer.append(default_state)
        return

    def fill_state_buffer_with_captures(self): # Working
        data_collector = pickle.load(open("capture_buffer.pkl", "rb"))
        for capture_traj in data_collector.capture_buffer:
            trunc_capture_traj = capture_traj[:28]
            while len(self.state_buffer) < self.NUM_STATES:
                choices = random.sample(trunc_capture_traj, random.randint(1, 10))
                for c in choices:
                    self.state_buffer.append(c) if len(self.state_buffer) < self.NUM_STATES else None
        return

    def fill_state_buffer(self):
        i = 0
        while len(self.state_buffer) < self.NUM_STATES:
            if (i % 100) == 0:
                print(i)
            i += 1
            state, _ = self.reset()
            NUM_TURNS = 0
            done = False
            while not done and NUM_TURNS < self.MAX_TURNS/2:
                action_choice = self.choose_action(state)
                state, reward, done, _, info = self.step(action_choice)
                NUM_TURNS += 1
                if self.eval_state(state):
                    if not self.in_buffer(self.unproc_state):
                        self.state_buffer.append(self.unproc_state)
                        break

    def in_buffer(self, state):
        for s in self.state_buffer:
            if np.linalg.norm(state - s)**2 < 1e-8:
                return True
        return False

    def eval_state(self, state):
        # Take the start state and add random perturbations to the X and Y positions
        # Like 100 meters off, instead of using fuel
        # Add a rotating frame (maybe later)
        angle = dyn.abs_angle_diff(state[0:2] * 5e6 + dyn.GEO, state[4:6] * 5e6 + dyn.GEO)
        dist = dyn.vec_norm(state[0:2]) * 5e6
        in_zone = abs(dist - dyn.GEO) < 5e6
        angle_left_proportion = angle / np.pi
        fuel_left_proportion = (self.MAX_FUEL - state[7]) / self.MAX_FUEL
        turn_left_proportion = (self.MAX_TURNS - state[6]) / self.MAX_TURNS
        good_fuel = (angle_left_proportion - 0.05) < fuel_left_proportion
        good_turn = (angle_left_proportion - 0.05) < turn_left_proportion
        #print(angle_left_proportion, fuel_left_proportion, turn_left_proportion)
        # if in_zone and good_fuel and good_turn:
        #     self.state_buffer.append(state)
        if in_zone and good_fuel and good_turn:
            distance = dyn.vec_norm(state[0:2] - state[4:6]) * 5e6
            print(angle_left_proportion, fuel_left_proportion, turn_left_proportion, distance)
            return True
        return False

    def reset(self, seed=None, options=None):
        if len(self.state_buffer) > 0 and np.random.uniform(0, 1) < 0.2:
            state_choice = random.choice(self.state_buffer)
            return self.det_reset_helper(state_choice)
        return super().reset()