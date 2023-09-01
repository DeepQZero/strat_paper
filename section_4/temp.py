from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
# TODO: stack callback with evaluation callback
# environment with no noise
class CaptureCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(CaptureCallback, self).__init__(verbose)
        self.prev_num_captures = 0
        self.new_num_captures  = 0

    def _on_step(self) -> bool:
        if self.model.get_env().env_method("is_capture")[0]:
            self.new_num_captures += 1
        return True

    def _on_rollout_end(self):
        self.logger.record("num_captures", (self.new_num_captures - self.prev_num_captures))
        self.prev_num_captures = self.new_num_captures
        return True

def fill_capture_buffer():
    data_collector = DataCollector()
    env = Env()
    i = 0
    while len(data_collector.capture_buffer) < 250:
        if (i % 100) == 0:
            print(i)
        if np.random.uniform(0, 1) < 0.5:
            state = data_collector.buffer_sample()
            if state is None:
                state, _ = env.reset()
            else:
                for u in data_collector.start_trajectory_buffer:
                    # data_collector.start_trajectory_buffer.pop(u)
                    if np.linalg.norm(u[0] - state) < 1e-6:
                        data_collector.current_trajectory = u[1]
                env.det_reset_helper(state)
        else:
            state, _ = env.reset()
        done = False
        while not done:
            rand_act = data_collector.choose_action(state)
            state, reward, done, _, info = env.step(rand_act)
            data_collector.filter_state(state)
            data_collector.current_trajectory.append(state)
            if env.is_capture():
                print("HAD A CAPTURE")
                print(data_collector.current_trajectory)
                data_collector.start_trajectory_buffer.append([state, data_collector.current_trajectory])
                data_collector.capture_buffer.append(data_collector.current_trajectory)
        data_collector.current_trajectory = []
        i += 1
    pickle.dump(data_collector, open("capture_buffer.pkl", "wb"))