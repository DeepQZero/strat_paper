from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
# TODO: stack callback with evaluation callback
# environment with no noise
class CaptureCallback(BaseCallback):
    def __init__(self, verbose=1):
        self.prev_num_captures = 0
        self.new_num_captures  = 0
        super(CaptureCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.model.get_env().env_method("is_capture")[0]:
            self.new_num_captures += 1
        return True

    def _on_rollout_end(self):
        self.logger.record("num_captures", (self.new_num_captures - self.prev_num_captures))
        self.prev_num_captures = self.new_num_captures
        return True

