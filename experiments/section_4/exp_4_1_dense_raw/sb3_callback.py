from stable_baselines3.common.callbacks import BaseCallback
# TODO: stack callback with evaluation callback
# environment with no noise
class CaptureCallback(BaseCallback):
    def __init__(self, verbose=1):
        self.num_captures = 0
        super(CaptureCallback, self).__init__(verbose)

    def _on_step(self):
        if self.training_env.env_method("is_capture"): # TODO: Doesn't work
            self.num_captures += 1
            self.logger.record("num_captures", self.num_captures)
        return True
