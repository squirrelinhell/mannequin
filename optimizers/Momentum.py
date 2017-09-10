
from . import Optimizer

class RunningMean(object):
    def __init__(self, decay):
        decay = float(decay)
        assert decay > 0.0
        assert decay < 1.0

        biased_mean = 0.0
        decay_power = 1.0

        def update(value):
            nonlocal biased_mean, decay_power

            biased_mean = biased_mean * decay + value * (1.0 - decay)
            decay_power *= decay

        self.get = lambda: biased_mean / (1.0 - decay_power)
        self.update = update

class Momentum(Optimizer):
    def __init__(self, value, lr, decay):
        import numpy as np
        import os

        value = np.asarray(value)
        value.setflags(write=False)

        lr = float(lr)
        assert lr > 0.0

        running_mean = RunningMean(decay)
        last_update = 0.0

        def norm(v):
            return np.sqrt(np.sum(np.square(v)))

        def get_info():
            return ("last update: %.2f" % norm(last_update))

        def get_requests():
            return [value]

        def feed_gradients(gradients):
            nonlocal value, last_update

            assert len(gradients) == 1
            grad = np.asarray(gradients[0])
            assert grad.shape == value.shape

            running_mean.update(grad)
            last_update = lr * running_mean.get()

            value = value + last_update
            value.setflags(write=False)

        self.get_info = get_info
        self.get_best_value = lambda: value
        self.get_requests = get_requests
        self.feed_gradients = feed_gradients