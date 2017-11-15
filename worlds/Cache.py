
from . import BaseWorld

class Cache(BaseWorld):
    def __init__(self, max_size=None, delay=0, pre_add=lambda trajs, hist:hist):
        import numpy as np

        all_data = []

        def add_trajectories(trajs):
            nonlocal all_data

            all_data = pre_add(trajs, all_data)
            for t in trajs:
                obs, act, rew = zip(*t)

                # Use vertical arrays to save some memory
                obs = np.array(obs, dtype=np.float32)
                act = np.array(act, dtype=np.float32)
                rew = np.array(rew, dtype=np.float32)
                assert rew.shape == (len(t),)

                # Prevent writing to slices
                obs.setflags(write=False)
                act.setflags(write=False)
                rew.setflags(write=False)

                all_data.append((obs, act, rew))

            if max_size is not None:
                if len(all_data) > max_size:
                    all_data = all_data[-max_size:]

        rng = np.random.RandomState()

        def trajectories(agent, n):
            assert agent == None
            assert n >= 1

            if len(all_data) < 1:
                raise ValueError("Experience cache is empty")

            size = max((len(all_data) + 1) // 2, len(all_data) - delay)

            return [
                [(o, a, r) for o, a, r in zip(*all_data[i])]
                for i in rng.choice(size, size=n)
            ]

        self.add_trajectories = add_trajectories
        self.add_trajectory = lambda *ts: add_trajectories(ts)
        self.trajectories = trajectories
