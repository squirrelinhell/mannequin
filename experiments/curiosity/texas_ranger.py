#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")

from worlds import Gym, StochasticPolicy, BaseWorld, ActionNoise, Cache
from models import Input, Affine, LReLU, Softmax, Tanh
from optimizers import Adam
from trajectories import policy_gradient, normalize, discount, print_reward, accuracy, get_rewards, replace_rewards

if "DEBUG" in os.environ:
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

AGENT_VARIATIONS=8
TRAJS_PER_VARIATION=4
EPISODE_DELAY=0
MAX_HISTORY=60

def cleanHistory(trajs, history):
    toAddSize = len(trajs)
    currentHistorySize = len(history)
    np.random.shuffle(history)
    toCut = toAddSize//2 + currentHistorySize//20
    return history[:-toCut]

class Curiosity(BaseWorld):
    def __init__(self, inner, *, classifier, for_classifier=lambda x: x, plot=None):
        history = Cache(pre_add=cleanHistory, delay=TRAJS_PER_VARIATION * AGENT_VARIATIONS * EPISODE_DELAY)
        classOpt = None

        def tag_traj(traj, tag):
            return [(t[0], tag, t[2]) for t in traj]

        def reset_classifier():
            nonlocal classOpt, history
            classOpt = None
            history = Cache(pre_add=cleanHistory, delay=TRAJS_PER_VARIATION * AGENT_VARIATIONS * EPISODE_DELAY)

        def trajectories(agents, n):
            nonlocal classOpt
            if classOpt is None:
                classOpt = Adam(
                    np.random.randn(classifier.n_params) * 1.,
                    lr=0.06,
                    memory=0.9,
                )
                history.add_trajectories(inner.trajectories(agents[-1], n))

            oldTrajs, innerTrajs = [], []
            for agent in agents:
                oldTrajs += history.trajectories(None, n)
                innerTrajs += inner.trajectories(agent, n)

            history.add_trajectories(innerTrajs)

            trajsForClass = for_classifier(
                [tag_traj(traj, [1,0]) for traj in oldTrajs]
                + [tag_traj(traj, [0,1]) for traj in innerTrajs]
            )
            trajsForClass = replace_rewards(trajsForClass, reward=lambda r: 1.)

            classifier.load_params(classOpt.get_value())
            grad = policy_gradient(trajsForClass, policy=classifier)
            classOpt.apply_gradient(grad)

            if plot is not None:
                plot(trajsForClass, classifier)

            curiosityTrajs = replace_rewards(
                for_classifier(innerTrajs),
                model=classifier,
                reward=lambda o: o[1],
            )
            return innerTrajs, curiosityTrajs

        self.trajectories = trajectories
        self.render = inner.render
        self.reset_classifier = reset_classifier

from models.BaseWrapper import BaseWrapper

def combine_rewards(trajss, weights):
    assert len(trajss) == len(weights)
    rewards = []
    for trajs in trajss:
        rewards.append(np.asarray([np.asarray([r for (_, _, r) in traj]) for traj in trajs]))
    rewards = [rew*weight for rew, weight in zip(rewards, weights)]
    rewards = np.sum(rewards, axis=0)
    return [[(o, a, r) for (o, a, _), r in zip(traj, rew)] for traj, rew in zip(trajss[0], rewards)]

def change_obs_space(trajs, changer=lambda x:x):
    return [[(changer(o, r),a,r) for o, a, r in t] for t in trajs]

def interesting_part(obs, r):
    result = (
            obs[2],
            obs[3],
            obs[1],
            )
    return result

STATE_SIZE = 24
ACTION_SIZE = 4
MAX_STEPS = 500

def walker():
    walker = Input(STATE_SIZE)
    walker = Affine(walker, 64)
    walker = LReLU(walker)
    walker = Affine(walker, ACTION_SIZE)
    walker = Tanh(walker)
    return walker

def run():
    classifier = Input(3)
    classifier = Affine(classifier, 32)
    classifier = LReLU(classifier)
    classifier = Affine(classifier, 32)
    classifier = LReLU(classifier)
    classifier = Affine(classifier, 2)
    classifier = Softmax(classifier)

    agent = walker()
    agent.load_params(np.random.randn(agent.n_params)*1.5)
    agents = [walker() for _ in range(AGENT_VARIATIONS-1)]

    MAX_TRAIN_TIME = 200
    trainTime = 0
    curAgentId = -1
    agentData = {"meanRewards":[],"firstPositive":[]}
    def plot_tagged_trajs(trajs, classifier):
        nonlocal trainTime, curAgentId
        COLORS = ["blue", "red"]
        #coords = np.mgrid[0:11,0:11,0:11,0:11,0:11].reshape(5, -1).T * [0.4, 0.04, 0.25, 0.25, 0.2] - [2.0, 0.2, 1.25, 1.25, 1.0]
        coords = np.mgrid[0:21,0:21,0:21].reshape(3, -1).T * [0.25, 0.25, 0.02] - [1.25, 1.25, 0.2]
        #coords = np.mgrid[0:21,0:21].reshape(2, -1).T * [0.1, 0.1] - [1., 1.]
        classifierResults = classifier.outputs(coords)[:,1].reshape(21,21,21)
        classifierResults = np.mean(classifierResults, axis=(2)).T[::-1,:]
        #classifierResults = classifierResults.T[::-1,:]
        plt.clf()
        plt.suptitle("Episode %d of agent %d"%(trainTime, curAgentId))
        for traj in trajs:
            tag = traj[0][1]
            xs, ys = [], []
            for state, _, _ in traj:
                x = state[0]
                y = state[1]
                xs.append(x)
                ys.append(y)
            plt.plot(xs, ys, color=COLORS[np.argmax(tag)], alpha=0.1)
        plt.imshow(classifierResults, zorder=0, aspect="auto", vmin=0.0, vmax=1.0, cmap="gray", interpolation="bicubic", extent=[-1., 1., -1., 1.])
        plt.gcf().set_size_inches(10, 8)
        plt.gcf().savefig(
            "__step_a%03d_t%05d.png" %
                (curAgentId, trainTime),
            dpi=100
        )


    world = Gym("BipedalWalker-v2", max_steps=MAX_STEPS)
    world = ActionNoise(world, stddev=0.2)
    world = Curiosity(
                world,
                classifier=classifier,
                for_classifier=lambda ts: change_obs_space(
                    ts,
                    changer=interesting_part
                    ),
                plot=plot_tagged_trajs
            )

    agentOpt = None

    def save_agent():
        np.save(
            "__ranger_a%03d_t%05d.npy" %
                (curAgentId, trainTime),
            agentOpt.get_value()
        )
    def reset_agent():
        nonlocal agentOpt, trainTime, curAgentId
        print("Resetting agent %d."%curAgentId)
        agentOpt = Adam(
            np.random.randn(agent.n_params)*0.5,
            lr=0.05,
            memory=0.9,
        )
        trainTime = 0
        curAgentId += 1
    reset_agent()
    curRewards = []
    firstPositive = MAX_TRAIN_TIME
    while True:
        parameters = agentOpt.get_value()

        agent.load_params(parameters)
        for i in range(AGENT_VARIATIONS-1):
            agents[i].load_params(parameters + np.random.randn(*parameters.shape)*0.9)
        realTrajs, curiosityTrajs = world.trajectories(agents+[agent], TRAJS_PER_VARIATION)

        print_reward(realTrajs, max_value=100.0, episode=np.sum, label="Real reward:    ")
        curReward = np.mean(get_rewards(realTrajs, episode=np.mean))
        curRewards.append(curReward)
        if firstPositive == MAX_TRAIN_TIME and curReward > 0.01:
            firstPositive = trainTime
        print_reward(curiosityTrajs, max_value=1.0, episode=np.max, label="Curiosity reward: ")
        if trainTime >= MAX_TRAIN_TIME:
            print("Timeout.")
            agentData["meanRewards"].append(np.mean(curRewards))
            curRewards = []
            agentData["firstPositive"].append(firstPositive)
            firstPositive = MAX_TRAIN_TIME
            print("Mean reward: %f; First positive at episode: %d"%(agentData["meanRewards"][-1], agentData["firstPositive"][-1]))
            save_agent()
            trainTime = 0
            reset_agent()
            if curAgentId > 10:
                print(agentData)
                print("Total mean reward: %f(deviation: %f); Mean first positive: %f(deviation: %f)"%(np.mean(agentData["meanRewards"]), np.std(agentData["meanRewards"]), np.mean(agentData["firstPositive"]), np.std(agentData["firstPositive"])))
                return
            world.reset_classifier()
            continue

        if trainTime % 20 == 0:
            save_agent()

        realTrajs = discount(realTrajs, horizon=200)
        realMean = np.abs(np.mean(get_rewards(realTrajs, episode=np.mean)))
        #curiosityTrajs = replace_rewards(curiosityTrajs, episode=np.max)
        curiosityTrajs = discount(curiosityTrajs, horizon=200)
        curiosityMean = np.mean(get_rewards(curiosityTrajs, episode=np.mean))
        curiosityTrajs = replace_rewards(curiosityTrajs, reward=lambda r: (r*realMean)/curiosityMean)
        realWeight = 0.05 + 0.9*(0.5 * (1 + np.cos(np.pi * trainTime / 40)))
        curiosityWeight = 1. - realWeight
        print("Real weight: %f; Curiosity weight: %f"%(realWeight, curiosityWeight))
        trajs = combine_rewards([realTrajs, curiosityTrajs], [realWeight, curiosityWeight])
        trajs = normalize(trajs)
        grad = policy_gradient(trajs, policy=agent)
        agentOpt.apply_gradient(grad)

        trainTime += 1

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        world = Gym("BipedalWalker-v2", max_steps=MAX_STEPS)
        agent = walker()
        for fn in sys.argv[1:]:
            agent.load_params(np.load(fn))
            world.render(agent)
    else:
        run()
