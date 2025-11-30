import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    td_target = r + gamma * np.max(Q[sprime])
    td_error = td_target - Q[s, a]
    Q[s, a] += alpha * td_error
    return Q


def epsilon_greedy(Q, s, epsilone):
    if np.random.rand() < epsilone:
        return np.random.randint(0, Q.shape[1])
    else:
        return np.argmax(Q[s])


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1
    gamma = 0.95
    epsilon = 0.2

    n_epochs = 2000
    max_itr_per_epoch = 200
    rewards = []

    for e in range(n_epochs):
        r = 0

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime,
                               alpha=alpha, gamma=gamma)

            S = Sprime

            if done:
                break

        print("episode #", e, " : r = ", r)
        rewards.append(r)

    print("Average reward = ", np.mean(rewards))
    print("Training finished.\n")

    env.close()
