import matplotlib.pyplot as plt
import numpy as np
import random


class NoStationaryBandit:
    def __init__(self):
        self.actual_value = 0.5

    @property
    def value(self):
        self.actual_value += np.random.normal()
        return self.actual_value


class Bandit:
    def __init__(self):
        self.actual_value = np.random.normal()

    @property
    def value(self):
        return self.actual_value + np.random.normal()


class Record:
    def __init__(self):
        self.values = []

    def append(self, value):
        self.values.append(value)


class ValueRecords:
    def __init__(self, num):
        self.value_records = [Record() for _ in range(num)]

    def __getitem__(self, item):
        return self.value_records[item].values

    def __len__(self):
        return len(self.value_records)


class IncrementRecord:
    def __init__(self):
        self.last_time_estimated = 0
        self.last_time_reward = 0
        self.chosen_time = 0

    def update_estimate(self, new_estimate):
        self.last_time_estimated = new_estimate

    def update_reward(self, reward):
        self.last_time_reward = reward
        self.chosen_time += 1


def estimate(increment_records):
    new_estimates = []
    for record in increment_records:
        k = record.chosen_time
        q_k = record.last_time_estimated
        r_k = record.last_time_reward
        new_estimate = 0

        if k > 0:
            new_estimate = q_k + 1 / k * (r_k - q_k)
            record.update_estimate(new_estimate)
        # alpha = 0.1
        #
        # new_estimate = q_k + alpha * (r_k - q_k)
        record.update_estimate(new_estimate)

        new_estimates.append(new_estimate)

    return new_estimates


def epsilon_greedy_policy(epsilon):
    def choose_action(estimated_values):
        if random.random() < epsilon:
            max_index = np.random.choice(range(len(estimated_values)))
        else:
            estimated_values = np.array(estimated_values)
            max_index = np.random.choice(np.flatnonzero(estimated_values == np.max(estimated_values)))

        return max_index

    return choose_action


def softmax_policy(tau):
    def choose_action(estimated_values):
        eps = 1e-5
        estimated_values = np.array(estimated_values)
        estimated_values -= np.max(estimated_values)
        estimated_values = np.exp(estimated_values / (tau + eps))
        probability_distribution = estimated_values / np.sum(estimated_values)

        return np.random.choice(range(len(estimated_values)), p=probability_distribution)

    return choose_action


def choose_bandit(step, policy_func, bandit_num=10):
    bandits = [NoStationaryBandit() for _ in range(bandit_num)]

    records = [IncrementRecord() for _ in range(bandit_num)]
    total_rewards = 0
    _average_rewards = []

    for i in range(step):
        estimated_values = estimate(records)
        chosen_index = policy_func(estimated_values)

        bandit = bandits[chosen_index]
        reward = bandit.value

        records[chosen_index].update_reward(reward)

        total_rewards += reward

        _average_rewards.append(total_rewards/(i+1))

        # print('average reward is {}'.format(total_rewards/(i+1)))

    return _average_rewards


def average_loops_choose_bandit(policy_func, step=1000, loop_time=2000):
    results = []
    for i in range(loop_time):
        if i%100 == 0: print('{}/{}'.format((i+1), loop_time))
        results.append(choose_bandit(step=step, policy_func=policy_func))

    return np.mean(np.array(results), axis=0)


def asserts():
    bandit_num = 10
    bandits = [Bandit() for i in range(bandit_num)]

    assert bandits[0].value != bandits[1].value

    value_records = ValueRecords(10)
    value_records[0].append(0.1)
    assert value_records[0] == [0.1], value_records[0]

    increment_records = [IncrementRecord() for _ in range(10)]

    estimated_values = estimate(increment_records)
    assert len(estimated_values) == 10

    assert estimated_values[0] == 0

    increment_records[1].update_reward(0.1)
    estimated_values = estimate(increment_records)
    # assert estimated_values[1] == 0.1
    assert estimated_values[0] == 0

    greedy = epsilon_greedy_policy(epsilon=0.1)

    target_num = 0
    for i in range(100):
        chosen_index = greedy(estimated_values)
        if chosen_index == 1: target_num += 1

    assert abs(target_num - 100 * (1 - 0.1)) <= 5

    # value_records[1].append(0.2)
    # estimated_values = estimate(value_records)
    # assert epsilon_greedy_policy(0)(estimated_values) == 1

    print('test done!')


asserts()


if __name__ == '__main__':
    # choose_bandit(1000)
    loop_time = 2000
    step = 1000

    greedy = epsilon_greedy_policy(epsilon=0)
    little_epsilon = epsilon_greedy_policy(epsilon=0.01)
    few_epsilon = epsilon_greedy_policy(epsilon=0.1)

    soft_max_little = softmax_policy(tau=0.1)
    soft_max_few = softmax_policy(tau=1)
    no_softmax = softmax_policy(tau=0)

    policy_functions = [
        # (soft_max_little, r'$\tau = 0.1$'),
        # (soft_max_few, r'$\tau = 1 $'),
        # (no_softmax, r'$tau=0$'),
        # (greedy, 'greedy'),
        # (little_epsilon, '$\epsilon = 0.01 $'),
        (few_epsilon, '$\epsilon = 0.1 $'),
    ]

    legends = []
    for policy, legend in policy_functions:
        average_rewards = average_loops_choose_bandit(policy_func=policy, step=step, loop_time=loop_time)
        plt.plot(average_rewards)
        legends.append(legend)

    plt.legend(legends)
    plt.savefig('img/no-stationary-with-no-constant-alpha.png')
    print('plot done!')
    plt.show()


