import matplotlib.pyplot as plt
import numpy as np
import random


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


def estimate(records_values):
    return [np.mean(records_values[i]) if len(records_values[i]) > 0 else 0
            for i in range(len(records_values))]


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
        estimated_values = np.array(estimated_values)
        estimated_values -= np.max(estimated_values)
        estimated_values = estimated_values / tau
        probability_distribution = estimated_values / np.sum(estimated_values)

        return np.random.choice(range(len(estimated_values)), p=probability_distribution)

    return choose_action


def choose_bandit(step, policy_func, bandit_num=10):
    bandits = [Bandit() for _ in range(bandit_num)]

    value_records = ValueRecords(bandit_num)
    total_rewards = 0
    average_rewards = []
    for i in range(step):
        estimated_values = estimate(value_records)
        chosen_index = policy_func(estimated_values)

        bandit = bandits[chosen_index]
        reward = bandit.value

        total_rewards += reward

        value_records[chosen_index].append(reward)
        average_rewards.append(total_rewards/(i+1))

        # print('average reward is {}'.format(total_rewards/(i+1)))

    return average_rewards


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

    estimated_values = estimate(value_records)
    assert len(estimated_values) == 10
    assert estimated_values[0] == 0.1
    assert estimated_values[1] == 0
    greedy = epsilon_greedy_policy(epsilon=0.1)
    assert greedy(estimated_values) == 0

    value_records[1].append(0.2)
    estimated_values = estimate(value_records)
    assert epsilon_greedy_policy(0)(estimated_values) == 1

    print('test done!')


asserts()


if __name__ == '__main__':
    # choose_bandit(1000)
    greedy = epsilon_greedy_policy(epsilon=0)
    loop_time = 1000
    step=1000
    greedy_average_rewards = average_loops_choose_bandit(policy_func=greedy, step=step, loop_time=loop_time)

    little_epsilon = epsilon_greedy_policy(epsilon=0.01)
    little_epsilon_rewards = average_loops_choose_bandit(policy_func=little_epsilon, step=step, loop_time=loop_time)

    few_epsilon = epsilon_greedy_policy(epsilon=0.1)
    few_epsilon_rewards = average_loops_choose_bandit(policy_func=few_epsilon, step=step, loop_time=loop_time)

    plt.plot(range(step), greedy_average_rewards)
    plt.plot(little_epsilon_rewards)
    plt.plot(few_epsilon_rewards)
    plt.legend(['greedy', '$\epsilon = 0.01 $', '$\epsilon = 0.1 $'])
    plt.savefig('img/greedy-with-epsilon.png')
    plt.show()


