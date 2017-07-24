import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


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


def choose_action(estimated_values):
    estimated_values = np.array(estimated_values)
    max_index = np.random.choice(np.flatnonzero(estimated_values == np.max(estimated_values)))
    return max_index


def choose_bandit(step, bandit_num=10):
    bandits = [Bandit() for _ in range(bandit_num)]

    value_records = ValueRecords(bandit_num)
    total_rewards = 0
    average_rewards = []
    for i in range(step):
        estimated_values = estimate(value_records)
        chosen_index = choose_action(estimated_values)

        bandit = bandits[chosen_index]
        reward = bandit.value

        total_rewards += reward

        if i % 100 == 0:
            # print('index: {}'.format(chosen_index))
            # print(bandits[chosen_index].actual_value)
            # print(value_records[chosen_index])
            print('average reward is {}'.format(total_rewards/(i+1)))

        value_records[chosen_index].append(reward)
        average_rewards.append(total_rewards/(i+1))

        # print('average reward is {}'.format(total_rewards/(i+1)))

    return average_rewards


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
    assert choose_action(estimated_values) == 0

    value_records[1].append(0.2)
    estimated_values = estimate(value_records)
    assert choose_action(estimated_values) == 1

    print('test done!')


asserts()


if __name__ == '__main__':
    choose_bandit(1000)


