"""Probabilistic Robotics, Chapter 2, Exercise 2(b).
"""

import random

NAMES = ["sunny", "cloudy", "rainy"]
T = [[0.8, 0.2, 0.0],
     [0.4, 0.4, 0.2],
     [0.2, 0.6, 0.2]]


def next_weather(state, rng):
    u = rng.random()
    cumulative = 0.0
    for j, probability in enumerate(T[state]):
        cumulative += probability
        if u < cumulative:
            return j
    return len(NAMES) - 1

rng = random.Random(42)
state = 0  # Day 1 is sunny.
sequence = [NAMES[state]]
for _ in range(29):
    state = next_weather(state, rng)
    sequence.append(NAMES[state])
print("Exercise 2(b): 30 simulated days")
print(", ".join(sequence))