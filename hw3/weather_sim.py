"""Probabilistic Robotics, Chapter 2, Exercise 2(b)(c).

Run with Python 3: python weather_sim.py
The simulator uses only the Python standard library.
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


if __name__ == "__main__":
    rng = random.Random(42)
    state = 0  # Day 1 is sunny.
    sequence = [NAMES[state]]
    for _ in range(29):
        state = next_weather(state, rng)
        sequence.append(NAMES[state])
    print("Exercise 2(b): 30 simulated days")
    print(", ".join(sequence))

    rng = random.Random(42)
    state = 0
    burn_in, samples = 1000, 1_000_000
    for _ in range(burn_in):
        state = next_weather(state, rng)
    counts = [0, 0, 0]
    for _ in range(samples):
        state = next_weather(state, rng)
        counts[state] += 1
    print("\nExercise 2(c): empirical stationary distribution")
    print(f"seed=42, burn-in={burn_in}, samples={samples}")
    for name, count in zip(NAMES, counts):
        print(f"{name:6s}: count={count:7d}, probability={count / samples:.6f}")
