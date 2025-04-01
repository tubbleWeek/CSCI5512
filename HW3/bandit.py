import sys
import math
import random

class Bandit:
    '''Class that defines an n-armed bandit

    Create an n-armed bandit with random seed s:
    b = Bandit(n, s)

    then pull arm k using b.pull(k)

    You can pass a seed for the random number generator into the 
    constructor. Each arm has its own seeded random number generator
    '''
    
    def __init__(self, n, seed):
        self.arms = []
        for i in range(n):
            r = random.Random(seed + i)
            mu = r.randrange(20, 200)
            sigma = r.uniform(1, 30)
            self.arms.append((r, mu, sigma))

    def pull(self, k):
        r, mu, sigma = self.arms[k]
        return r.gauss(mu, sigma)
            
def main():
    # b = Bandit(4, 1234)
    if len(sys.argv) != 3:
        print("Usage: python3 bandit_agent.py <iterations> <seed>")
        sys.exit(1)
    
    iterations = int(sys.argv[1])
    seed = int(sys.argv[2])
    random.seed(seed)  # Seed for agent's exploration
    
    bandit = Bandit(5, seed)
    optimal_mu = max(arm[1] for arm in bandit.arms)
    
    counts = [0] * 5
    values = [0.0] * 5
    total_reward = 0.0
    total_regret = 0.0
    epsilon = 0.1
    for t in range(iterations):
        
        
        if random.random() < epsilon:
            chosen_arm = random.randint(0, 4)
        else:
            max_val = max(values)
            best_arms = [i for i, v in enumerate(values) if v == max_val]
            chosen_arm = random.choice(best_arms)
        
        reward = bandit.pull(chosen_arm)
        counts[chosen_arm] += 1
        values[chosen_arm] += (reward - values[chosen_arm]) / counts[chosen_arm]
        
        total_reward += reward
        total_regret += (optimal_mu - reward)
    
    print(f"With seed {seed}, after {iterations} iterations, total reward is {total_reward:.2f}, estimated regret is {total_regret:.2f}.")

if __name__ == '__main__':
    main()
