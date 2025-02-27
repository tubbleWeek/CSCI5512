#!/usr/bin/env python3
import sys
import random

def likelihood_weighting(num_samples, evidence):
    T = len(evidence)
    prior = {True: 0.5, False: 0.5}
    trans = {
        True: {True: 0.7, False: 0.3},
        False: {True: 0.3, False: 0.7}
    }
    sensor = {
        True: {True: 0.9, False: 0.1},
        False: {True: 0.2, False: 0.8}
    }
    
    total_weight_true = 0.0
    total_weight = 0.0
    
    for _ in range(num_samples):
        weight = 1.0
        r = True if random.random() < prior[True] else False
        for t in range(T):
            r = True if random.random() < trans[r][True] else False
            weight *= sensor[r][evidence[t]]
        if r:
            total_weight_true += weight
        total_weight += weight
    return total_weight_true / total_weight if total_weight != 0 else 0

def main():
    if len(sys.argv) < 12:
        print("Less than 10 evidence")
        return
    num_samples = int(sys.argv[1])
    evidence = [arg.upper() == 'T' for arg in sys.argv[2:12]]
    
    runs = 10
    estimates = []
    for _ in range(runs):
        est = likelihood_weighting(num_samples, evidence)
        estimates.append(est)
    
    avg_est = sum(estimates) / len(estimates)
    variance = sum((x - avg_est) ** 2 for x in estimates) / len(estimates)
    
    print("Likelihood Weighting with {} samples (averaged over {} runs):".format(num_samples, runs))
    print("Average estimate of P(R10=True | evidence):", avg_est)
    print("Variance:", variance)

if __name__ == '__main__':
    main()
