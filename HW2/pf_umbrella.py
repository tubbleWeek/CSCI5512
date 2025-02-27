#!/usr/bin/env python3
import sys
import random

def resample(particles, weights, num_particles):
    new_particles = []
    total_weight = sum(weights)
    if total_weight == 0:
        return [particles[random.randint(0, len(particles)-1)] for _ in range(num_particles)]
    normalized = [w / total_weight for w in weights]
    
    cumulative = []
    cum_sum = 0.0
    for w in normalized:
        cum_sum += w
        cumulative.append(cum_sum)
    
    for _ in range(num_particles):
        r = random.random()
        for i, cum_val in enumerate(cumulative):
            if r <= cum_val:
                new_particles.append(particles[i])
                break
    return new_particles

def particle_filter(num_particles, evidence):

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
    
    particles = []
    for _ in range(num_particles):
        r0 = True if random.random() < prior[True] else False
        particles.append(r0)
    
    for t in range(T):
        new_particles = []
        weights = []
        for particle in particles:
            new_state = True if random.random() < trans[particle][True] else False
            new_particles.append(new_state)
            weights.append(sensor[new_state][evidence[t]])
        particles = resample(new_particles, weights, num_particles)
    
    count_true = sum(1 for state in particles if state)
    return count_true / num_particles

def main():
    if len(sys.argv) < 12:
        print("Less than 10 evidence")
        return
    num_particles = int(sys.argv[1])
    evidence = [arg.upper() == 'T' for arg in sys.argv[2:12]]
    
    runs = 10
    estimates = []
    for _ in range(runs):
        est = particle_filter(num_particles, evidence)
        estimates.append(est)
    
    avg_est = sum(estimates) / len(estimates)
    variance = sum((x - avg_est) ** 2 for x in estimates) / len(estimates)
    
    print("Particle Filtering with {} particles (averaged over {} runs):".format(num_particles, runs))
    print("Average estimate of P(R10=True | evidence):", avg_est)
    print("Variance:", variance)

if __name__ == '__main__':
    main()
