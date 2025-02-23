import sys
#!/usr/bin/env python3
import sys

def normalize(dist):
    total = sum(dist.values())
    if total == 0:
        return dist
    return {k: v / total for k, v in dist.items()}

def main():
    evidence = sys.argv[1:]
    if not evidence:
        print("Usage: smooth_hmm.py <EVIDENCE SEQUENCE(T F ...)>")
        sys.exit(1)
    # Convert evidence to booleans: True for 'T', False for 'F'
    e = [x.upper() == 'T' for x in evidence]
    n = len(e)
    
    # HMM parameters:
    # Prior: P(X0 = T) = 0.5, P(X0 = F) = 0.5.
    prior = {True: 0.5, False: 0.5}
    
    # Transition model:
    #   P(X_{t+1} = T | X_t = T) = 0.7, so F=0.3;
    #   P(X_{t+1} = T | X_t = F) = 0.4, so F=0.6.
    trans = {
        True: {True: 0.7, False: 0.3},
        False: {True: 0.4, False: 0.6}
    }
    
    # Sensor model:
    #   P(e_t = T | X_t = T) = 0.9, so F=0.1;
    #   P(e_t = T | X_t = F) = 0.3, so F=0.7.
    sensor = {
        True: {True: 0.9, False: 0.1},
        False: {True: 0.3, False: 0.7}
    }
    
    # ----- Forward Pass -----
    # f[t] will be a dict giving P(X_t, e_1:t).
    f = []
    # Time t = 1 (first observation): sum over X0.
    f1 = {}
    for x in [True, False]:
        prob = 0.0
        for prev in [True, False]:
            prob += trans[prev][x] * prior[prev]
        # Incorporate sensor evidence for time 1.
        f1[x] = sensor[x][e[0]] * prob
    f1 = normalize(f1)
    f.append(f1)
    
    # For t = 2 to n.
    for t in range(1, n):
        ft = {}
        for x in [True, False]:
            prob = 0.0
            for prev in [True, False]:
                prob += trans[prev][x] * f[t-1][prev]
            ft[x] = sensor[x][e[t]] * prob
        ft = normalize(ft)
        f.append(ft)
    
    # ----- Backward Pass -----
    # b[t] will be a dict giving P(e_{t+1:n} | X_t)
    b = [None] * n
    # Base case: at time n, there is no future evidence.
    b_last = {True: 1.0, False: 1.0}
    b[n-1] = b_last
    
    # For t = n-1 down to 1.
    for t in range(n-2, -1, -1):
        bt = {}
        for x in [True, False]:
            total = 0.0
            for x_next in [True, False]:
                total += trans[x][x_next] * sensor[x_next][e[t+1]] * b[t+1][x_next]
            bt[x] = total
        bt = normalize(bt)
        b[t] = bt
    
    # ----- Smoothing -----
    # Compute the smoothed estimate for each time: proportional to f[t] * b[t].
    smoothed = []
    for t in range(n):
        posterior = {}
        for state in [True, False]:
            posterior[state] = f[t][state] * b[t][state]
        posterior = normalize(posterior)
        # We output the probability that X_t = T.
        smoothed.append(posterior[True])
    
    print(smoothed)

if __name__ == '__main__':
    main()
