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

    e = [x.upper() == 'T' for x in evidence]
    n = len(e)
    
    prior = {True: 0.5, False: 0.5}
    
    trans = {
        True: {True: 0.7, False: 0.3},
        False: {True: 0.4, False: 0.6}
    }

    sensor = {
        True: {True: 0.9, False: 0.1},
        False: {True: 0.3, False: 0.7}
    }

    f = []
    f1 = {}
    for x in [True, False]:
        prob = 0.0
        for prev in [True, False]:
            prob += trans[prev][x] * prior[prev]
        f1[x] = sensor[x][e[0]] * prob
    f1 = normalize(f1)
    f.append(f1)
    
    for t in range(1, n):
        ft = {}
        for x in [True, False]:
            prob = 0.0
            for prev in [True, False]:
                prob += trans[prev][x] * f[t-1][prev]
            ft[x] = sensor[x][e[t]] * prob
        ft = normalize(ft)
        f.append(ft)

    b = [None] * n
    b_last = {True: 1.0, False: 1.0}
    b[n-1] = b_last
    
    for t in range(n-2, -1, -1):
        bt = {}
        for x in [True, False]:
            total = 0.0
            for x_next in [True, False]:
                total += trans[x][x_next] * sensor[x_next][e[t+1]] * b[t+1][x_next]
            bt[x] = total
        bt = normalize(bt)
        b[t] = bt

    smoothed = []
    for t in range(n):
        posterior = {}
        for state in [True, False]:
            posterior[state] = f[t][state] * b[t][state]
        posterior = normalize(posterior)
        smoothed.append(posterior[True])
    
    print(smoothed)

if __name__ == '__main__':
    main()
