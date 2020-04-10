import time
import torch
import argparse

def test_matmul(M, K, N, dtype='float'):
    # init tensor directly in GPU, not on cpu
    activation = torch.cuda.FloatTensor(M, K).uniform_()
    weight = torch.cuda.FloatTensor(K, N).uniform_()
    # start to record time after data ready
    time1 = time.time()
    _ = torch.matmul(activation, weight)
    time2 = time.time()
    return time2 - time1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MATMUL PROFILER')
    parser.add_argument('shapes', metavar='N', type=int, nargs=3)
    parser.add_argument('n_repeats', type=int)
    args = parser.parse_args()
    M = args.shapes[0]
    K = args.shapes[1]
    N = args.shapes[2]
    time_accum = 0.0
    for idx in range(args.n_repeats):
        time_accum += test_matmul(M, K, N)
    print(time_accum)

