import numpy as np
import time
import torch

def test_normal(_mean, _std, _count):
    time1 = time.time()
    np.random.normal(_mean, _std, _count)
    time2 = time.time()
    print('Time cost with np.random.normal on CPU with scale of {}, ---->>>> {}'.format(_count, time2 - time1))

def test_uniform(_mean, _std, _count):
    time1 = time.time()
    a = np.random.uniform(_mean, _std, _count)
    time2 = time.time()
    print('Time cost with np.random.uniform on CPU with scale of {}, ---->>>> {}'.format(_count, time2 - time1))

def test_normal_pytorch_1d(_count, dtype='float'):
    torch.cuda.empty_cache()
    '''
    time1 = time.time()
    # torch.FloatTensor(10, 10).uniform_() is equivalent to torch.rand(10, 10)
    if dtype == 'float':
        a = torch.randn(_count, dtype=torch.float)
    elif dtype == 'half':
        a = torch.randn(_count, dtype=torch.float16)
    time2 = time.time()
    print('Time cost with 1D normal distribution on CPU with size of {}, ---->>>> {}'.format(_count, time2 - time1))
    del a
    torch.cuda.empty_cache()
    '''
    time3 = time.time()
    if dtype == 'float':
        a = torch.cuda.FloatTensor(_count).normal_()
    elif dtype == 'half':
        a = torch.cuda.HalfTensor(_count).normal_()
    time4 = time.time()
    print('Time cost with 1D normal distribution on GPU with size of {}, ---->>>> {}'.format(_count, time4 - time3))
    del a
    torch.cuda.empty_cache()

def test_uniform_pytorch_1d(_count, dtype='float'):
    torch.cuda.empty_cache()
    time1 = time.time()
    # torch.FloatTensor(10, 10).uniform_() is equivalent to torch.rand(10, 10)
    '''
    if dtype == 'float':
        a = torch.rand(_count, dtype=torch.float)
    elif dtype == 'half':
        a = torch.rand(_count, dtype=torch.float16)
    time2 = time.time()
    print('Time cost with 1D uniform distribution on CPU with size of {}, ---->>>> {}'.format(_count, time2 - time1))
    del a
    '''
    torch.cuda.empty_cache()
    time3 = time.time()
    if dtype == 'float':
        a = torch.cuda.FloatTensor(_count).normal_()
    elif dtype == 'half':
        a = torch.cuda.HalfTensor(_count).uniform_()
    time4 = time.time()
    print('Time cost with 1D uniform distribution on GPU with size of {}, ---->>>> {}'.format(_count, time4 - time3))
    del a
    torch.cuda.empty_cache()


def test_normal_pytorch_2d(_high, _width, dtype='float'):
    torch.cuda.empty_cache()
    '''
    time1 = time.time()
    # torch.FloatTensor(10, 10).uniform_() is equivalent to torch.rand(10, 10)
    if dtype == 'float':
        a = torch.randn(_high, _width, dtype=torch.float)
    elif dtype == 'half':
        a = torch.randn(_high, _width, dtype=torch.float16)
    time2 = time.time()
    print('Time cost with 2D normal distribution on CPU with height of {}, width of {}, ---->>>> {}'.format(_high, _width, time2 - time1))
    del a
    torch.cuda.empty_cache()
    '''
    time3 = time.time()
    if dtype == 'float':
        a = torch.cuda.FloatTensor(_high, _width).normal_()
    elif dtype == 'half':
        a = torch.cuda.HalfTensor(_high, _width).normal_()
    time4 = time.time()
    print('Time cost with 2D normal distribution on GPU with height of {}, width of {}, ---->>>> {}'.format(_high, _width, time4 - time3))
    del a
    torch.cuda.empty_cache()


def test_uniform_pytorch_2d(_high, _width, dtype='float'):
    '''
    torch.cuda.empty_cache()
    time1 = time.time()
    # torch.FloatTensor(10, 10).uniform_() is equivalent to torch.rand(10, 10)
    if dtype == 'float':
        a = torch.rand(_high, _width, dtype=torch.float)
    elif dtype == 'half':
        a = torch.rand(_high, _width, dtype=torch.float16)
    time2 = time.time()
    print('Time cost with 2D normal distribution on CPU with height of {}, width of {}, ---->>>> {}'.format(_high, _width, time2 - time1))
    del a
    '''
    torch.cuda.empty_cache()
    time3 = time.time()
    if dtype == 'float':
        a = torch.cuda.FloatTensor(_high, _width).uniform_()
    elif dtype == 'half':
        a = torch.cuda.HalfTensor(_high, _width).uniform_()
    time4 = time.time()
    print('Time cost with 2D normal distribution on GPU with height of {}, width of {}, ---->>>> {}'.format(_high, _width, time4 - time3))
    del a
    torch.cuda.empty_cache()
    
def test():
    a = torch.cuda.FloatTensor(10240, 224, 224, 3).uniform_()
    time1 = time.time()
    for i in range(10240):
        b = a[i] + torch.cuda.FloatTensor(224, 224, 3).normal_()
    time2 = time.time()
    print(time2 - time1)
    time3 = time.time()
    torch.cuda.FloatTensor(10240, 224, 224, 3).normal_()
    time4 = time.time()
    print(time4 - time3)

if __name__ == '__main__':
    test()
    assert(0)
    _mean = 0.0
    _std = 0.1
    test_uniform_pytorch_1d(10000)
    print('start')
    for i in range(1, 10):
        _high = int(i * 10000)
        _width = int(i * 10000)
        _count = int(_high * _width)
        _dtype = 'float'
        # _count = 1000000
        # test on CPU
        # test_normal(_mean, _std, _count)
        # test_uniform(_mean, _std, _count)
        # test on GPU
        # test_normal_pytorch_2d(_high, _width, _dtype)
        test_uniform_pytorch_1d(_count, _dtype)
        test_normal_pytorch_1d(_count, _dtype)
        # test_uniform_pytorch_2d(_high, _width, _dtype)
        # _high *= 4
        # _width //= 4
        # test_normal_pytorch_2d(_high, _width, _dtype)
