import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import time

def normal_module(_high, _width, device='cpu'):
    time1 = time.time()
    with tf.device('/{}:0'.format(device)):
        a = tf.random_normal((_high, _width), dtype=tf.float32)
    time2 = time.time()
    print(a)
    print('Time cost with 2D B-NORMAL distribution on {}, with {} * {} size, -->> {}'.format(device, _high, _width, time2 - time1))
    del a

def uniform_module(_high, _width, device='cpu'):
    time1 = time.time()
    with tf.device('/{}:0'.format(device)):
        a = tf.random_uniform((_high, _width), dtype=tf.float32)
    time2 = time.time()
    print(a)
    print('Time cost with 2D A-UNIFORM distribution on {}, with {} * {} size, -->> {}'.format(device, _high, _width, time2 - time1))
    del a
    

if __name__ == '__main__':
    '''
    for i in range(1, 10):
        _high = int(i * 10000)
        _width = int(i * 10000)
        _count = int(_high * _width)
        # uniform_module(_high, _width, 'cpu')
        # uniform_module(_high, _width, 'gpu')
        # normal_module(_high, _width, 'cpu')
        normal_module(_high, _width, 'gpu')
    normal_module(10000, 10000, 'gpu')
    normal_module(10000, 10000, 'gpu')
    normal_module(10000, 10000, 'gpu')
    normal_module(10000, 10000, 'gpu')
    normal_module(10000, 10000, 'gpu')
    normal_module(10000, 10000, 'gpu')
    '''
    uniform_module(10000, 10000, 'cpu')
    uniform_module(10000, 10000, 'cpu')
    uniform_module(10000, 10000, 'cpu')
    

