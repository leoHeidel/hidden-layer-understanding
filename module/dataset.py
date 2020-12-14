import tensorflow_datasets as tfds


def get_cifar_dataset(batch_size=32):
    '''
    Cifar dataset with 10 classes, 60000 images, perfect for experiments.
    '''
    train = tfds.load('cifar10', split='train', shuffle_files=True, data_dir="data", as_supervised=True)
    test = tfds.load('cifar10', split='test', shuffle_files=True, data_dir="data", as_supervised=True)
    return train.cache().batch(batch_size), test.cache().batch(batch_size)