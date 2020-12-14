import module


def test_dataset():
    train, test = module.dataset.get_cifar_dataset()
    for ds in [train,test]:
        for x,y in ds.take(1):
            print(y)
            assert x.shape[0] == y.shape[0]