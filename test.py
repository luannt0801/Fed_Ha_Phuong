 elif datasetname == "Cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]),
                                       transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465],
                                                            std=[1., 1., 1.]),
                                       ])
        trainset = torchvision.datasets.CIFAR10(root='~/data', train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='~/data', train=False,
                                               download=True, transform=transform_test)
        trainset.targets = torch.tensor(trainset.targets)
        testset.targets = torch.tensor(testset.targets)