import torch


def multilabelsimilarityloss(labels_batchsize, labels_train, hashrepresentations_batchsize,
                                hashrepresentations__train):
    batch_size = labels_batchsize.shape[0]
    num_train = labels_train.shape[0]
    labels_batchsize = labels_batchsize / torch.sqrt(torch.sum(torch.pow(labels_batchsize, 2), 1)).unsqueeze(1)
    labels_train = labels_train / torch.sqrt(torch.sum(torch.pow(labels_train, 2), 1)).unsqueeze(1)
    hashrepresentations_batchsize = hashrepresentations_batchsize / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations_batchsize, 2), 1)).unsqueeze(1)
    hashrepresentations__train = hashrepresentations__train / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations__train, 2), 1)).unsqueeze(1)
    labelsSimilarity = torch.matmul(labels_batchsize, labels_train.t())  # [0,1]
    hashrepresentationsSimilarity = torch.relu(
        torch.matmul(hashrepresentations_batchsize, hashrepresentations__train.t()))  # [0,1]
    MSEloss = torch.sum(torch.pow(hashrepresentationsSimilarity - labelsSimilarity, 2)) / (num_train * batch_size)

    return MSEloss
