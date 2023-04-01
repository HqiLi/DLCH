import torch


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def mean_average_precision(query_code, database_code, query_labels, database_labels, device, topk=None,):
    num_query = query_labels.shape[0]
    mean_AP = 0.0
    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()
        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())
        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]
        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()
        # Can not retrieve images
        if retrieval_cnt == 0:
            continue
        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt)
        if device:
            score = torch.linspace(1, retrieval_cnt, retrieval_cnt).cuda()
        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()
        mean_AP += (score / index).mean()
    mean_AP = mean_AP / num_query
    return mean_AP


