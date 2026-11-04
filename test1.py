import torch

A = torch.tensor([
    [1,1,1],
    [1,1,1],
    [1,1,1]
], dtype = torch.float)
A = A.masked_fill_((torch.tril(A, diagonal = 0) == 0), float("-inf"))
A = torch.nn.functional.softmax(A)
print(A)