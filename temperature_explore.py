
import torch
import torch.nn.functional as F

x_1 = torch.randn(1, 51257)
y_true = torch.LongTensor([0] * 1)

# def entropy_of_true(
#     logits: torch.FloatTensor, 
#     y_true: torch.LongTensor, 
#     temperature = 1.0,
# ):
#     import pdb; pdb.set_trace()
#     logits = logits / temperature
#     logits = F.softmax(logits, dim=-1)
#     N = logits.size(0)
#     entropy = logits[torch.arange(N), y_true]
#     entropy = -torch.log(entropy) * entropy
#     entropy = entropy.mean()
#     return entropy
# entropy = entropy_of_true(x_1, y_true)
# print(entropy.item())

cross_entropy = F.cross_entropy(x_1, y_true)
print(cross_entropy.item())
import pdb; pdb.set_trace()

