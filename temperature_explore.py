
import matplotlib.pyplot as plt
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

temperatures = torch.linspace(0.01, 1.0, 100)  # Avoid division by zero
cross_entropies = []

for temp in temperatures:
    logits = x_1 / temp
    cross_entropy = F.cross_entropy(logits, y_true)
    cross_entropies.append(cross_entropy.item())

plt.plot(temperatures.numpy(), cross_entropies)
plt.xlabel('Temperature')
plt.ylabel('Cross-Entropy')
plt.title('Cross-Entropy vs Temperature')
plt.grid(True)
plt.show()

