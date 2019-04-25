# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
import torch

x: torch.Tensor = torch.rand(5, 3)

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!