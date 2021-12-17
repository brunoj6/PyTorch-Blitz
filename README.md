# PyTorch-Blitz
Useful notes from the 60 minute blitz documentation


## Tensors

Raw to Tensor
```
data = [[2, 2], [3, 1]]
t_data = torch.tensor(data)
```

Np Array to Tensor
```
arr = np.array(data)
t_np = torch.from_numpy(arr)
```

Tensor Shape 
```
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```
