# PyTorch-Blitz
Useful notes from the 60 minute blitz documentation

## Tensor Function Documentation

https://pytorch.org/docs/stable/torch.html

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

Attributes of a Tensor (Shape, Type, Device)
```
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

To Use GPU
```
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")
```


## Autograd

Forward Prop : NN makes guesses about correct output given some input.
Back Prop : NN updates parameters based on error of previous output. 

Load some model
```
model = torchvision.models.resnet18(pretrained=True)
```

Single Training Step
```
prediction = model(data) # forward pass
loss = (prediction - labels).sum() # calculate loss
loss.backward() # backward pass
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9) # call an optimizer
optim.step() # gradient descent
```

When Tensor Differentiation is Required
```
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
```


