#Tutorials followed- Aladin Persson on Youtube
import torch
import numpy as np

#Initializing Tensor
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

#we can also specify type of the tensor in the above
#and even the device if you have a cuda-enabled GPU by 
#specifying it as a parameter as device = 'cuda'

#OR by just doing the following

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device = device, requires_grad=True)


#print(my_tensor)
'''print(my_tensor.device) #printing tensor device
print(my_tensor.dtype) #print tensor type
print(my_tensor.shape) #print tensor shape
print(my_tensor.requires_grad)#check if it requires gradient or not'''


#Other Initialization Methods
x = torch.empty(size = (3,3)) 
x = torch.empty((3,3))
x = torch.zeros(3,3) #3 x 3 matrix filled with 0's
x = torch.rand((3,3)) ##3 x 3 matrix with values from a uniform distribution in the interval 0-1
x = torch.ones((3,3)) #3 x 3 matrix filled with ones
x = torch.eye(5,5) # identity matrix, I = eye
x = torch.arange(start=0, end=5, step = 1) #similar to for (int i =i; i <5; i++)
x = torch.linspace(start = 0.1, end=1, steps=10) # computes start + ([end - start]/ [steps - 1])
x = torch.empty(size=(1, 5)).normal_(mean=0, std=2)
x = torch.diag(torch.ones(3)) # same as creating 3 x 3 diagonal matrix, an I matrix
#print(x)

#initilizaing tensors to different types
tensor = torch.arange(5)
tensor_boolean = tensor.bool() #convert to boolean
tensor_short = tensor.short() #to int16
tensor_long = tensor.long() #to int64
tensor_half = tensor.half() #to float16
tensor_float = tensor.float() #to float32 
tensor_double = tensor.double() #to float

#Array to tensors convertions and vice-versa
np_array = np.zeros((5,5 ))
np_to_tensor = torch.from_numpy(np_array)

#Converting back to numpy
np_array_back = np_to_tensor.numpy()

#Tensor Math & Comparison Operations
x = torch.tensor([1, 2, 3])
y = torch.tensor ([9, 8, 7])

#Addition
addition = torch.add(x, y) #can still be done with the method below

z1 = torch.empty(3)
torch.add(x, y, out= z1) # or just simply use the following 

z = x + y

#Substration
sub = x - y

#Division
div = torch.true_divide(x, y) #element-wise division if they are of thesame shape

#Inplace Operation
#Wheenever an operation is followed by a leading underscore "_", it means the operation is inplace - meaning doesn't create an extra copy
t = torch.zeros(3)
t.add_(x)

t += x #another popular example of inplace - but t = t + x is not inplace

#Exponentiation
z = x.pow(2) # or x ** 2

#Simple Comparison
z = x > 0
z = x < 0 

#Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand(5, 3)
x3 = torch.mm(x1, x2)

#Matrix Exponentiation
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3)

#Element-wise multiplication
z = x * y

#dot product
z = torch.dot(x, y)

#Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))
z = x1 - x2
#Ideally, the above wouldn't have been possible, but broadcasting makes it possible for the vector to have 
#same rows as the matrix which inturn makes the subtraction or any other operations possible

#Orther useful tensor operations
x = torch.tensor([[1,2,3], [4, 5, 6], [7, 8, 9]])
sum_x = torch.sum(x, dim= 1) # sum along horizontal axis
sum_x = torch.sum(x, dim= 0) # sum along vertical axis
values, indices = torch.max(x, dim=0) #max values and position. You could also do x.max(dim =0)
values, indices = torch.min(x, dim=0) #min values and position
abs_x = torch.abs(x)

z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)

x = torch.tensor([-1, 2, 15])
y = torch.tensor ([9, 8, 7])
z = torch.eq(x, y) #element-wise comparison -> is x == y?
sorted_y, indices = torch.sort(y, dim=0, descending=False) # returns the ascending order and the indices
z = torch.clamp(x, min = 0, max = 10) # set values of x that are < 0 to 0 and values of x > 10 to 10

#Tensor Indexing
batch_size = 10
features  = 25
x = torch.rand((batch_size, features))
#print(x[0, :].shape) # accessing features
#print(x[:, 0].shape) #accessing batches

#print(x[2, 0:10])#third example in the batch and the first 10 features

#Fancy Indexing
x = torch.arange(10)
indices = [2, 5, 8]
#print(x[indices])

x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
rows = torch.tensor([1, 0])
cols = torch.tensor([2, 0])
#print(x[rows, cols]) #the author made mistake in the real explanation of this

#More advanced indexing
x = torch.arange(10)
#print(x[(x < 2) | (x > 8)]) #prints our elements less than or elemnts greater than 8
#print(x[x.remainder(2) == 0]) #prints  numbers, such that when divided by 2, the remainder is 0

#useful operations
#print(torch.where(x > 5, x, x * 2)) #applies the last condition if the first condition is true. else, do nothing
#print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique()) #prints out the unique value  - (without repition)
#print(x.ndimension()) #gets the dimension
#print(x.numel()) #counts the number of elements in x


#Reshaping Tensors
x = torch.arange(9)
x_3x3 = x.view(3, 3)
x_3x3 = x.reshape(3,3) # view and reshape are somewhat similar

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
#print(torch.cat((x1, x2), dim=1).shape)
#print(torch.cat((x1, x2), dim=0).shape)

z = x1.view(-1) #flatten x1 -> merge all elements and turn it into a vector
#print(z.shape)

#similar example of flattening
batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1) #keep the batch and merge the other two
#print(z.shape)


















