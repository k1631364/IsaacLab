import torch

# Example tensor
# tensor = torch.randn(30, 20, 5)
tensor = torch.randint(low=0, high=10, size=(30, 20, 5))

# For demonstration, manually set some values in dim2=4 to 1
tensor[0, 5, 4] = 1
tensor[1, 10, 4] = 1
tensor[2, 15, 4] = 1

# Step 1: Find the first occurrence of 1 in dim2=4 for each sequence
first_one_idx = (tensor[:,:,4] == 1).int().argmax(dim=1)
print(first_one_idx)

# Step 2: Create a new tensor filled with zeros to store the modified sequences
modified_tensor = torch.zeros_like(tensor)

# # Step 3: Iterate over each sequence and cut and pad accordingly
# for i in range(tensor.size(0)):
#     idx = first_one_idx[i].item()
#     # Copy the part of the sequence after and including the position of 1
#     modified_tensor[i, :tensor.size(1) - idx, :] = tensor[i, idx:, :]

for i in range(tensor.size(0)):
    idx = first_one_idx[i].item()
    if idx > 0 or tensor[i, idx, 4] == 1:  # Ensure 1 is found
        length_to_copy = idx + 1
        start_idx = tensor.size(1) - length_to_copy
        modified_tensor[i, start_idx:, :] = tensor[i, :length_to_copy, :]

# The modified_tensor now contains the sequences where the first part is cut after the occurrence of 1, and the rest is padded with zeros

# Check the result for the first few sequences
print(tensor[0])
print(modified_tensor[0])
# print(modified_tensor[1])
# print(modified_tensor[2])