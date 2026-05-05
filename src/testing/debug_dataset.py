import matplotlib.pyplot as plt
from dataset import DWIDataset2D

dataset = DWIDataset2D(mode="train")

x_noisy, x_clean, tensor, _, _ = dataset[0]

# show one diffusion channel
plt.imshow(x_clean[0], cmap='gray')
plt.title("DWI channel")
plt.show()

# show tensor components
for i in range(6):
    plt.imshow(tensor[i], cmap='gray')
    plt.title(f"Tensor component {i}")
    plt.show()
