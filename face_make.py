import matplotlib.pyplot as plt

def show_eigenfaces(model, h=112, w=92, num=10):
    plt.figure(figsize=(10, 4))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(model.eigenfaces[:, i].reshape(h, w), cmap="gray")
        plt.axis("off")
    plt.suptitle("Top Eigenfaces")
    plt.show()
