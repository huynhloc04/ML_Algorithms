
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from KMeans import KMeansClustering

#   1. MNIST Classification
#   2. Object Segmentation
def plot_seg(data, labels, values, origin_img):
    data[labels == 0] = values[0]
    data[labels == 1] = values[1]
    data[labels == 2] = values[2]

    img = data.reshape(origin_img.shape)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    img_path = 'O:\My_Documents\MACHINE LEARNING\Datasets\girl.jpg'
    img = mpimg.imread(img_path)
    img_height, img_width, channels = img.shape
    data = img.reshape(-1, 3)
    no_classes = 3

    model = KMeansClustering(data, no_classes)
    values, labels = model.fit()
    plot_seg(data, labels, values, img)


#   3. Image Compression

#   We use more number of center point instead of 3 points => Enhance image quality compare to using 3 centers => Image Compression

def plot_seg(data, labels, values, origin_img):
    no_classes = len(set(labels))
    for i in range(no_classes):
        data[labels == i] = values[i]

    img = data.reshape(origin_img.shape)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    img_path = 'O:\My_Documents\MACHINE LEARNING\Datasets\girl.jpg'
    img = mpimg.imread(img_path)
    img_height, img_width, channels = img.shape
    data = img.reshape(-1, 3)
    no_classes = 20

    model = KMeansClustering(data, no_classes)
    values, labels = model.fit()
    plot_seg(data, labels, values, img)