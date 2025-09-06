from PIL import Image

path = "D:/DeepfakeGuard/data_split/test/fake/000_003_1.jpg"
img = Image.open(path)
print(img.format, img.size, img.mode)
