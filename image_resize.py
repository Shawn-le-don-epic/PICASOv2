from PIL import Image
image = Image.open('OPimg2.png')
print(f"Current size : {image.size}")
resized_image = image.resize((1500, 2249))
resized_image.save('OPimg2-resized.png')