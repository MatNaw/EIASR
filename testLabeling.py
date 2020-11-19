import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

from csv import reader

trainPath = './train_images'
testPath = './test_images'
valPath = './val_images'
labelPath = './labels.csv'

imageName = '09-february-2017-ufa-russia-260nw-1062076784.jpg'
chosenImage = trainPath + '/' + imageName

im = np.array(Image.open(chosenImage), dtype=np.uint8)
print('SHAPE')
print(im.shape)
with open(labelPath, 'r') as read_obj:
    csv_reader = reader(read_obj)
    labels = list(csv_reader)
    
#delete header
labels = labels[1:] 
#every second row is empty... don't ask why... so I delete it
del labels[::2] 

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

# label = [name, typ, xmin, xmax, ymin, ymax]
# method below supports showing all drone labels from image
for label in labels: 
    if label[0] == imageName:
        print('Got you!')
        print(label[0])
        # Create a Rectangle patch
        rect = patches.Rectangle((int(label[2]),int(label[4])),int(label[3])-int(label[2]),int(label[5])-int(label[4]),linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

plt.show()





