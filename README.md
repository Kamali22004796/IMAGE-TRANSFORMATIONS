# IMAGE-TRANSFORMATIONS

## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import numpy module as np and pandas as pd.
## Step2:

Assign the values to variables in the program.
## Step3:

Get the values from the user appropriately.
## Step4:

Continue the program by implementing the codes of required topics.
## Step5:

Thus the program is executed in google colab.


## Program:
```python
Developed By:kamali E
Register Number:212222110015
```
i)Image Translation
```
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("rajini.jpg")

# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Disable x & y axis
plt.axis('off')

# Show the image
plt.imshow(input_image)
plt.show()

# Get the image shape
rows, cols, dim = input_image.shape

# Transformation matrix for translation
M = np.float32([[1, 0, 100],
                [0, 1, 200],
                [0, 0, 1]])  # Fixed the missing '0' and added correct dimensions

# Apply a perspective transformation to the image
translated_image = cv2.warpPerspective(input_image, M, (cols, rows))

# Disable x & y axis
plt.axis('off')

# Show the resulting image
plt.imshow(translated_image)
plt.show()
```

ii) Image Scaling
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'rajini.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define scale factors
scale_x = 1.5  # Scaling factor along x-axis
scale_y = 1.5  # Scaling factor along y-axis

# Apply scaling to the image
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Display original and scaled images
print("Original Image:")
show_image(image)
print("Scaled Image:")
show_image(scaled_image)
```


iii)Image shearing
```
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'rajini.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define shear parameters
shear_factor_x = 0.5  # Shear factor along x-axis
shear_factor_y = 0.2  # Shear factor along y-axis

# Define shear matrix
shear_matrix = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])

# Apply shear to the image
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

# Display original and sheared images
print("Original Image:")
show_image(image)
print("Sheared Image:")
show_image(sheared_image)
```


iv)Image Reflection

```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'rajini.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Reflect the image horizontally
reflected_image_horizontal = cv2.flip(image, 1)

# Reflect the image vertically
reflected_image_vertical = cv2.flip(image, 0)

# Reflect the image both horizontally and vertically
reflected_image_both = cv2.flip(image, -1)

# Display original and reflected images
print("Original Image:")
show_image(image)
print("Reflected Horizontally:")
show_image(reflected_image_horizontal)
print("Reflected Vertically:")
show_image(reflected_image_vertical)
print("Reflected Both:")
show_image(reflected_image_both)
```


v)Image Rotation
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'rajini.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define rotation angle in degrees
angle = 45

# Get image height and width
height, width = image.shape[:2]

# Calculate rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

# Perform image rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display original and rotated images
print("Original Image:")
show_image(image)
print("Rotated Image:")
show_image(rotated_image)
```



vi)Image Cropping

```

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'rajini.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define cropping coordinates (x, y, width, height)
x = 100  # Starting x-coordinate
y = 50   # Starting y-coordinate
width = 200  # Width of the cropped region
height = 150  # Height of the cropped region

# Perform image cropping
cropped_image = image[y:y+height, x:x+width]

# Display original and cropped images
print("Original Image:")
show_image(image)
print("Cropped Image:")
show_image(cropped_image)

```


## Output:
### i)Image Translation

![318646615-3a3508f7-1f2e-4fcd-b98e-2b67c63eaa23](https://github.com/Kamali22004796/IMAGE-TRANSFORMATIONS/assets/120567837/de559df1-0e1d-4d6f-a211-7379424bc5d8)


### ii) Image Scaling

![318646648-12a942e8-0f0e-4681-a65e-f855e9eca9bf](https://github.com/Kamali22004796/IMAGE-TRANSFORMATIONS/assets/120567837/e1906c9b-b82f-4da9-85ea-5535bfabc06f)


### iii)Image shearing

![318646669-5635d702-4757-4c2d-8920-8f5594064847](https://github.com/Kamali22004796/IMAGE-TRANSFORMATIONS/assets/120567837/8f83d0f6-11ab-4b73-851f-87bbc56f5ef6)


### iv)Image Reflection

![318646770-8675e6b7-54e9-4725-9aac-981efc38efaf](https://github.com/Kamali22004796/IMAGE-TRANSFORMATIONS/assets/120567837/af704c64-c734-4a6f-a86f-34c39c14a51b)


### v)Image Rotation

![318646804-f2ce487c-c0d2-4fa2-8a7e-add41d474f09](https://github.com/Kamali22004796/IMAGE-TRANSFORMATIONS/assets/120567837/882330d6-f32d-4d59-8d8c-2e08347ada06)



### vi)Image Cropping

![318646823-27faf39e-f615-4e0e-9b72-230d30b071ec](https://github.com/Kamali22004796/IMAGE-TRANSFORMATIONS/assets/120567837/a85baca5-68da-4005-a027-b3e13bc07521)



## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
