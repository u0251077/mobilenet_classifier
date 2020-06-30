from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)
import numpy as np
import keras
from keras.preprocessing import image
from keras.models import load_model

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    
    img_tensor = np.expand_dims(img_tensor, axis=0)  
    img_tensor /= 255.                                    

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

model=load_model("./model/model00000025.h5")

img_path = './test_img/orange_test.jpeg'
new_image = load_image(img_path)
pred = model.predict(new_image)
print(pred)

if pred[0][0]>pred[0][1]:
 print("apple")
else:
 print("orange")

# --- if class > 2 use this to select the best socre prediction class
# pred = np.argmax(pred[0])
# --- output is 0~class_numbers
