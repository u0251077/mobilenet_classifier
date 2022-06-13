
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)
import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D, Dropout
from keras.applications import MobileNet
from dlm.vgg19 import VGG19
from dlm.imagenet_utils import preprocess_input
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
mobile = keras.applications.mobilenet.MobileNet()

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

#base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
base_model = VGG19(weights='imagenet', include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(2048,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='softmax')(x) #final layer with softmax activation
model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers:
    layer.trainable=False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
eval_datagen= ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator=train_datagen.flow_from_directory('./datasets/apple2orange_train',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

eval_generator=eval_datagen.flow_from_directory('./datasets/apple2orange_eval',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
step_size_eval=eval_generator.n//eval_generator.batch_size
checkpoint = keras.callbacks.ModelCheckpoint('./model/model{epoch:08d}.h5', period=1) 

model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data=eval_generator,
                   validation_steps=step_size_train,
                   epochs=25,
                   workers=12,verbose=2,callbacks=[checkpoint])
  

