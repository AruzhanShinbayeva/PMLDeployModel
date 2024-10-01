import cv2
import os
import imghdr
from keras import utils, layers, Sequential
from tensorflow import data as tf_data
from PIL import Image

data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
types = ['happy', 'sad']
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]
scale_layer = layers.Rescaling(1.0 / 255)


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


for i in range(len(types)):
    image_class = types[i]
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)

        img = Image.open(image_path)
        icc_profile = img.info.get('icc_profile')
        if icc_profile:
            os.remove(image_path)
            continue

        try:
            image_path = os.path.join(data_dir, image_class, image)
            img_cv = cv2.imread(image_path)
            ext = imghdr.what(image_path)
            if ext not in image_exts:
                os.remove(image_path)
        except Exception as e:
            continue

train, val = utils.image_dataset_from_directory(
    data_dir,
    class_names=['happy', 'sad'],
    validation_split=0.2,
    seed=228,
    subset="both"
)

train = train.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)

input_shape = (256, 256, 3)
model = Sequential([
    layers.Input(shape=input_shape),
    scale_layer,
    layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
    layers.Dense(1, activation='sigmoid'),
])

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


epochs = 50
history = model.fit(
    train,
    validation_data=val,
    epochs=epochs,
)

model.save('models/imageclassifier.h5')
