import keras
import numpy as np
from keras import applications
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

from dataset import load_data

batch_size = 32
num_classes = 102
epochs = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 224, 224
# The CIFAR10 images are RGB.
img_channels = 3
augmentation_times = 3
top_model_weights_path = 'datasets/bottleneck_fc_model.h5'

def save_bottlebeck_features():
    x_train, y_train = load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')

    # Convert class vectors to binary class matrices.
    y_train = y_train.reshape((len(y_train), 1))
    y_train -= 1
    y_train = keras.utils.to_categorical(y_train, num_classes)

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)

    model = applications.VGG16(include_top=False, weights='imagenet')
    # 一定要指定shuffle的值
    generator = datagen.flow(x_train, y_train, batch_size=1, shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, augmentation_times*x_train.shape[0], verbose=1)
    np.save(open('datasets/bottleneck_features_train.npy', 'wb+'), bottleneck_features_train)
    np.save(open("datasets/bottleneck_features_labels.npy", 'wb+'), np.array(y_train.tolist()*augmentation_times))

def train_top_model():
    train_data = np.load(open('datasets/bottleneck_features_train.npy', 'rb+'))
    train_labels = np.load(open("datasets/bottleneck_features_labels.npy", 'rb+'))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    reduce_leanring_rate = callbacks.ReduceLROnPlateau(monitor="acc", verbose=1)
    tensorboard = callbacks.TensorBoard(log_dir='./logs', write_images=True)
    early_stop = callbacks.EarlyStopping(monitor="acc", patience=10, verbose=1)
    check_point = callbacks.ModelCheckpoint('./logs/weights-best.hdf5',
                                            monitor='acc', verbose=1,save_best_only=True)
    model.fit(x=train_data, y=train_labels,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[reduce_leanring_rate, tensorboard, early_stop, check_point],
              validation_split=0.1)
    model.save_weights(top_model_weights_path)



# save_bottlebeck_features()
train_top_model()