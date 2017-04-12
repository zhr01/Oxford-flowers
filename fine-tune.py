from keras import applications, utils, callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense

from dataset import load_data, split_dataset

top_model_weights_path = r'D:\Contest\practice\Oxford-flowers\datasets\weights-best.hdf5'
# dimensions of our images.
img_width, img_height = 224, 224
num_classes = 102
epochs = 100
batch_size = 16

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_weights = base_model.get_weights()
print('Model loaded.')



top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(1024, activation='relu'))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

top_model.load_weights(top_model_weights_path)
top_weights = top_model.get_weights()
# build a classifier model to put on top of the convolutional model
top_model = base_model.output
top_model = Flatten()(top_model)
top_model = Dense(1024, activation='relu')(top_model)
top_model = Dense(512, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(num_classes, activation='softmax')(top_model)


model = Model(inputs=base_model.input, outputs=top_model)
model.set_weights(base_weights+top_weights)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.rmsprop(lr=1e-4),
                  metrics=['accuracy'])

x_train, y_train = load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
# Convert class vectors to binary class matrices.
y_train = y_train.reshape((len(y_train), 1))
y_train -= 1
y_train = utils.to_categorical(y_train, num_classes)

(x_train, y_train), (x_validation, y_validation) = split_dataset(x_train, y_train, 0.8)
# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow(x_validation, y_validation, batch_size=batch_size)

reduce_leanring_rate = callbacks.ReduceLROnPlateau(monitor="val_acc", verbose=1)
tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_images=True)
early_stop = callbacks.EarlyStopping(monitor="val_acc", patience=10, verbose=1)
check_point = callbacks.ModelCheckpoint('./logs/full-weights-best.hdf5',
                                        monitor='val_acc', verbose=1,save_best_only=True)
monitors = [reduce_leanring_rate, tensorboard, early_stop, check_point]

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=x_train.shape[0]//batch_size,
    epochs=epochs,
    callbacks=monitors,
    validation_data=validation_generator,
    validation_steps=x_validation.shape[0]//batch_size
)