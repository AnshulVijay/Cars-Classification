
from keras.models import Model
from layers import Flatten, Dense, Dropout
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics

TRAIN_DATASET_PATH  = 'Desktop/train1'
VALIDATE_DATASET_PATH = 'Desktop/test'
IMAGE_SIZE    = (150, 150)
NUM_CLASSES   = 3
BATCH_SIZE    = 8  
FREEZE_LAYERS = 2  
NUM_EPOCHS    = 10
WEIGHTS_FINAL = 'modelresnet50.h5'


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,channel_shift_range=10, horizontal_flip=True, fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(TRAIN_DATASET_PATH, target_size=IMAGE_SIZE, interpolation='bicubic', class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_batches = valid_datagen.flow_from_directory(VALIDATE_DATASET_PATH, target_size=IMAGE_SIZE, interpolation='bicubic', class_mode='categorical', shuffle=False, batch_size=BATCH_SIZE)


for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

    
classify = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
classify_final = Model(inputs=net.input, outputs=output_layer)
for layer in classify_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in classify_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True
classify_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])


classify_final.fit_generator(train_batches, steps_per_epoch = train_batches.samples // BATCH_SIZE, validation_data = valid_batches, validation_steps = valid_batches.samples // BATCH_SIZE, epochs = NUM_EPOCHS)

Y_pred = classify_final.predict_generator(valid_batches, steps=5)
y_pred = np.argmax(Y_pred,axis=1)
print(classify_final.summary())
net_final.save(WEIGHTS_FINAL)
