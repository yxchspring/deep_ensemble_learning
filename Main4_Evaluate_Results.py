from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.utils import to_categorical

######################### Step1:  Initialize the parameters and file paths #########################
# configure the parameters
batch_size = 1
num_classes = 6
epochs = 50
image_height = 224
image_width = 224

# Set the corresponding file paths
model_folder = 'Action_ResNet50_NCNN_Model'
# Configure the train, val, and test p
base_dir = './Action_ForCNN'
test_dir = os.path.join(base_dir, 'test')

######################### Step2:  Obtain the test dataflow #########################
datagen = ImageDataGenerator(
    samplewise_center=True,
    # rescale=1./255
)
def extract_features(directory, sample_count,Channels):
    features = np.zeros(shape=(sample_count, image_height, image_width, Channels),dtype=np.float32)
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)
    i = 0
    for inputs_batch, labels_batch in generator:
        # features_batch = model.predict(inputs_batch)
        features_batch = inputs_batch
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

test_features, test_labels = extract_features(test_dir, 120, 3)
test_labels = to_categorical(test_labels)

######################### Step3:  Load the best model trained before and evaluate its performance #########################
# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), model_folder, 'checkpoints')
# load the best model
# If the error appears 'Error in loading the saved optimizer', it doesn't matter!
best_model = load_model(os.path.join(checkpoint_dir, "weights-best.hdf5"))

######################### Step3.1:  Obtain the loss and acc
# Score trained best_model.
print('The evaluation starts!\n')

scores = best_model.evaluate(
test_features,
test_labels
)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

######################### Step3.2:  Obtain the precision 、recall 、f1-score
# Obtain the prediction
pred_prob = best_model.predict(
test_features
)

# The index for prediction of testing set
pred_labels = np.argmax(pred_prob, axis=1)
# The scores for prediction of testing set
pred_scores = np.amax(pred_prob, axis=1)

# The true labels of testing set
true_labels = np.argmax(test_labels, axis=1)

# confusion matrix
cfm = confusion_matrix(true_labels, pred_labels)
print(classification_report(true_labels, pred_labels, digits=4))









