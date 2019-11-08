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
model_folder_VGG16 = 'Action_VGG16_Model'
model_folder_VGG19 = 'Action_VGG19_Model'
model_folder_ResNet50 = 'Action_ResNet50_Model'
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
checkpoint_dir_VGG16 = os.path.join(os.getcwd(), model_folder_VGG16, 'checkpoints')
# load the best model
# If the error appears 'Error in loading the saved optimizer', it doesn't matter!
best_model_VGG16 = load_model(os.path.join(checkpoint_dir_VGG16, "weights-best.hdf5"))

# checkpoint
checkpoint_dir_VGG19 = os.path.join(os.getcwd(), model_folder_VGG19, 'checkpoints')
# load the best model
# If the error appears 'Error in loading the saved optimizer', it doesn't matter!
best_model_VGG19 = load_model(os.path.join(checkpoint_dir_VGG19, "weights-best.hdf5"))

# checkpoint
checkpoint_dir_ResNet50 = os.path.join(os.getcwd(), model_folder_ResNet50, 'checkpoints')
# load the best model
# If the error appears 'Error in loading the saved optimizer', it doesn't matter!
best_model_ResNet50= load_model(os.path.join(checkpoint_dir_ResNet50, "weights-best.hdf5"))

######################### Step3.1:  Obtain the precision 、recall 、f1-score
# Obtain the prediction
print('The prediction starts!\n')
pred_prob_VGG16 = best_model_VGG16.predict(test_features)
pred_prob_VGG19 = best_model_VGG19.predict(test_features)
pred_prob_ResNet50 = best_model_ResNet50.predict(test_features)

print('The max voting prediction starts!\n')
pred_labels_VGG16 = np.argmax(pred_prob_VGG16, axis=1)
pred_labels_VGG19 = np.argmax(pred_prob_VGG19, axis=1)
pred_labels_ResNet50= np.argmax(pred_prob_ResNet50, axis=1)

# majority voting
pred_labels_maj = np.empty((len(test_features)),dtype=np.int32)
for i in range(0,len(test_features)):
    list_each = list([pred_labels_VGG16[i], pred_labels_VGG19[i], pred_labels_ResNet50[i]])
    counter_each_classes = np.array([list_each.count(0), list_each.count(1), list_each.count(2), list_each.count(3),
                                     list_each.count(4), list_each.count(5)])
    pred_labels_maj[i] = np.argmax(counter_each_classes)

# The true labels of testing set
true_labels = np.argmax(test_labels, axis=1)

cfm_maj = confusion_matrix(true_labels, pred_labels_maj)
print(classification_report(true_labels, pred_labels_maj, digits=4))

print('The Averaging voting prediction starts!\n')
# Averaging voting
pred_prob_avg = (pred_prob_VGG16 + pred_prob_VGG19 + pred_prob_ResNet50)/3
# pred_prob_mean = pred_prob_VGG16*0.4 + pred_prob_VGG19*0.3 + pred_prob_ResNet50*0.3
# The index for prediction of testing set
pred_labels_avg = np.argmax(pred_prob_avg, axis=1)
# The scores for prediction of testing set
pred_scores_avg = np.amax(pred_prob_avg, axis=1)

# The true labels of testing set
true_labels = np.argmax(test_labels, axis=1)

cfm_avg = confusion_matrix(true_labels, pred_labels_avg)
print(classification_report(true_labels, pred_labels_avg, digits=4))

print('The Weighted average voting prediction starts!\n')
import pandas as pd
np.random.seed(123)
df = pd.DataFrame(columns=('w1', 'w2', 'w3', 'acc'))
i = 0
for w1 in range(1,6):
    for w2 in range(1,6):
        for w3 in range(1,6):
            if len(set((w1,w2,w3))) == 1: # skip if all weights are equal
                continue
            pred_prob_weighted = (pred_prob_VGG16*w1 + pred_prob_VGG19*w1 + pred_prob_ResNet50*w1) / (w1+w2+w3)
            pred_labels_weighted = np.argmax(pred_prob_weighted, axis=1)
            acc_number =  sum(true_labels==pred_labels_weighted)
            acc = acc_number/len(true_labels)
            df.loc[i] = [w1, w2, w3, acc]
            i += 1

df.sort_values(by=['acc'], ascending=False)

# Averaging voting
w1_new = df.iloc[0,0]
w2_new = df.iloc[0,1]
w3_new = df.iloc[0,2]
pred_prob_weighted = (pred_prob_VGG16*w1_new  + pred_prob_VGG19*w2_new  + pred_prob_ResNet50*w3_new )/(w1_new+w2_new+w3_new )
# pred_prob_mean = pred_prob_VGG16*0.4 + pred_prob_VGG19*0.3 + pred_prob_ResNet50*0.3
# The index for prediction of testing set
pred_labels_weighted = np.argmax(pred_prob_weighted, axis=1)
# The scores for prediction of testing set
pred_scores_weighted = np.amax(pred_prob_weighted, axis=1)

# The true labels of testing set
true_labels = np.argmax(test_labels, axis=1)

cfm_weighted = confusion_matrix(true_labels, pred_labels_weighted)
print(classification_report(true_labels, pred_labels_weighted, digits=4))





