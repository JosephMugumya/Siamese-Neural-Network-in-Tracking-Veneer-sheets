"""
This file is used for building the Siamese network model.
"""

# Import packages
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Lambda, Dropout
from tensorflow.keras.applications import VGG16, mobilenet_v2

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sn
from natsort import natsorted, ns

def dot_product(vecs, normalize=False):
    """
    This function returns dot product of the input vectors.

    Parameters
    ----------
    vecs : Flattened output vectors of VGG16 feature extractor
    normalize : The default is False.

    Returns
    -------
    Dot product of the vectors

    """
    vec_x, vec_y = vecs

    # if normalize:
    #     vec_x = K.l2_normalize(vec_x, axis=0)
    #     vec_y = K.l2_normalize(vec_x, axis=0)

    return K.prod(K.stack([vec_x, vec_y], axis=1), axis=1)

# This defines the similarity model.
# The inputs are extracted features from VGG16 and outputs probabilities for
# Pair and Not-pair
def similarity_model(vector_a, vector_b):
    """
    This function defines the similarity model by taking outputs from feature
    extraction models as inputs, merges them with lambda layer, adds
    fully-connected layers and finally outputs probabilities for input images
    being pair and not being pair.

    Parameters
    ----------
    vector_A : Output of VGG16 with image A
    vector_B : Output of VGG16 with image B

    Returns
    -------
    pred : probabilities for Pair and Not-pair

    """

    merged = Lambda(dot_product,
                    output_shape=vector_a[0])([vector_a, vector_b])

    # fc = Dense(512, kernel_initializer='he_normal')(merged)
    # fc = Dropout(0.2)(fc)
    # fc = Activation("relu")(fc)

    # fc = Dense(128, kernel_initializer='he_normal')(fc)
    fc = Dense(128, kernel_initializer='he_normal')(merged)
    fc = Dropout(0.1)(fc)
    fc = Activation("relu")(fc)

    fc = Dense(10, kernel_initializer='he_normal')(fc)
    fc = Activation("relu")(fc)

    pred = Dense(1, kernel_initializer='normal')(fc)
    pred = Activation("softmax")(pred)

    return pred

def full_vgg16_model():
    # Define feature extraction models for both inputs. Here we use VGG16 with
    # pretrained weights
    model_1 = VGG16(weights='imagenet', include_top=True)
    model_2 = VGG16(weights='imagenet', include_top=True)

    # Let's freeze the VGG16 layers and give them unique names
    for layer in model_1.layers:
        layer.trainable = False
        layer._name = layer._name + "_1"
    for layer in model_2.layers:
        layer.trainable = False
        layer._name = layer._name + "_2"

    # Get outputs from Flatten layers
    v1 = model_1.get_layer("flatten_1").output
    v2 = model_2.get_layer("flatten_2").output

    # Now we can compile the whole model
    preds = similarity_model(v1, v2)
    model = Model(inputs=[model_1.input, model_2.input], outputs=preds)

    return model

def full_mobilenet_v2_model():
    # Define feature extraction models for both inputs. Here we use VGG16 with
    # pretrained weights
    model_1 = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=True)
    model_2 = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=True)

    # Let's freeze the layers and give them unique names
    for layer in model_1.layers:
        layer.trainable = False
        layer._name = layer._name + "_1"
    for layer in model_2.layers:
        layer.trainable = False
        layer._name = layer._name + "_2"

    # Get outputs from Flatten layers
    v1 = model_1.get_layer('global_average_pooling2d_1').output
    v2 = model_2.get_layer('global_average_pooling2d_1_2').output

    # Now we can compile the whole model
    preds = similarity_model(v1, v2)
    model = Model(inputs=[model_1.input, model_2.input], outputs=preds)

    return model

def create_generator(generator, dataframe, target_size, batch_size, color_mode):
    """

    Parameters
    ----------
    generator : base generator
    batch_size : Intended batch size

    Yields
    ------
    Image batches: batch_size * (image_A, image_B, label)

    """
    gen_x1 = generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col='files_wet',
        y_col='labels',
        target_size=target_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode='raw',
        directory=None,
        shuffle=False,
        seed=7)

    gen_x2 = generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col='files_dry',
        y_col='labels',
        target_size=target_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode='raw',
        directory=None,
        shuffle=False,
        seed=7)

    while True:
        x_1 = gen_x1.next()
        x_2 = gen_x2.next()
        yield [x_1[0], x_2[0]], x_2[1]

def create_dataframe(dir_wet, dir_dry):

    # Read image dirs to get sorted list of the images' absolute paths
    files_wet = []
    files_dry = []
    # files_dry_shuffled = []
    veneers_dry_unique_count = 0
    veneers_dry_unique = []
    for file in natsorted(os.listdir(dir_wet), alg=ns.IGNORECASE):
        if file.endswith('.png'):
            files_wet.append(os.path.join(dir_wet, file))

    for file in natsorted(os.listdir(dir_dry), alg=ns.IGNORECASE):
        if file.endswith('.png'):
            
            files_dry.append(os.path.join(dir_dry, file))
            
            # Count the number of unique veneers present
            if file.split('_')[1] not in veneers_dry_unique:
                veneers_dry_unique_count += 1
                veneers_dry_unique.append(file.split('_')[1])

    # Shuffle the dry veneers but keep the grids at the same position
    files_dry_shuffled = np.array(files_dry).reshape((veneers_dry_unique_count, -1))
    rng = np.random.default_rng(seed=7)
    files_dry_shuffled = rng.permutation(files_dry_shuffled)
    files_dry_shuffled = files_dry_shuffled.flatten().tolist()

    """Form the dataframe like this: # pylint: disable=pointless-string-statement
    path to wet             path to dry                             label
    _____________________________________________________________________
    wet veneer 1 grid 1     dry veneer 1 grid 1                         1
    wet veneer 1 grid 2     dry veneer 1 grid 2                         1
    wet veneer 1 grid 3     dry veneer 1 grid 3                         1
    ...
    wet veneer 2 grid 1     dry veneer 2 grid 1                         1
    ...             ...
    wet veneer n grid m     dry veneer n grid m                         1
    wet veneer 1 grid 1     dry veneer 123 (randomized) grid 1          0
    wet veneer 1 grid 2     dry veneer 123 (randomized) grid 2          0
    wet veneer 1 grid 3     dry veneer 123 (randomized) grid 3          0
    ...
    wet veneer 2 grid 1     dry veneer 2500 (randomized) grid 1         0
    ...             ...
    wet veneer n grid m     dry veneer 97 (randomized) grid m           0
    _____________________________________________________________________
    """
    
    labels = [1.] * len(files_wet) + [0.] * len(files_wet)
    files_wet = files_wet + files_wet
    files_dry = files_dry + files_dry_shuffled
    df = pd.DataFrame(list(zip(files_wet, files_dry, labels)),
               columns =['files_wet', 'files_dry', 'labels'])
    
    # Shuffle rows
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 120, 150 and 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    learning_rate = 1e-3
    if epoch > 180:
        learning_rate *= 1e-3
    elif epoch > 150:
        learning_rate *= 1e-2
    elif epoch > 120:
        learning_rate *= 1e-1
    print('Learning rate: ', learning_rate)
    return learning_rate

def save_best_acc(DIRNAME):

    acc_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(DIRNAME, 'saved_model'),
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)
    
    return acc_callback

class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time.time()-self.starttime)

def preprocess_img(img):
    img *= 1./255
    # img = exposure.equalize_hist(img)

    return img

def plot_training_curves(dir, history):

    # Restore Matplotlib settings
    matplotlib.rc_file_defaults()
    
    # Accuracy curve
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.savefig(os.path.join(dir, 'acc.pdf'), dpi=300)
    plt.clf()

    # Loss curve
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Categorical Crossentropy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.savefig(os.path.join(dir, 'loss.pdf'), dpi=300)
    plt.clf()

    # F1-score curve
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.ylabel('F1-score')
    plt.xlabel('Epoch')
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.savefig(os.path.join(dir, 'f1.pdf'), dpi=300)
    plt.clf()

    # Precision curve
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.savefig(os.path.join(dir, 'precision.pdf'), dpi=300)
    plt.clf()

    # Recall curve
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.savefig(os.path.join(dir, 'recall.pdf'), dpi=300)
    plt.clf()

    # Combined Accuracy and Loss curve
    fig, ax1 = plt.subplots()

    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_yticks(np.arange(0.50, 1.05, 0.05))
    ax1.set_ylim(0.50, 1.0)
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(history.history['loss'], '--')
    ax2.plot(history.history['val_loss'], '--')
    ax2.set_ylabel('Categorical Crossentropy')
    ax2.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax2.set_ylim(0.0, 1.0)

    fig.legend(['Training Accuracy', 'Testing Accuracy', 'Training Loss', 'Testing Loss'], bbox_to_anchor=(0.9, 0.5))
    plt.savefig(os.path.join(dir, 'acc_loss.pdf'), dpi=300)
    plt.clf()

    return None

def plot_cm(dir, df_cm):
    sn.set(font_scale=1.4) # for label size
    ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='YlGnBu')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.tight_layout()
    plt.savefig(os.path.join(dir, 'cm.pdf'), dpi=300)
    plt.clf()

    return None

def plot_auc(dir, fpr, tpr, auc):

    # Restore Matplotlib settings
    matplotlib.rc_file_defaults()

    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(dir, 'roc.pdf'), dpi=300)
    plt.clf()

    return None

def plot_tsne(dir, tsne, predicted_classes):
    
    # Restore Matplotlib settings
    matplotlib.rc_file_defaults()

    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    sn.scatterplot(x=tx, y=ty, hue=predicted_classes, legend='full', cmap='YlGnBu')
    plt.tight_layout()
    plt.savefig(os.path.join(dir, 't-sne.pdf'), dpi=300)
    plt.clf()

    return None

def split_metrics(cm):

    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    f1 = 2*tp / (2*tp + fp + fn)

    return accuracy, precision, tpr, fpr, fnr, f1

def cv_metrics(ACCURACIES, PRECISIONS, TPRs, FPRs, FNRs, F1s, TIMES):
    AVG_ACC = np.average(ACCURACIES)
    STD_ACC = np.std(ACCURACIES)

    AVG_PRECISIONS = np.average(PRECISIONS)
    STD_PRECISIONS = np.std(PRECISIONS)

    AVG_TPR = np.average(TPRs)
    STD_TPR = np.std(TPRs)

    AVG_FPR = np.average(FPRs)
    STD_FPR = np.std(FPRs)

    AVG_FNR = np.average(FNRs)
    STD_FNR = np.std(FNRs)

    AVG_F1 = np.average(F1s)
    STD_F1 = np.std(F1s)

    AVG_TIMES = np.average(TIMES)
    STD_TIMES = np.std(TIMES)

    METRICS = [{
    'accuracies': ACCURACIES,
    'avg_acc': AVG_ACC,
    'std_acc': STD_ACC,
    'precisions': PRECISIONS,
    'avg_precisions': AVG_PRECISIONS,
    'std_precisions': STD_PRECISIONS,
    'TPRs': TPRs,
    'avg_TPR': AVG_TPR,
    'std_TPR': STD_TPR,
    'FPRs': FPRs,
    'avg_FPR': AVG_FPR,
    'std_FPR': STD_FPR,
    'FNRs': FNRs,
    'avg_FNR': AVG_FNR,
    'std_FNR': STD_FNR,
    'F1s': F1s,
    'avg_F1': AVG_F1,
    'std_F1': STD_F1,
    'avg_TIMES': AVG_TIMES,
    'std_TIMES': STD_TIMES
    }]

    return METRICS

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


