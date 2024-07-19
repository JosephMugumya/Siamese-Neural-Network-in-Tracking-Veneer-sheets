"""
@author: Tuomas Jalonen
"""
#pip install tensorflow

import os
import json
import sys
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
parent_dir_parent_directory = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir_parent_directory)

from Utils.utils import lr_schedule, save_best_acc, TimingCallback, preprocess_img
from Utils.utils import full_vgg16_model, full_mobilenet_v2_model, create_generator, create_dataframe
from Utils.utils import plot_training_curves, plot_cm, plot_auc, plot_tsne, split_metrics, cv_metrics

base_dir = r'E:\Downloads\Thesis\Datasets\src\Training'

# Parameters
epochs = 100
batch_size = 10

models = {
    'VGG16': {'input_size': (224, 224, 3), 'target_size': (224, 224), 'color_mode': 'rgb', 'number_of_blocks': '0'},
    # 'mobilenet_v2': {'input_size': (224, 224, 3), 'target_size': (224, 224), 'color_mode': 'rgb', 'number_of_blocks': '0'},
    }

# loop models
for m in models.keys(): # pylint: disable=consider-using-dict-items
    print('Model', m, 'started')
    model_dir = os.path.join(base_dir, m)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Set lists for metrics
    accuracies, precisions, tprs, fprs, fnrs, f1s, training_times = [], [], [], [], [], [], []
    
    # Model parameters
    input_size = (models[m]['input_size'])
    target_size = (models[m]['target_size'])
    color_mode = models[m]['color_mode']
    number_of_blocks = int(models[m]['number_of_blocks'])

    # Loop splits
    for i in range(5):

        print('Fold', i, 'started')
        tf.keras.backend.clear_session()

        split_dir = os.path.join(model_dir, 'Split{}'.format(i))
        if not os.path.exists(split_dir):
            os.mkdir(split_dir)

        train_dir_wet =r'E:\Downloads\Thesis\Datasets\src\data_cross_new\Split{}\Train\Wet'.format(i)
        train_dir_dry = r'E:\Downloads\Thesis\Datasets\src\data_cross_new\Split{}\Train\Dry'.format(i)
        test_dir_wet = r'E:\Downloads\Thesis\Datasets\src\data_cross_new\Split{}\Test\Wet'.format(i)
        test_dir_dry = r'E:\Downloads\Thesis\Datasets\src\data_cross_new\Split{}\Test\Dry'.format(i)

        train_df = create_dataframe(train_dir_wet, train_dir_dry)
        test_df = create_dataframe(test_dir_wet, test_dir_dry)
        # Drop 90 % of data in dev phase
        train_df = train_df.drop(train_df.sample(frac=.9).index)
        test_df = test_df.drop(test_df.sample(frac=.9).index)

        train_df.to_excel(os.path.join(split_dir, 'train_df.xlsx'))
        test_df.to_excel(os.path.join(split_dir, 'test_df.xlsx'))

        # Data generators
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_img)
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_img)

        train_gen = create_generator(generator=train_datagen,
                                     dataframe=train_df,
                                     target_size=target_size,
                                     batch_size=batch_size,
                                     color_mode=color_mode)

        test_gen = create_generator(generator=test_datagen,
                                    dataframe=test_df,
                                    target_size=target_size,
                                    batch_size=batch_size,
                                    color_mode=color_mode)

        # Get model
        if m == 'VGG16':
            model = full_vgg16_model()
        if m == 'mobilenet_v2':
            model = full_mobilenet_v2_model()
        # elif m == 'Zhou_2021':
        #     model = zhou_2021_model()
        # elif m == 'kDenseNet_BC_L100_12ch':
        #     model = kDenseNet_BC_L100_12ch_model()
        # else:
        #     model = cnn_model(input_size, number_of_blocks)



        # Training settings
        lr_scheduler = LearningRateScheduler(lr_schedule)
        acc_callback = save_best_acc(split_dir)
        timing_callback = TimingCallback()
        opt = Adam(learning_rate=lr_schedule(0))

        model.compile(optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy', Precision(), Recall(), F1Score(average='micro')])

        model.summary()

        history = model.fit(train_gen,
                validation_data=test_gen,
                epochs=epochs,
                steps_per_epoch=len(train_df)/batch_size,
                validation_steps=len(test_df)/batch_size,
                shuffle=True,
                callbacks=[lr_scheduler, acc_callback, timing_callback])

        np.save(os.path.join(split_dir, 'history.npy'), history.history)
        hist_df = pd.DataFrame(history.history)
        with open(os.path.join(split_dir, 'history.json'), 'w') as fp:
            hist_df.to_json(fp)

        plot_training_curves(split_dir, history)

        model.load_weights(os.path.join(split_dir, 'saved_model'))
        results = model.evaluate(test_gen, batch_size=batch_size)

        class_dict = test_gen.class_indices
        class_dict = dict((v,k) for k, v in class_dict.items())

        predictions = model.predict(test_gen, steps=test_gen.samples/batch_size)
        predicted_indices = np.argmax(predictions, axis=-1)
        true_indices = test_gen.classes

        cm = confusion_matrix(true_indices, predicted_indices, normalize='true')

        np.save(os.path.join(split_dir, 'predictions.npy'), predictions)
        with open(os.path.join(split_dir, 'predictions.json'), 'w') as fp:
            pd.DataFrame(predictions).to_json(fp)
       
        np.save(os.path.join(split_dir, 'predicted_indices.npy'), predicted_indices)
        with open(os.path.join(split_dir, 'predicted_indices.json'), 'w') as fp:
            pd.DataFrame(predicted_indices).to_json(fp)
        
        np.save(os.path.join(split_dir, 'true_indices.npy'), true_indices)
        with open(os.path.join(split_dir, 'true_indices.json'), 'w') as fp:
            pd.DataFrame(true_indices).to_json(fp)

        np.save(os.path.join(split_dir, 'cm.npy'), cm)
        with open(os.path.join(split_dir, 'cm.json'), 'w') as fp:
            pd.DataFrame(cm).to_json(fp)

        accuracy, precision, tpr, fpr, fnr, f1 = split_metrics(cm)

        accuracies.append(accuracy)
        precisions.append(precision)
        tprs.append(tpr)
        fprs.append(fpr)
        fnrs.append(fnr)
        f1s.append(f1)

        # Append training times only for the first split
        if i == 0:
            training_times.append(timing_callback.logs)

        df_cm = pd.DataFrame(cm, class_dict.values(), class_dict.values())
        plot_cm(split_dir, df_cm)

        fpr, tpr, thresholds = roc_curve(true_indices, predictions[:, -1])
        AUC = auc(fpr, tpr)

        plot_auc(split_dir, fpr, tpr, AUC)

        # Get misclassification filenames and save them to .npy and .csv
        misclassified_indices = np.where(np.not_equal(predicted_indices, true_indices))[0]
        all_test_filenames=np.array(test_gen.filenames)
        misclassified_filenames = all_test_filenames[misclassified_indices]
        np.save(os.path.join(split_dir, 'misclassified_filenames.npy'), misclassified_filenames)
        np.savetxt(os.path.join(split_dir, 'misclassified_filenames.csv'), misclassified_filenames, delimiter='  ', comments='', fmt='%s')

    metrics = cv_metrics(accuracies, precisions, tprs, fprs, fnrs, f1s, training_times)
    print(metrics)
    with open(os.path.join(model_dir, 'cross-validation_metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=4)

