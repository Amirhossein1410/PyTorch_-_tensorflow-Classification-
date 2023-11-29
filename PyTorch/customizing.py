import pandas as pd
import shutil
from plot_model import PlotModel
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
from sklearn.metrics import confusion_matrix , classification_report
import random
import tensorflow as tf
import  matplotlib.pylab as plt
import seaborn as sns
import os
from sklearn.metrics import f1_score
import sys
import torch
class Customize:
    def __init__(self , framework ):
        self.framework = framework

    def make_data_frames(self , train_dir , test_dir, val_dir , limiter):
        from sklearn.model_selection import train_test_split

        bad_images = []
        # check what directories exist
        if test_dir == None and val_dir == None:
            dirlist = [train_dir]
            names = ['train']
        elif test_dir == None:
            dirlist = [train_dir, val_dir]
            names = ['train', 'valid']
        elif val_dir == None:
            dirlist = [train_dir, test_dir]
            names = ['train', 'test']
        else:
            dirlist = [train_dir, test_dir, val_dir]
            names = ['train', 'test', 'valid']
        ht = 0  # set initial value of height counter
        wt = 0  # set initial value of width counter
        total_good_files = 0  # set initial value of total number of good image files counter
        zipdir = zip(names, dirlist)
        for name, d in zipdir:  # iterate through the  names and directories
            filepaths = []  # initialize list of filepaths
            labels = []  # initialize list of class labels
            classlist = sorted(os.listdir(d))  # get a list of all the classes in alphanumeric order
            for klass in classlist:  # iterate through the list of classes
                msg = f'processing images in {name} directory for class {klass}'
                print(msg, '\r', end='')
                good_file_count = 0  # initialize the good_file count for this class
                classpath = os.path.join(d, klass)  # define the full path to the class
                if os.path.isdir(classpath):  # ensure we are working with a directory and not a spurious file
                    flist = sorted(os.listdir(classpath))  # make a list of all the files for this class
                    if limiter != None:  # check if a limiter value was specified that determmine how many files to use in any class
                        if limiter < len(flist):  # if there are more files than the value of limiter than randomly sample a limiter number of files
                            flist = np.random.choice(flist, limiter, replace=False)
                    for f in flist:
                        fpath = os.path.join(classpath, f)  # create the full path to the image file
                        index = f.rfind('.')
                        ext = f[index + 1:].lower()  # the file's extension
                        if ext not in ['jpg', 'jpeg', 'tiff', 'png',
                                       'bmp']:  # make sure the file extension is one that works with Keras
                            bad_images.append(fpath)  # if not a proper extension store the filepath in the bad images list
                        else:
                            try:  # check if image files are defective if so do not include in dataframe
                                img = cv2.imread(fpath)
                                h = img.shape[0]
                                w = img.shape[1]
                                ht += h  # add images height and width to the counters
                                wt += w
                                good_file_count += 1
                                total_good_files += 1
                                filepaths.append(fpath)  # append the filepath to the list of valid filepaths
                                labels.append(klass)  # append the file's class label to the labels list

                            except:
                                bad_images.append(fpath)  # if the image file is defective add the filepath to the list of bad images
            print('')
            Fseries = pd.Series(filepaths,name='filepaths')  # make a pandas series for the filenames and labels lists
            Lseries = pd.Series(labels, name='labels')
            df = pd.concat([Fseries, Lseries], axis=1)  # make a dataframe with columns filepaths and labels
            # depending on which directory we are iterating through create dataframes
            if name == 'valid':
                valid_df = df
            elif name == 'test':
                test_df = df
            else:
                if test_dir == None and val_dir == None:  # create train_df, test_df and valid_df from df
                    pdf = df
                    train_df, dummy_df = train_test_split(pdf, train_size=.8, shuffle=True,
                                                          random_state=123, stratify=pdf['labels'])
                    valid_df, test_df = train_test_split(dummy_df, train_size=.5, shuffle=True,
                                                         random_state=123,
                                                         stratify=dummy_df['labels'])
                elif test_dir == None:  # create just a train_df and test_df
                    pdf = df
                    train_df, test_df = train_test_split(pdf, train_size=.8, shuffle=True,
                                                         random_state=123, stratify=pdf['labels'])
                elif val_dir == None:  # create a train_df and a valid_df
                    pdf = df
                    train_df, valid_df = train_test_split(pdf, train_size=.8, shuffle=True,
                                                          random_state=123, stratify=pdf['labels'])
                else:
                    train_df = df  # test and valid dataframes exists so train_df is just df
        classes = sorted(train_df['labels'].unique())
        class_count = len(classes)
        # calculate the average image height and with
        have = int(ht / total_good_files)
        wave = int(wt / total_good_files)
        aspect_ratio = have / wave
        print('number of classes in processed dataset= ', class_count)
        counts = list(train_df['labels'].value_counts())
        print('the maximum files in any class in train_df is ', max(counts),
              '  the minimum files in any class in train_df is ', min(counts))
        print('train_df length: ', len(train_df), '  test_df length: ', len(test_df),'  valid_df length: ', len(valid_df))
        if len(bad_images) == 0:
            PlotModel.print_in_color('All image files were properly processed and used in the dataframes')
        else:
            PlotModel.print_in_color(f'the are {len(bad_images)} bad image files and {total_good_files} proper image files in the dataset')
            for f in bad_images:
                print(f)
        plot_title = 'Images per Label in train set'
        PlotModel.plot_label_count(train_df, plot_title)
        return train_df, test_df, valid_df, classes, class_count, max(counts), min(counts), have, wave

    def make_gens(self , batch_size, ycol, train_df, test_df, valid_df, img_size):
        from keras.preprocessing.image import ImageDataGenerator
        gen = ImageDataGenerator()
        msg = '{0:70s} for train generator'.format(' ')
        print(msg, '\r', end='')  # prints over on the same line
        train_gen = gen.flow_from_dataframe(train_df, x_col='filepaths', y_col=ycol, target_size=img_size,
                                            class_mode='categorical', color_mode='rgb', shuffle=True,
                                            batch_size=batch_size)
        msg = '{0:70s} for valid generator'.format(' ')
        print(msg, '\r', end='')  # prints over on the same line
        valid_gen = gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col=ycol, target_size=img_size,
                                            class_mode='categorical', color_mode='rgb', shuffle=False,
                                            batch_size=batch_size)
        # for the test_gen we want to calculate the batch size and test steps such that batch_size X test_steps= number of samples in test set
        # this insures that we go through all the sample in the test set exactly once.
        length = len(test_df)
        test_batch_size = \
        sorted([int(length / n) for n in range(1, length + 1) if length % n == 0 and length / n <= 80],
               reverse=True)[0]
        test_steps = int(length / test_batch_size)
        msg = '{0:70s} for test generator'.format(' ')
        print(msg, '\r', end='')  # prints over on the same line
        test_gen = gen.flow_from_dataframe(test_df, x_col='filepaths', y_col=ycol, target_size=img_size,
                                           class_mode='categorical', color_mode='rgb', shuffle=False,
                                           batch_size=test_batch_size)
        # from the generator we can get information we will need later
        classes = list(train_gen.class_indices.keys())
        class_indices = list(train_gen.class_indices.values())
        class_count = len(classes)
        labels = test_gen.labels
        return train_gen, test_gen, valid_gen, test_steps, class_count

    def check_dataset_size(self , train_dir):
        classes = sorted(os.listdir(train_dir))
        ftotal = 0
        flargest = 0
        fsmallest = 100000000
        for klass in classes:
            classpath = os.path.join(train_dir, klass)
            if os.path.isdir(classpath):
                flist = os.listdir(classpath)
                fcount = len(flist)
                if fcount > flargest:
                    flargest = fcount
                    maxclass = klass
                if fcount < fsmallest:
                    fsmallest = fcount
                    minclass = klass
                ftotal += fcount
        return ftotal, flargest, maxclass, fsmallest, minclass

    def balance(self , df, n, column, working_dir, img_size):
        def get_augmented_image(image):  # given an image this function returns an augmented image
            width = int(image.shape[1] * .8)
            height = int(image.shape[0] * .8)
            transform = A.Compose([
                A.HorizontalFlip(p=.5),
                A.Rotate(limit=30, p=.25),
                A.RandomBrightnessContrast(p=.5),
                A.RandomGamma(p=.5),
                A.RandomCrop(width=width, height=height, p=.25) ,
                ToTensorV2()]
            )
            return transform(image=image)['image']
        def dummy(image):
            return image
        df = df.copy()
        print('Initial length of dataframe is ', len(df))
        aug_dir = os.path.join(working_dir, 'aug')  # directory to store augmented images
        if os.path.isdir(aug_dir):  # start with an empty directory
            shutil.rmtree(aug_dir)
        os.mkdir(aug_dir)
        for label in df[column].unique():
            dir_path = os.path.join(aug_dir, label)
            os.mkdir(dir_path)  # make class directories within aug directory
        # create and store the augmented images
        total = 0
        groups = df.groupby(column)  # group by class
        for label in df[column].unique():  # for every class
            msg = f'augmenting images in train set  for class {label}'
            print(msg, '\r', end='')
            group = groups.get_group(label)  # a dataframe holding only rows with the specified label
            sample_count = len(group)  # determine how many samples there are in this class
            if sample_count < n:  # if the class has less than target number of images
                aug_img_count = 0
                delta = n - sample_count  # number of augmented images to create
                target_dir = os.path.join(aug_dir, label)  # define where to write the images
                desc = f'augmenting class {label:25s}'
                for i in range(delta):
                    j = i % sample_count  # need this because we may have to go through the image list several times to get the needed number
                    img_path = group['filepaths'].iloc[j]
                    img = cv2.imread(img_path)
                    img = get_augmented_image(img)
                    fname = os.path.basename(img_path)
                    fname = 'aug' + str(i) + '-' + fname
                    dest_path = os.path.join(target_dir, fname)
                    cv2.imwrite(dest_path, img)
                    aug_img_count += 1
                total += aug_img_count
        print('')
        print('Total Augmented images created= ', total)
        # create aug_df and merge with train_df to create composite training set ndf
        aug_fpaths = []
        aug_labels = []
        classlist = sorted(os.listdir(aug_dir))
        for klass in classlist:
            classpath = os.path.join(aug_dir, klass)
            flist = sorted(os.listdir(classpath))
            for f in flist:
                fpath = os.path.join(classpath, f)
                aug_fpaths.append(fpath)
                aug_labels.append(klass)
        Fseries = pd.Series(aug_fpaths, name='filepaths')
        Lseries = pd.Series(aug_labels, name='labels')
        aug_df = pd.concat([Fseries, Lseries], axis=1)
        df = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
        print('Length of augmented dataframe is now ', len(df))
        return df

    def set_image_size(self ,have, wave):
        if have <= 224 and wave <= 224:
            img_size = (have, wave)
        else:
            if have >= wave:
                img_size = (224, int(224 * wave / have))
            else:
                img_size = (int(224 * have / wave))
        return img_size

    def auto_balance(self , train_df, img_size, max_samples, min_samples, working_dir):
        msg = 'enter the number of images you want to have in each class of the train data set'
        max_images = int(input(msg))
        msg = 'enter the minimum number of images a class must have to be included in the train data set'
        min_images = int(input(msg))
        train_df, classes, class_count = self.trim(train_df, max_images, min_images, 'labels')
        train_df = self.balance(train_df, max_images, 'labels', working_dir, img_size)
        plot_title = 'Images per Label after Auto Balance of train data set'
        PlotModel.plot_label_count(train_df, plot_title)
        return train_df

    def manual_balance(self , train_df, img_size, max_samples, min_samples, working_dir):
        msg = f' maximum images in a class is {max_samples}, minimum images in a class is {min_samples}'
        PlotModel.print_in_color(msg)
        msg = ' if you want to trim the train set so no class has more than n images\n enter the maximun number of images allowed in a class or press enter to not trim'
        ans = input(msg)
        if ans == '':
            max_images = max_samples
        else:
            max_images = int(ans)
        msg = ' if you want to eliminate classes that have less than a minimum number of images\n enter the minimum number of images a class must have to be included in the dataset or press enter to include all classes'
        ans = input(msg)
        if ans == '':
            min_images = min_samples
        else:
            min_images = int(ans)
        train_df, classes, class_count = self.trim(train_df, max_images, min_images, 'labels')
        plot_title = 'Images per Label after trimming the dataset'
        PlotModel.print_in_color(train_df ,  plot_title)
        msg = ' if you trimmed the data set it may still not be balanced or if it is balanced it may not have an adequate number of images.'
        PlotModel.print_in_color(msg)
        msg = 'if you want to balance the dataset or you want to create more images in each class enter \n the number of images you want in each class if not press enter'
        ans = input(msg)
        if ans != '':
            n = int(ans)
            train_df = self.balance(train_df, n, 'labels', working_dir, img_size)
            plot_title = 'Images per Label after manually balancing the train data set'
            PlotModel.print_in_color(train_df, plot_title)
        return train_df


    def preprocess_dataset(self , train_df, img_size, max_samples, min_samples, working_dir):
        msg = ' enter A  to auto balance the train set or enter \n M to manually balance or hit enter to leave train set unchanged'
        ans = input(msg)
        if ans == 'A' or ans == 'a':
            train_df = self.auto_balance(train_df, img_size, max_samples, min_samples, working_dir)
        elif ans == 'M' or ans == 'm':
            train_df = self.manual_balance(train_df, img_size, max_samples, min_samples, working_dir)
        else:
            msg = f'training data set will be used as is '
            PlotModel.print_in_color(msg)
        classes = list(train_df['labels'].unique())
        class_count = len(classes)
        return train_df, img_size, classes, class_count

    def trim(self, df, max_samples, min_samples, column):
        # column specifies which column of the dataframe to use, typically this is the labels column
        # df is typically train_df
        df = df.copy()
        classes = df[column].unique()  # get the classes in df
        class_count = len(classes)
        length = len(df)
        print('dataframe initially is of length ', length, ' with ', class_count, ' classes')
        groups = df.groupby(column)  # creates a set of  dataframes that only contains rows that have the class label
        trimmed_df = pd.DataFrame(columns=df.columns)  # create an empty dataframe with columns filepaths, labels
        for label in df[column].unique():  # iterate through each class label
            group = groups.get_group(label)  # get the dataframe associate with the label
            count = len(group)  # determine how many files are in the dataframe
            if count > max_samples:  # if there more files in the dataframe sample it so the sampled files has only n rows
                sampled_group = group.sample(n=max_samples, random_state=123, axis=0)
                trimmed_df = pd.concat([trimmed_df, sampled_group], axis=0)
            else:
                if count >= min_samples:  # if the dataframe has more than the minimum number of files include it in the dataset
                    sampled_group = group
                    trimmed_df = pd.concat([trimmed_df, sampled_group], axis=0)
        print('after trimming, the maximum samples in any class is now ', max_samples,
              ' and the minimum samples in any class is ', min_samples)
        classes = trimmed_df[column].unique()  # return this in case some classes have less than min_samples
        class_count = len(
            classes)  # return this in case some classes have less than min_samples and so will have less classes in it
        length = len(trimmed_df)
        print('the trimmed dataframe now is of length ', length, ' with ', class_count, ' classes')
        return trimmed_df, classes, class_count




















    def print_errors(self , error_list, error_pred_list, delimiter):
        if len(error_list) == 0:
            PlotModel.print_in_color('There were no errors in predicting the test set')
        else:
            if len(error_list) > 500:
                PlotModel.print_in_color('There were over 50 misclassifications, the error list will not be printed')
            else:
                PlotModel.print_in_color('Below is a list of test files that were miss classified \n')
                PlotModel.print_in_color('{0:^50s}{1:^50s}'.format('Test File', ' Predicted as'))
                for i in range(len(error_list)):
                    fpath = error_list[i]
                    split = fpath.split(delimiter)
                    slength = len(split)
                    f = split[slength - 2] + '-' + split[slength - 1]
                    print(f'{f:^50s}{error_pred_list[i]:^50s}')

    def predictor(self, model, test_gen):
        y_pred = []
        error_list = []
        error_pred_list = []
        y_true = test_gen.labels
        classes = list(test_gen.class_indices.keys())
        class_count = len(classes)
        errors = 0
        preds = model.predict(test_gen, verbose=1)
        tests = len(preds)
        for i, p in enumerate(preds):
            pred_index = np.argmax(p)
            true_index = test_gen.labels[i]  # labels are integer values
            if pred_index != true_index:  # a misclassification has occurred
                errors = errors + 1
                file = test_gen.filenames[i]
                error_list.append(file)
                error_class = classes[pred_index]
                error_pred_list.append(error_class)
            y_pred.append(pred_index)
        acc = (1 - errors / tests) * 100
        msg = f'there were {errors} errors in {tests} tests for an accuracy of {acc:6.2f}'
        PlotModel.print_in_color(msg, (0, 255, 255), (100, 100, 100))  # cyan foreground
        ypred = np.array(y_pred)
        ytrue = np.array(y_true)
        f1score = f1_score(ytrue, ypred, average='weighted') * 100
        if class_count <= 30:
            cm = confusion_matrix(ytrue, ypred)
            # plot the confusion matrix
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
            plt.xticks(np.arange(class_count) + .5, classes, rotation=90)
            plt.yticks(np.arange(class_count) + .5, classes, rotation=0)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.show()
        clr = classification_report(y_true, y_pred, target_names=classes, digits=4)  # create classification report
        print("Classification Report:\n----------------------\n", clr)
        return errors, tests, error_list, error_pred_list, f1score

    def seed_everything(self , seed=42):
        # Seed value for TensorFlow
        tf.random.set_seed(seed)

        # Seed value for NumPy
        np.random.seed(seed)

        # Seed value for Python's random library
        random.seed(seed)

        # Force TensorFlow to use single thread
        # Multiple threads are a potential source of non-reproducible results.
        session_conf = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1
        )

        # Make sure that TensorFlow uses a deterministic operation wherever possible
        tf.compat.v1.set_random_seed(seed)

        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)
        tf.compat.v1.keras.backend.set_session(sess)


    def F1_score(self , y_true, y_pred):  # taken from old keras source code
        from keras import backend as K
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val
