import matplotlib.pyplot as plt
import numpy as np

class PlotModel:
    @staticmethod
    def print_in_color( txt_msg, fore_tupple=(0, 255, 255), back_tupple=(100, 100, 100), same_line=False):
        # prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple
        # text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)
        # default parameter print in cyan foreground and gray background
        rf, gf, bf = fore_tupple
        rb, gb, bb = back_tupple
        msg = '{0}' + txt_msg
        mat = '\33[38;2;' + str(rf) + ';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' + str(gb) + ';' + str(
            bb) + 'm'
        if same_line:
            print(msg.format(mat), end='', flush=True)  # does not go to a new line so next print is on the same line
        else:
            print(msg.format(mat), flush=True)
        print('\33[0m', end='', flush=True)  # returns default print color to back to black
        return
    @staticmethod
    def plot_label_count(df, plot_title):
        column = 'labels'
        xaxis_label = 'CLASS'
        yaxis_label = 'IMAGE COUNT'
        vcounts = df[column].value_counts()
        labels = vcounts.keys().tolist()
        values = vcounts.tolist()
        lcount = len(labels)
        if lcount > 55:
            PlotModel.print_in_color('The number of labels is >55, no plot will be produced')
        else:
            width = lcount * 4
            width = np.min([width, 20])
            plt.figure(figsize=(width, 5))
            form = {'family': 'serif', 'color': 'blue', 'size': 25}
            plt.bar(labels, values)
            plt.title(plot_title, fontsize=24, color='blue')
            plt.xticks(rotation=90, fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel(xaxis_label, fontdict=form)
            plt.ylabel(yaxis_label, fontdict=form)
            if lcount >= 8:
                rotation = 'vertical'
            else:
                rotation = 'horizontal'
            for i in range(lcount):
                plt.text(i, values[i] / 2, str(values[i]), fontsize=12, rotation=rotation, color='yellow', ha='center')
            plt.show()
    @staticmethod
    def tr_plot(tr_data):
        start_epoch = 0
        # Plot the training and validation data
        tacc = tr_data.history['accuracy']
        tloss = tr_data.history['loss']
        vacc = tr_data.history['val_accuracy']
        vloss = tr_data.history['val_loss']
        tf1 = tr_data.history['F1_score']
        vf1 = tr_data.history['val_F1_score']
        Epoch_count = len(tacc) + start_epoch
        Epochs = []
        for i in range(start_epoch, Epoch_count):
            Epochs.append(i + 1)
        index_loss = np.argmin(vloss)  # this is the epoch with the lowest validation loss
        val_lowest = vloss[index_loss]
        index_acc = np.argmax(vacc)
        acc_highest = vacc[index_acc]
        indexf1 = np.argmax(vf1)
        vf1_highest = vf1[indexf1]
        sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
        vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
        f1_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 10))
        axes[0].plot(Epochs, tloss, 'r', label='Training loss')
        axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
        axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
        axes[0].scatter(Epochs, tloss, s=100, c='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epochs', fontsize=18)
        axes[0].set_ylabel('Loss', fontsize=18)
        axes[0].legend()
        axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
        axes[1].scatter(Epochs, tacc, s=100, c='red')
        axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
        axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epochs', fontsize=18)
        axes[1].set_ylabel('Accuracy', fontsize=18)
        axes[1].legend()
        axes[2].plot(Epochs, tf1, 'r', label='Training F1 score')
        axes[2].plot(Epochs, vf1, 'g', label='Validation F1 score')
        index_tf1 = np.argmax(tf1)  # this is the epoch with the highest training F1 score
        tf1max = tf1[index_tf1]
        index_vf1 = np.argmax(vf1)  # thisiis the epoch with the highest validation F1 score
        vf1max = vf1[index_vf1]
        axes[2].scatter(index_vf1 + 1 + start_epoch, vf1max, s=150, c='blue', label=vc_label)
        axes[2].scatter(Epochs, tf1, s=100, c='red')
        axes[2].set_title('Training and Validation F1 score')
        axes[2].set_xlabel('Epochs', fontsize=18)
        axes[2].set_ylabel('F1  score', fontsize=18)
        axes[2].legend()
        plt.tight_layout
        plt.show()
        return
    @staticmethod
    def show_misclassification( error_list, error_pred_list, test_gen, delimiter):
        if len(error_list) == 0:
            PlotModel.print_in_color('there were no errors in predicting the test images')
        else:
            if len(error_list) < 10:
                length = len(error_list)
            else:
                length = 10  # show 10 images
            msg = 'The images below show 10 misclassified test images on left and an example of an image in the  misclassified class'
            PlotModel.print_in_color(msg)
            test_files = test_gen.filenames
            plt.figure(figsize=(15, length * 5))
            for i in range(length):
                fpath = error_list[i]
                test_img = plt.imread(fpath)
                pred_class = error_pred_list[i]
                # find a test file that is the same class as the pred_class
                for f in test_gen.filenames:
                    split = list(f.split(delimiter))
                    klass = split[len(split) - 2]
                    if klass == pred_class:
                        pred_img_path = f
                pred_img = plt.imread(pred_img_path)
                for j in range(2):
                    k = i * 2 + j + 1
                    plt.subplot(length, 2, k)
                    plt.axis('off')
                    if j == 0:
                        plt.imshow(test_img)
                        split = fpath.split(delimiter)
                        slength = len(split)
                        # print (split)
                        title = split[slength - 2] + '-' + split[slength - 1]
                        title = 'TEST IMAGE\n' + title
                        plt.title(title, color='blue', fontsize=16)
                    else:
                        plt.imshow(pred_img)
                        split = pred_img_path.split(delimiter)
                        slength = len(split)
                        title = split[slength - 2] + '-' + split[slength - 1]
                        title = 'PREDICTED CLASS EXAMPLE\n' + title
                        plt.title(title, color='blue', fontsize=16)
                    plt.show()

    @staticmethod
    def show_image_samples(gen):
        msg = 'Below are some example training images'
        PlotModel.print_in_color(msg)
        t_dict = gen.class_indices
        classes = list(t_dict.keys())
        images, labels = next(gen)  # get a sample batch from the generator
        plt.figure(figsize=(25, 25))
        length = len(labels)
        if length < 25:  # show maximum of 25 images
            r = length
        else:
            r = 25
        for i in range(r):
            plt.subplot(5, 5, i + 1)
            image = images[i] / 255
            plt.imshow(image)
            index = np.argmax(labels[i])
            class_name = classes[index]
            plt.title(class_name, color='blue', fontsize=18)
            plt.axis('off')
        plt.show()

