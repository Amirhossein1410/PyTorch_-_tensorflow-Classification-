import os
from plot_model import PlotModel
import tensorflow as tf
from keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import Adam, Adamax
from keras.metrics import categorical_crossentropy
from keras import regularizers
from keras.models import Model
from customization import Customize
from sklearn.metrics import f1_score
from tensorflow.core import framework


class MakeModel:
    @staticmethod
    def save_model(model,subject, classes, img_size, f1score, working_dir):
        name=subject + '-' + str(len(classes)) + '-(' + str(img_size[0]) + ' X ' + str(img_size[1]) + ')'
        save_id=f'{name}-{f1score:5.2f}.h5'
        model_save_loc=os.path.join(working_dir, save_id)
        model.save(model_save_loc)
        msg= f'model was saved as {model_save_loc}'
        PlotModel.print_in_color(msg, (0,255,255), (100,100,100)) # cyan foreground

    @staticmethod
    def make_model( img_size, class_count, lr, ans):
        def F1_score(y_true, y_pred):  # taken from old keras source code
            from keras import backend as K
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            recall = true_positives / (possible_positives + K.epsilon())
            f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
            return f1_val
        
        img_shape = (img_size[0], img_size[1], 3)
        if ans == 's' or ans == 'S':
            base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet",
                                                                           input_shape=img_shape, pooling='max')
            msg = 'Created EfficientNet B0 model'
        elif ans == 'l' or ans == 'L':
            base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights="imagenet",
                                                                           input_shape=img_shape, pooling='max')
            msg = 'Created EfficientNet B7 model'
        else:
            base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights="imagenet",
                                                                           input_shape=img_shape, pooling='max')
            msg = 'Created EfficientNet B3 model'
        base_model.trainable = True
        x = base_model.output
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
        x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
                  bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
        x = Dropout(rate=.4, seed=123)(x)
        output = Dense(class_count, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy', F1_score, 'AUC'])
        msg = msg + f' with initial learning rate set to {lr}'
        PlotModel.print_in_color(msg)
        return model