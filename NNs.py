import os
import sys
import numpy as np
import pandas as pd
from LocallyDirectedConnected_tf2 import LocallyDirected1D
import tensorflow as tf
#tf.compat.v1.disable_v2_behavior() 
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Flatten, Input, Layer, MaxPooling2D, Conv2D, Reshape, Dropout
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.regularizers import l1, l2, l1_l2
from sklearn.metrics import balanced_accuracy_score
tf.keras.backend.set_epsilon(0.0000001)
from utils import Logger
import datetime
import importlib
from tensorflow.python.ops import array_ops, control_flow_ops, math_ops, state_ops

import adam_local
importlib.reload(adam_local)
from adam_local import AdamW 
import gradient
importlib.reload(gradient)
import gradient2
importlib.reload(gradient2)
from gradient2 import eval_grads
#sess = tf.Session(graph=tf.get_default_graph(), config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
#sess = K.get_session()
#K.set_session(sess)
class MyMomentumOptimizer(keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.9, name="MyMomentumOptimizer", **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        print ("initializing the optimizer!!!!!")
        super().__init__(name, **kwargs)
        print ( self._initial_decay)
        for key, value in kwargs.items():
            print("{0} = {1}".format(key, value))
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate)) # handle lr=learning_rate
        self._set_hyper("decay", self._initial_decay) # 
        self._set_hyper("momentum", momentum)
        for key, value in kwargs.items():
            print("{0} = {1}".format(key, value))
        
        print ( self._initial_decay)
    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "momentum")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable
        """
        print ("dense called")
        print (var)
        #tf.print("sdfd", output_stream=sys.stderr)
        var_dtype = var.dtype.base_dtype
        local_step = math_ops.cast(10 + 1, var_dtype)
        lr_t = self._decayed_lr(var_dtype) # handle learning rate decay
        momentum_var = self.get_slot(var, "momentum")
        momentum_hyper = self._get_hyper("momentum", var_dtype)
        momentum_var.assign(momentum_var * momentum_hyper - (1. - momentum_hyper)* grad)
        var_t = momentum_var * lr_t
        noise = tf.random.uniform(var_t.shape, -0.01, 0.01)
        var_t2 = math_ops.sub(var_t, math_ops.cast(noise, var_dtype))
        print (var_t2)
        var.assign_add(var_t2)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
        }


class LR_Scheduler():

    def __init__ (self, init_lr=0.001):
        self.best_loss=100
        self.no_improvement_counter=0
        self.reduced_counter=0
        self.init_lr = init_lr

    def update_lr(self, old_lr, epoch, loss):

        if loss < self.best_loss:
            print("use old lr")
            self.best_loss = loss
            self.no_improvement_counter=0
            return old_lr

        if self.reduced_counter>=4:
            print("restarted lr")
            self.reduced_counter=0
            self.no_improvement_counter=0
            return self.init_lr

        if self.no_improvement_counter > 100:
            print ("decrease lr")
            self.no_improvement_counter=0
            self.reduced_counter+=1
            return old_lr/2

        
        self.no_improvement_counter+=1
        return old_lr


            
        

# BRI-NET MODEL CLASS
class BRInet():

    def __init__(self, epochs, bs, lr, lr_decay, l1_reg, l2_reg, num_cf_reg, wpc, wnc, weights_dir_path, topology_path, input_size, grad_threshold, noise_radius):

        # SETTING CLASS VARIABLES
        self.epochs = epochs
        self.bs = int(bs)
        self.lr = lr
        self.lr_decay = lr_decay
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.num_cf_reg = num_cf_reg
        self.wpc = wpc
        self.wnc = wnc
        self.weights_dir_path = weights_dir_path
        self.encoder_output_size = np.nan # to be set in build_endoer()
        self.input_size = input_size
        # SETTING VARIOUS OPTIMIZERS
        #self.optimizer = AdamN(learning_rate = self.lr)
        self.optimizer = AdamW(learning_rate=lr, grad_threshold=grad_threshold, noise_radius=noise_radius)
        #self.optimizer =myAadam(learning_rate = self.lr, decay = self.lr_decay)
        #self.optimizer = Adam(learning_rate=lr)
        # INPUT LAYER OF MODEL
        #input_size = self.get_feature_count()
        self.input_layer = keras.Input((self.input_size,), name='input_layer')

        # DEFINE THE ENCODER: USED FOR FEATURE EXTRACTION
        feature_dense_enc = self.build_encoder(topology_path)
        feature_dense_enc = Flatten()(feature_dense_enc)
        self.encoder = Model(self.input_layer, feature_dense_enc, name="encoder")

        # DEFINE THE REGRESSOR: USED FOR BIAS PREDICTOR
        self.regressor = self.build_regressor()
        self.regressor.compile(loss=['mse'], optimizer = self.optimizer)
        # DEFINE THE DISTILLER: USED FOR BIAS REMOVAL
        self.regressor.trainable = False
        cf = self.regressor(feature_dense_enc)
        self.distiller = Model(self.input_layer, cf, name="distiller")
        self.distiller.compile(loss = self.correlation_coefficient_loss,
                               optimizer = self.optimizer)

        #self.distiller.compile(loss = self.negative_mse,
        #                       optimizer = self.optimizer)

        # DEFINE THE CLASSIFIER: USED FOR TASK CLASSIFICATION
        input_feature_clf = Input(shape = (self.encoder_output_size,), name='input_features')
        prediction_score  = Dense(1, name='output_layer', kernel_regularizer = l1_l2(l1 =self.l1_reg, l2=self.l2_reg))(input_feature_clf)
        self.classifier   = Model(input_feature_clf, prediction_score, name="classifier")

        # DEFINE THE ENTIRE WORKFLOW: USED FOR TESTING THE TRAINED MODEL
        prediction_score_workflow = self.classifier(feature_dense_enc)
        label_workflow = Activation('sigmoid')(prediction_score_workflow)
        self.workflow = Model(self.input_layer, label_workflow, name="workflow")
        self.workflow.compile(loss = self.weighted_binary_crossentropy,
                              optimizer = self.optimizer,
                              metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                                       keras.metrics.AUC(name='auc'),
                                       keras.metrics.Precision(name='precision'),
                                       keras.metrics.Recall(name='recall'),
                                       keras.metrics.TruePositives(name='TP'),
                                       keras.metrics.TrueNegatives(name='TN'),
                                       keras.metrics.FalsePositives(name='FP'),
                                       keras.metrics.FalseNegatives(name='FN')])

        
    # BUILD THE REGRESSOR FOR BIAS PREDICTION
    def build_regressor(self):

        # HIDDEN LAYERS
        inputs_x = Input(shape = (self.encoder_output_size,))
        layer_1 = Dense(128, activation = 'tanh', name="regressor_1")(inputs_x)
        layer_2 = Dense(64, activation = 'tanh', name="regressor_2")(layer_1)
        layer_3 = Dense(16, activation = 'tanh', name="regressor_3")(layer_2)

        # OUTPUT LAYERS
        out_reg = Dense(self.num_cf_reg, activation='linear', kernel_regularizer = l1_l2(l1 =self.l1_reg, l2=self.l2_reg), name="regressor_output" )(layer_3)  #kernel_regularizer = l1(l = 0.20)), TODO: consider adding

        return Model(inputs = inputs_x, outputs = [out_reg])



    # TRAINING THE ENTIRE MODEL
    def train(self, train_data_x, train_data_y, train_data_cf_reg, test_data_x, test_data_y, fold=0):
        print (train_data_cf_reg.shape)
        print (train_data_cf_reg.sum())
        scheduler = LR_Scheduler(self.lr)
        # MAX TEST ACCURACY VARIABLE
        min_loss = 1000
        max_auc = 0
        max_f1_score = 0
        weight_path = self.weights_dir_path + '/best_model_fold_' + str(fold) + '_.h5'
        weight_path_best_auc = self.weights_dir_path + '/best_auc_fold_' + str(fold) + '_.h5'
        weight_path_best_F1 = self.weights_dir_path + '/best_F1_fold_' + str(fold) + '_.h5'
        log_path_train_error = os.path.join(self.weights_dir_path, "log_train_fold_{}.txt".format(fold))
        log_path_test_error = os.path.join(self.weights_dir_path, "log_test_fold_{}.txt".format(fold))
        log_path_grads = os.path.join(self.weights_dir_path, "grad_mag_{}.txt".format(fold))
        log_path_lr = os.path.join(self.weights_dir_path, "lr_{}.txt".format(fold))
        metrics_names_path = os.path.join(self.weights_dir_path, "metrics_names.txt")
        timing_info_path = os.path.join(self.weights_dir_path, "timing_info.txt")
        logger_train = Logger(log_path_train_error)
        logger_test= Logger(log_path_test_error)
        logger_grads= Logger(log_path_grads)
        logger_lr= Logger(log_path_lr)
        logger_time = Logger(timing_info_path)
        logger_time.log(str(datetime.datetime.now()))
        distiller_loss_path = os.path.join(self.weights_dir_path, "distiller_loss_fold_{}.txt".format(fold))
        regressor_loss_path = os.path.join(self.weights_dir_path, "regressor_loss_fold_{}.txt".format(fold))
        logger_distiller = Logger(distiller_loss_path)
        logger_regressor = Logger(regressor_loss_path)
        for epoch in range(self.epochs):
            #print (keras.backend.get_value(self.workflow.optimizer.learning_rate))
            #print (self.workflow.optimizer.get_learning_rate())
            if epoch%50==0:
                print ("epoch : {}".format(epoch))
                logger_time.log("epoch: {}, time: {}".format(epoch, str(datetime.datetime.now())))
            # SELECT A RANDOM BATCH
            ctr_idx_total = np.where(train_data_y==0)[0]
            trt_idx_total = np.where(train_data_y==1)[0]
            ctr_idx = np.random.choice(ctr_idx_total, self.bs)
            idx = np.concatenate([np.random.choice(ctr_idx_total, int(self.bs/2)), np.random.choice(trt_idx_total, int(self.bs/2))])
            #idx = np.random.permutation(idx)
            training_x_ctr_batch = train_data_x[ctr_idx]
            training_y_ctr_batch = train_data_y[ctr_idx]
            training_cf_reg_ctr_batch = train_data_cf_reg[ctr_idx]

            training_x_batch = train_data_x[idx]
            training_y_batch = train_data_y[idx]
            training_cf_reg_batch = train_data_cf_reg[idx]

            #  TRAIN THE REGRESSOR
            encoded_feature_batch = self.encoder.predict(training_x_ctr_batch)
            r_loss = self.regressor.train_on_batch(encoded_feature_batch, [training_cf_reg_ctr_batch])
            g_loss = self.distiller.train_on_batch(training_x_ctr_batch, [training_cf_reg_ctr_batch])
            #g_loss_test = self.distiller.test_on_batch(training_x_ctr_batch, [training_cf_reg_ctr_batch])
            logger_regressor.log('{}'.format(r_loss))
            logger_distiller.log('{}'.format(g_loss))
            #eval_grads(self.workflow, training_x_batch, training_y_batch)
            #grads = gradient.get_gradients(self.workflow, training_x_batch, training_y_batch, self.workflow.trainable_weights)
            #mags = gradient.summerize_grads(grads)
            #mags = [str(x) for x in mags]
            #logger_grads.log("{}, {}".format(epoch, ",".join(mags)))
            
            # TRAIN THE ENCODER AND CLASSIFIER
            c_loss = self.workflow.train_on_batch(training_x_batch, training_y_batch)

            # TEST THE ENTIRE MODEL ON THE TEST SET
            c_loss_test = self.workflow.evaluate(test_data_x, test_data_y, verbose = 0, batch_size = self.bs)
            precision, recall = c_loss_test[3], c_loss_test[4]            
            f1_score = (2*precision*recall)/(precision+recall+1e-10)
            auc = c_loss_test[2]
            # GET MODEL WEIGHTS OF MAX PERFORMING MODEL
            if c_loss_test[0] < min_loss:
                min_loss = c_loss_test[0]
                self.workflow.save(weight_path)

            if auc > max_auc:
                max_auc = auc
                self.workflow.save(weight_path_best_auc)

            if f1_score > max_f1_score:
                max_f1_score = f1_score
                self.workflow.save(weight_path_best_F1)                

            #weight_path_epoch = self.weights_dir_path + '/model_epoch_'+str(epoch)+"_fold_" + str(fold) + '_.h5'
            #self.workflow.save(weight_path_epoch)
            old_lr = keras.backend.get_value(self.workflow.optimizer.learning_rate)
            new_lr = scheduler.update_lr(old_lr, epoch, c_loss_test[0] )
            keras.backend.set_value(self.workflow.optimizer.learning_rate, new_lr )
            logger_lr.log("{}, {}".format(epoch, old_lr))
            # fold, epoch, loss, Accuracy, AUC, Precision, Recall, Balanced Accuracy
            #TP = c_loss[5]
            #TN = c_loss[6]
            #FP = c_loss[7]
            #FN = c_loss[8]
            #b_acc_train = ((TP / (TP+FN)) + (TN / (FP+TN)))/2

            #TP = c_loss_test[5]
            #TN = c_loss_test[6]
            #FP = c_loss_test[7]
            #FN = c_loss_test[8]
            #b_acc_test = ((TP / (TP+FN)) + (TN / (FP+TN)))/2

            #self.train_acc.append([fold, epoch, c_loss[0], c_loss[1], c_loss[2], c_loss[3], c_loss[4], b_acc_train])
            #self.test_acc.append([fold, epoch, c_loss_test[0], c_loss_test[1], c_loss_test[2], c_loss_test[3], c_loss_test[4], b_acc_test])
            message_test = ",".join(['{:.2f}'.format(x) for x in c_loss_test])
            message_train = ",".join(['{:.2f}'.format(x) for x in c_loss])
            message_test = "{},".format(epoch)+message_test
            message_train = "{},".format(epoch)+message_train
            logger_train.log(message_train)
            logger_test.log(message_test)            
        metrics_names = ",".join(self.workflow.metrics_names)
        metrics_names = "epoch,"+metrics_names
        metrics_names_logger = Logger(metrics_names_path)
        metrics_names_logger.log(metrics_names)
    # BUILD ENCODER FOR FEATURE EXTRACTION
    def build_encoder(self, masks_path):
        
        # HELPER FUNCTION FOR BUILDING LAYERS
        def layer_block(model, mask, i):
            model = LocallyDirected1D(mask=mask,
                                      kernel_regularizer=l1_l2(l1 =self.l1_reg, l2=self.l2_reg),
                                      filters=1,
                                      input_shape=(mask.shape[0], 1),
                                      name="LocallyDirected_" + str(i))(model)
            model = keras.layers.Activation("tanh")(model)
            model = keras.layers.BatchNormalization(center=False, scale=False)(model)

            return model

        # PROCESS MODEL DEFINING FILES AND BUILD THE MODEL
        masks = np.load(masks_path, allow_pickle=True)
        self.encoder_output_size = masks[-1].shape[-1]
        model = keras.layers.Reshape(input_shape=(self.input_size,), target_shape=(self.input_size, 1))(self.input_layer)

        for i in range(len(masks)):
            mask = masks[i]
            model = layer_block(model, mask, i)
        return model

    # ADVERSARIAL LOSS USED FOR DISTILLER REGRESSION
    # USE PEARSONS COEFFICIENT FOR A MEASURE OF CORRELATION BETWEEN TWO NUMERICAL VARIABLES    
#    def correlation_coefficient_loss(self, y_true, y_pred):
#        print ("initiatedddddddddddddddddddddddddddddddddddddddddd")
#        y_true = tf.cast(y_true, tf.float32)
#        y_pred = tf.cast(y_pred, tf.float32)
#        corr_sum = 0
#        for i in range(0, self.num_cf_reg):
#            x = y_true[:, i]
#            y = y_pred[:, i]
#            mx = K.mean(x)
#            my = K.mean(y)
#            xm, ym = x-mx, y-my
#            r_num = K.sum(tf.multiply(xm,ym))
#            r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym)))) + 1e-5
#            r = r_num / r_den
#            r = K.maximum(K.minimum(r, 1.0), -1.0)
#            corr_sum = corr_sum + K.square(r)
#        print (corr_sum)
#        return corr_sum
    def correlation_coefficient_loss(self, y_true, y_pred):
        x = y_true
        y = y_pred
        mx = K.mean(x)
        my = K.mean(y)
        xm, ym = x-mx, y-my
        r_num = K.sum(tf.multiply(xm,ym))
        r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym)))) + 1e-5
        r = r_num / r_den
        r = K.maximum(K.minimum(r, 1.0), -1.0)
        return K.square(r)


    # CUSTOM WEIGHTED BINARY CROSS ENTROPY LOSS FUNCTION FOR ENTIRE WORKFLOW
    def weighted_binary_crossentropy(self, y_true, y_pred):
        epsilon=1e-12
        y_true = K.clip(tf.cast(y_true, tf.float64), epsilon, 1- epsilon)
        y_pred = K.clip(tf.cast(y_pred, tf.float64), epsilon, 1- epsilon)

        return K.mean(-y_true * K.log(y_pred) * self.wpc - (1 - y_true) * K.log(1 - y_pred) * self.wnc)

    def mean_squared_error(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=0)

    def negative_mse(self, y_true, y_pred):
      #y_pred = tf.convert_to_tensor_v2(y_pred)
      y_true = tf.cast(y_true, y_pred.dtype)
      print (y_true.shape)
      print (y_pred.shape)
      return -tf.reduce_mean(math_ops.square(y_pred - y_true))

## BASELINE MODEL CLASS
#class Baseline():
#
#    def __init__(self, epochs, bs, lr, lr_decay, l1_reg, wpc, wnc, weights_dir_path):
#
#        # SETTING VARIABLES
#        self.epochs = epochs
#        self.bs = bs
#        self.lr = lr
#        self.lr_decay = lr_decay
#        self.l1_reg = l1_reg
#        self.wpc = wpc
#        self.wnc = wnc
#        self.weights_dir_path = weights_dir_path
#
#        self.train_acc = []
#        self.test_acc = []
#        self.optimizer = Adam(learning_rate = self.lr, decay = self.lr_decay)
#
#        # ENCODER: FEATURE EXTRACTOR
#        self.workflow = self.build_classifier()
#
#
#
#    # TRAINING THE ENTIRE MODEL
#    def train(self, train_data_x, train_data_y, test_data_x, test_data_y, fold=0):
#
#        # MAX TEST ACCURACY VARIABLE
#        max_acc = 0
#        weight_path = self.weights_dir_path + '/baseline_best_model_fold_' + str(fold) + '_.h5'
#
#        for epoch in range(self.epochs):
#
#            # SELECT A RANDOM BATCH FROM TRAINING DATA
#            idx_perm = np.random.permutation(int(train_data_x.shape[0]/2))
#            idx = idx_perm[:int(self.bs/2)]
#            idx = np.concatenate((idx,idx+int(train_data_x.shape[0]/2)))
#            training_x_batch = train_data_x[idx]
#            training_y_batch = train_data_y[idx]
#
#            # TRAIN THE WORKFLOW ON TRAINING DATA
#            c_loss = self.workflow.train_on_batch(training_x_batch, training_y_batch)
#
#            # TEST THE WORKFLOW ON THE TEST DATA
#            c_loss_test = self.workflow.evaluate(test_data_x, test_data_y, verbose = 0, batch_size = self.bs)
#
#            # GET MODEL WEIGHTS OF MAX PERFORMING MODEL
#            if c_loss_test[1] > max_acc:
#                max_acc = c_loss_test[1]
#                self.workflow.save(weight_path)
#
#            # fold, epoch, loss, Accuracy, AUC, Precision, Recall, Balanced Accuracy
#            TP = c_loss[5]
#            TN = c_loss[6]
#            FP = c_loss[7]
#            FN = c_loss[8]
#            b_acc_train = ((TP / (TP+FN)) + (TN / (FP+TN)))/2
#
#            TP = c_loss_test[5]
#            TN = c_loss_test[6]
#            FP = c_loss_test[7]
#            FN = c_loss_test[8]
#            b_acc_test = ((TP / (TP+FN)) + (TN / (FP+TN)))/2
#
#            self.train_acc.append([fold, epoch, c_loss[0], c_loss[1], c_loss[2], c_loss[3], c_loss[4], b_acc_train])
#            self.test_acc.append([fold, epoch, c_loss_test[0], c_loss_test[1], c_loss_test[2], c_loss_test[3], c_loss_test[4], b_acc_test])
#
#
#
#    # BUILD CLASSIFIER USING GenNet CODE
#    def build_classifier(self):
#
#        # LOCATION OF MODEL FILES
#        datapath = './Pre_Processing_Files/Datasets/Model_Files/Baseline/'
#
#        # HELPER FUNCTION FOR BUILDING LAYERS
#        def layer_block(model, mask, i):
#            model = LocallyDirected1D(mask=mask,
#                                      filters=1,
#                                      input_shape=(mask.shape[0], 1),
#                                      name="LocallyDirected_" + str(i))(model)
#            model = keras.layers.Activation("tanh")(model)
#            model = keras.layers.BatchNormalization(center=False, scale=False)(model)
#
#            return model
#
#        # PROCESS MODEL DEFINING FILES AND BUILD THE MODEL
#        masks = []
#        network_csv = pd.read_csv(datapath + "topology.csv")
#        network_csv = network_csv.filter(like="node", axis=1)
#        columns = list(network_csv.columns.values)
#        network_csv = network_csv.sort_values(by=columns, ascending=True)
#
#        inputsize = self.get_feature_count()
#
#        input_layer = keras.Input((inputsize,), name='input_layer')
#        model = keras.layers.Reshape(input_shape=(inputsize,), target_shape=(inputsize, 1))(input_layer)
#
#        for i in range(len(columns) - 1):
#            network_csv2 = network_csv.drop_duplicates(columns[i])
#            matrix_ones = np.ones(len(network_csv2[[columns[i], columns[i + 1]]]), np.bool)
#            matrix_coord = (network_csv2[columns[i]].values, network_csv2[columns[i + 1]].values)
#            if i == 0:
#                matrixshape = (inputsize, network_csv2[columns[i + 1]].max() + 1)
#            else:
#                matrixshape = (network_csv2[columns[i]].max() + 1, network_csv2[columns[i + 1]].max() + 1)
#            mask = scipy.sparse.coo_matrix(((matrix_ones), matrix_coord), shape = matrixshape)
#            masks.append(mask)
#           model = layer_block(model, mask, i)
#
#        model = keras.layers.Flatten()(model)
#
#        output_layer = keras.layers.Dense(units=1,
#                                          name="output_layer",
#                                          kernel_regularizer=tf.keras.regularizers.l1(l=self.l1_reg))(model)
#
#        output_layer = keras.layers.Activation("sigmoid")(output_layer)
#
#        model = keras.Model(inputs=input_layer, outputs=output_layer)
#
#        model.compile(loss = self.weighted_binary_crossentropy,
#                      optimizer = self.optimizer,
#                      metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
#                               keras.metrics.AUC(name='auc'),
#                               keras.metrics.Precision(name='precision'),
#                               keras.metrics.Recall(name='recall'),
#                               keras.metrics.TruePositives(name='TP'),
#                               keras.metrics.TrueNegatives(name='TN'),
#                               keras.metrics.FalsePositives(name='FP'),
#                               keras.metrics.FalseNegatives(name='FN')])
#
#        return model
#
#
#
#    # GET NUMBER OF INPUT FEATURES (NUMBER OF UNIQUE SNPS)
#    def get_feature_count(self):
#        datapath = './Pre_Processing_Files/Datasets/Model_Files/Baseline/'
#        h5file = tables.open_file(datapath + "genotype.h5", "r")
#        input_shape = h5file.root.data.shape[1]
#        h5file.close()
#
#        return input_shape
#
#
#    # CUSTOM WEIGHTED BINARY CROSS ENTROPY LOSS FUNCTION FOR ENTIRE WORKFLOW
#    def weighted_binary_crossentropy(self, y_true, y_pred):
#        y_true = K.clip(tf.cast(y_true, tf.float64), 0, 1)
#        y_pred = K.clip(tf.cast(y_pred, tf.float64), 0, 1)
#
#        return K.mean(-y_true * K.log(y_pred) * self.wpc - (1 - y_true) * K.log(1 - y_pred) * self.wnc)
#
#



# BRI-NET MODEL CLASS
class Baseline():

    def __init__(self, epochs, bs, lr, lr_decay, l1_reg, l2_reg, wpc, wnc, weights_dir_path, topology_path, input_size, grad_threshold, noise_radius):

        # SETTING CLASS VARIABLES
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.lr_decay = lr_decay
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.wpc = wpc
        self.wnc = wnc
        self.weights_dir_path = weights_dir_path

        self.encoder_output_size = np.nan # to be set in build_endoer()
        self.input_size = input_size
        # SETTING VARIOUS OPTIMIZERS
        #self.optimizer = Adam(learning_rate = self.lr, decay = self.lr_decay)
        self.optimizer = AdamW(learning_rate=lr, grad_threshold=grad_threshold, noise_radius=noise_radius)
        # INPUT LAYER OF MODEL
        #input_size = self.get_feature_count()
        self.input_layer = keras.Input((self.input_size,), name='input_layer')

        # DEFINE THE ENCODER: USED FOR FEATURE EXTRACTION
        feature_dense_enc = self.build_encoder(topology_path)
        feature_dense_enc = Flatten()(feature_dense_enc)
        self.encoder = Model(self.input_layer, feature_dense_enc)

        # DEFINE THE CLASSIFIER: USED FOR TASK CLASSIFICATION
        input_feature_clf = Input(shape = (self.encoder_output_size,), name='input_features')
        prediction_score = Dense(1, name='output_layer', kernel_regularizer = l1_l2(l1 =self.l1_reg, l2=self.l2_reg))(input_feature_clf) #kernel_regularizer = l1(l = self.l1_reg)
        self.classifier = Model(input_feature_clf, prediction_score)

        # DEFINE THE ENTIRE WORKFLOW: USED FOR TESTING THE TRAINED MODEL
        prediction_score_workflow = self.classifier(feature_dense_enc)
        label_workflow = Activation('sigmoid')(prediction_score_workflow)
        self.workflow = Model(self.input_layer, label_workflow)
        self.workflow.compile(loss = self.weighted_binary_crossentropy,
                              optimizer = self.optimizer,
                              metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                                       keras.metrics.AUC(name='auc'),
                                       keras.metrics.Precision(name='precision'),
                                       keras.metrics.Recall(name='recall'),
                                       keras.metrics.TruePositives(name='TP'),
                                       keras.metrics.TrueNegatives(name='TN'),
                                       keras.metrics.FalsePositives(name='FP'),
                                       keras.metrics.FalseNegatives(name='FN')])



    # BUILD THE REGRESSOR FOR BIAS PREDICTION

    # TRAINING THE ENTIRE MODEL
    def train(self, train_data_x, train_data_y, train_data_cf_reg, test_data_x, test_data_y, fold=0):
        scheduler = LR_Scheduler(self.lr)
        # MAX TEST ACCURACY VARIABLE
        min_loss = 1000
        max_auc = 0
        max_f1_score = 0       
        weight_path = self.weights_dir_path + '/best_model_fold_' + str(fold) + '_.h5'
        weight_path_best_auc = self.weights_dir_path + '/best_auc_fold_' + str(fold) + '_.h5'
        weight_path_best_F1 = self.weights_dir_path + '/best_F1_fold_' + str(fold) + '_.h5'        
        log_path_train_error = os.path.join(self.weights_dir_path, "log_train_fold_{}.txt".format(fold))
        log_path_test_error = os.path.join(self.weights_dir_path, "log_test_fold_{}.txt".format(fold))
        log_path_grads = os.path.join(self.weights_dir_path, "grad_mag_{}.txt".format(fold))
        log_path_lr = os.path.join(self.weights_dir_path, "lr_{}.txt".format(fold))        
        timing_info_path = os.path.join(self.weights_dir_path, "timing_info.txt")
        metrics_names_path = os.path.join(self.weights_dir_path, "metrics_names.txt")
        logger_train = Logger(log_path_train_error)
        logger_test= Logger(log_path_test_error)
        logger_grads= Logger(log_path_grads)
        logger_lr= Logger(log_path_lr)
        logger_time = Logger(timing_info_path)
        logger_time.log(str(datetime.datetime.now()))        
        for epoch in range(self.epochs):
            if epoch%50==0:
                print ("epoch : {}".format(epoch))
                logger_time.log("epoch: {}, time: {}".format(epoch, str(datetime.datetime.now())))
            # SELECT A RANDOM BATCH
            idx_perm = np.random.permutation(int(train_data_x.shape[0]/2))
            idx = idx_perm[:int(self.bs/2)]
            idx = np.concatenate((idx,idx+int(train_data_x.shape[0]/2)))

            training_x_batch = train_data_x[idx]
            training_y_batch = train_data_y[idx]

            grads = gradient.get_gradients(self.workflow, training_x_batch, training_y_batch, self.workflow.trainable_weights)
            mags = gradient.summerize_grads(grads)
            mags = [str(x) for x in mags]
            logger_grads.log("{}, {}".format(epoch, ",".join(mags)))
    

            # TRAIN THE ENCODER AND CLASSIFIER
            c_loss = self.workflow.train_on_batch(training_x_batch, training_y_batch)

            # TEST THE ENTIRE MODEL ON THE TEST SET
            c_loss_test = self.workflow.evaluate(test_data_x, test_data_y, verbose = 0, batch_size = self.bs)
            precision, recall = c_loss_test[3], c_loss_test[4]
            f1_score = (2*precision*recall)/(precision+recall+1e-10)
            auc = c_loss_test[2]

            # GET MODEL WEIGHTS OF MAX PERFORMING MODEL
            if c_loss_test[0] < min_loss:
                min_loss = c_loss_test[0]
                self.workflow.save(weight_path)
            if auc > max_auc:
                max_auc = auc
                self.workflow.save(weight_path_best_auc)

            if f1_score > max_f1_score:
                max_f1_score = f1_score
                self.workflow.save(weight_path_best_F1)

            #weight_path_epoch = self.weights_dir_path + '/model_epoch_'+str(epoch)+"_fold_" + str(fold) + '_.h5'
            #self.workflow.save(weight_path_epoch)
            old_lr = keras.backend.get_value(self.workflow.optimizer.learning_rate)
            new_lr = scheduler.update_lr(old_lr, epoch, c_loss_test[0] )
            keras.backend.set_value(self.workflow.optimizer.learning_rate, new_lr )


            message_test = ",".join(['{:.2f}'.format(x) for x in c_loss_test])
            message_train = ",".join(['{:.2f}'.format(x) for x in c_loss])
            message_test = "{},".format(epoch)+message_test
            message_train = "{},".format(epoch)+message_train
            logger_train.log(message_test)
            logger_test.log(message_train)            
        metrics_names = ",".join(self.workflow.metrics_names)
        metrics_names = "epoch,"+metrics_names
        metrics_names_logger = Logger(metrics_names_path)
        metrics_names_logger.log(metrics_names)                        



    # BUILD ENCODER FOR FEATURE EXTRACTION
    def build_encoder(self, masks_path):

        # HELPER FUNCTION FOR BUILDING LAYERS
        def layer_block(model, mask, i):
            model = LocallyDirected1D(mask=mask,
                                      kernel_regularizer=l1_l2(l1 =self.l1_reg, l2=self.l2_reg),
                                      filters=1,
                                      input_shape=(mask.shape[0], 1),
                                      name="LocallyDirected_" + str(i))(model)
            model = keras.layers.Activation("tanh")(model)
            model = keras.layers.BatchNormalization(center=False, scale=False)(model)

            return model

        # PROCESS MODEL DEFINING FILES AND BUILD THE MODEL
        masks = np.load(masks_path, allow_pickle=True)
        self.encoder_output_size = masks[-1].shape[-1]
        model = keras.layers.Reshape(input_shape=(self.input_size,), target_shape=(self.input_size, 1))(self.input_layer)

        for i in range(len(masks)):
            mask = masks[i]
            model = layer_block(model, mask, i)
        return model

    # CUSTOM WEIGHTED BINARY CROSS ENTROPY LOSS FUNCTION FOR ENTIRE WORKFLOW

    def weighted_binary_crossentropy(self, y_true, y_pred):
        epsilon=1e-12
        y_true = K.clip(tf.cast(y_true, tf.float64), epsilon, 1- epsilon)
        y_pred = K.clip(tf.cast(y_pred, tf.float64), epsilon, 1- epsilon)

        return K.mean(-y_true * K.log(y_pred) * self.wpc - (1 - y_true) * K.log(1 - y_pred) * self.wnc)





# Multi Modal Gennet
class Baseline_MM():

    def __init__(self, epochs, bs, lr, lr_decay, l1_reg, l2_reg, wpc, wnc, weights_dir_path, topology_path, input_size, grad_threshold, noise_radius, confounders):

        # SETTING CLASS VARIABLES
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.lr_decay = lr_decay
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.wpc = wpc
        self.wnc = wnc
        self.weights_dir_path = weights_dir_path
        self.confounders = confounders

        self.encoder_output_size = np.nan # to be set in build_endoer()
        self.input_size = input_size
        # SETTING VARIOUS OPTIMIZERS
        #self.optimizer = Adam(learning_rate = self.lr, decay = self.lr_decay)
        self.optimizer = AdamW(learning_rate=lr, grad_threshold=grad_threshold, noise_radius=noise_radius)
        

        # INPUT LAYER OF MODEL
        #input_size = self.get_feature_count()
        self.input_layer = keras.Input((self.input_size,), name='input_layer')

        # DEFINE THE ENCODER: USED FOR FEATURE EXTRACTION
        feature_dense_enc = self.build_encoder(topology_path)
        feature_dense_enc = Flatten()(feature_dense_enc)
        self.encoder = Model(self.input_layer, feature_dense_enc)

        

        # DEFINE THE CLASSIFIER: USED FOR TASK CLASSIFICATION
        input_feature_clf = Input(shape = (self.encoder_output_size,), name='input_features')
        prediction_score = Dense(1, name='output_layer', kernel_regularizer = l1_l2(l1 =self.l1_reg, l2=self.l2_reg))(input_feature_clf) #kernel_regularizer = l1(l = self.l1_reg)
        self.classifier = Model(input_feature_clf, prediction_score)

        # DEFINE THE ENTIRE WORKFLOW: USED FOR TESTING THE TRAINED MODEL
        prediction_score_workflow = self.classifier(feature_dense_enc)
        label_workflow = Activation('sigmoid')(prediction_score_workflow)
        self.workflow = Model(self.input_layer, label_workflow)
        self.workflow.compile(loss = self.weighted_binary_crossentropy,
                              optimizer = self.optimizer,
                              metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                                       keras.metrics.AUC(name='auc'),
                                       keras.metrics.Precision(name='precision'),
                                       keras.metrics.Recall(name='recall'),
                                       keras.metrics.TruePositives(name='TP'),
                                       keras.metrics.TrueNegatives(name='TN'),
                                       keras.metrics.FalsePositives(name='FP'),
                                       keras.metrics.FalseNegatives(name='FN')])



    # BUILD THE REGRESSOR FOR BIAS PREDICTION

    # TRAINING THE ENTIRE MODEL
    def train(self, train_data_x, train_data_y, train_data_cf_reg, test_data_x, test_data_y, fold=0):
        scheduler = LR_Scheduler(self.lr)
        # MAX TEST ACCURACY VARIABLE
        min_loss = 1000
        max_auc = 0
        max_f1_score = 0       
        weight_path = self.weights_dir_path + '/best_model_fold_' + str(fold) + '_.h5'
        weight_path_best_auc = self.weights_dir_path + '/best_auc_fold_' + str(fold) + '_.h5'
        weight_path_best_F1 = self.weights_dir_path + '/best_F1_fold_' + str(fold) + '_.h5'        
        log_path_train_error = os.path.join(self.weights_dir_path, "log_train_fold_{}.txt".format(fold))
        log_path_test_error = os.path.join(self.weights_dir_path, "log_test_fold_{}.txt".format(fold))
        log_path_grads = os.path.join(self.weights_dir_path, "grad_mag_{}.txt".format(fold))
        log_path_lr = os.path.join(self.weights_dir_path, "lr_{}.txt".format(fold))        
        timing_info_path = os.path.join(self.weights_dir_path, "timing_info.txt")
        metrics_names_path = os.path.join(self.weights_dir_path, "metrics_names.txt")
        logger_train = Logger(log_path_train_error)
        logger_test= Logger(log_path_test_error)
        logger_grads= Logger(log_path_grads)
        logger_lr= Logger(log_path_lr)
        logger_time = Logger(timing_info_path)
        logger_time.log(str(datetime.datetime.now()))        
        for epoch in range(self.epochs):
            if epoch%50==0:
                print ("epoch : {}".format(epoch))
                logger_time.log("epoch: {}, time: {}".format(epoch, str(datetime.datetime.now())))
            # SELECT A RANDOM BATCH
            idx_perm = np.random.permutation(int(train_data_x.shape[0]/2))
            idx = idx_perm[:int(self.bs/2)]
            idx = np.concatenate((idx,idx+int(train_data_x.shape[0]/2)))

            training_x_batch = train_data_x[idx]
            training_y_batch = train_data_y[idx]

            grads = gradient.get_gradients(self.workflow, training_x_batch, training_y_batch, self.workflow.trainable_weights)
            mags = gradient.summerize_grads(grads)
            mags = [str(x) for x in mags]
            logger_grads.log("{}, {}".format(epoch, ",".join(mags)))
    

            # TRAIN THE ENCODER AND CLASSIFIER
            c_loss = self.workflow.train_on_batch(training_x_batch, training_y_batch)

            # TEST THE ENTIRE MODEL ON THE TEST SET
            c_loss_test = self.workflow.evaluate(test_data_x, test_data_y, verbose = 0, batch_size = self.bs)
            precision, recall = c_loss_test[3], c_loss_test[4]
            f1_score = (2*precision*recall)/(precision+recall+1e-10)
            auc = c_loss_test[2]

            # GET MODEL WEIGHTS OF MAX PERFORMING MODEL
            if c_loss_test[0] < min_loss:
                min_loss = c_loss_test[0]
                self.workflow.save(weight_path)
            if auc > max_auc:
                max_auc = auc
                self.workflow.save(weight_path_best_auc)

            if f1_score > max_f1_score:
                max_f1_score = f1_score
                self.workflow.save(weight_path_best_F1)

            #weight_path_epoch = self.weights_dir_path + '/model_epoch_'+str(epoch)+"_fold_" + str(fold) + '_.h5'
            #self.workflow.save(weight_path_epoch)
            old_lr = keras.backend.get_value(self.workflow.optimizer.learning_rate)
            new_lr = scheduler.update_lr(old_lr, epoch, c_loss_test[0] )
            keras.backend.set_value(self.workflow.optimizer.learning_rate, new_lr )


            message_test = ",".join(['{:.2f}'.format(x) for x in c_loss_test])
            message_train = ",".join(['{:.2f}'.format(x) for x in c_loss])
            message_test = "{},".format(epoch)+message_test
            message_train = "{},".format(epoch)+message_train
            logger_train.log(message_test)
            logger_test.log(message_train)            
        metrics_names = ",".join(self.workflow.metrics_names)
        metrics_names = "epoch,"+metrics_names
        metrics_names_logger = Logger(metrics_names_path)
        metrics_names_logger.log(metrics_names)                        



    # BUILD ENCODER FOR FEATURE EXTRACTION
    def build_encoder(self, masks_path):

        # HELPER FUNCTION FOR BUILDING LAYERS
        def layer_block(model, mask, i):
            model = LocallyDirected1D(mask=mask,
                                      kernel_regularizer=l1_l2(l1 =self.l1_reg, l2=self.l2_reg),
                                      filters=1,
                                      input_shape=(mask.shape[0], 1),
                                      name="LocallyDirected_" + str(i))(model)
            model = keras.layers.Activation("tanh")(model)
            model = keras.layers.BatchNormalization(center=False, scale=False)(model)

            return model

        # PROCESS MODEL DEFINING FILES AND BUILD THE MODEL
        masks = np.load(masks_path, allow_pickle=True)
        self.encoder_output_size = masks[-1].shape[-1]
        model = keras.layers.Reshape(input_shape=(self.input_size,), target_shape=(self.input_size, 1))(self.input_layer)

        for i in range(len(masks)):
            mask = masks[i]
            model = layer_block(model, mask, i)
        return model

    # CUSTOM WEIGHTED BINARY CROSS ENTROPY LOSS FUNCTION FOR ENTIRE WORKFLOW

    def weighted_binary_crossentropy(self, y_true, y_pred):
        epsilon=1e-12
        y_true = K.clip(tf.cast(y_true, tf.float64), epsilon, 1- epsilon)
        y_pred = K.clip(tf.cast(y_pred, tf.float64), epsilon, 1- epsilon)

        return K.mean(-y_true * K.log(y_pred) * self.wpc - (1 - y_true) * K.log(1 - y_pred) * self.wnc)


