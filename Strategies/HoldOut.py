from Strategies.Abstract.Strategy import Strategy
from Utils.LossHistory import LossHistory


class HoldOut(Strategy):

    def __init__(self, params):
        'Constructor'
        super(HoldOut, self).__init__('HoldOut', params)

    def fit(self, model, data, new_data):
        import time
        import keras
        import numpy
        from keras.callbacks import ModelCheckpoint, EarlyStopping
        from sklearn.model_selection import train_test_split

        params = self.get_params()
        X = data['X']

        if model.get_name().lower() == 'vgg16':
            X = self.prepare_data(X, True)
        elif model.get_name().lower() == 'mobile_net' or model.get_name().lower() == 'resnet' or model.get_name().lower() == 'unet':
            X = self.prepare_data(X, False)

        Y = data['Y']

        if params['use_new_data']:
            new_X = new_data['X']
            new_Y = new_data['Y']
            if (self.get_params()[
                'problem_type'] == 'classification'):
                new_Y = keras.utils.to_categorical(new_Y, num_classes=len(numpy.unique(Y)))
            if model.get_name().lower() == 'vgg16':
                new_X = self.prepare_data(new_X, True)
            elif model.get_name().lower() == 'mobile_net' or model.get_name().lower() == 'resnet' or model.get_name().lower() == 'unet':
                new_X = self.prepare_data(new_X, False)


        title = model.get_name() + '_' + time.strftime("%Y_%m_%d_%H_%M_%S")

        validation_percentage = params['validation_split']

        repetitions_scores = []
        repetitions_scores_new_data = []
        matrix_orig = []
        matrix_predicted = []
        for i in range(params['repetitions']):
            architecture = model.get_model()

            test_results = []
            test_orig = []

            test_results_new_data = []
            test_orig_new_data = []

            if self.get_params()['verbose']:
                print('Repetition: ' + str(i+1) + '\n')

            import random
            seed = random.randint(1, 1000)

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=params['holdout_test_split'], random_state=seed)  # We randomize split too


            # Class weight if there are unbalanced classes
            from sklearn.utils import class_weight
            # class_weight = class_weight.compute_class_weight('balanced',numpy.unique(Y), Y)
            sample_weight = class_weight.compute_sample_weight(class_weight='balanced', y=Y_train)

            if (self.get_params()['problem_type'] == 'classification'): #and not (params['use_distillery'] and model.get_name().lower() != 'distillery_network'):
                    Y_train = keras.utils.to_categorical(Y_train, num_classes=len(numpy.unique(Y)))
                    Y_test = keras.utils.to_categorical(Y_test, num_classes=len(numpy.unique(Y)))

            callbacks_list = []
            callbacks_list.append(
                ModelCheckpoint(title + "_holdout_weights_improvement.hdf5", monitor='val_loss',
                                verbose=params['verbose'], save_best_only=True, mode='min'))
            callbacks_list.append(
                EarlyStopping(monitor='val_loss', min_delta=params['min_delta'], patience=params['patience'],
                              verbose=params['verbose'], mode='min'))

            history = LossHistory((X_train, Y_train))
            callbacks_list.append(history)

            # Fit the architecture
            architecture.fit(X_train, Y_train, epochs=params['epochs'], batch_size=params['batch'],
                             validation_split=validation_percentage,
                             callbacks=callbacks_list,
                             verbose=params['verbose'], sample_weight=sample_weight)

            #import matplotlib.pyplot as plt
            #plt.plot(history.losses[:])
            #plt.plot(history.accuracies[:])
            #plt.title('model loss and acc')
            #plt.ylabel('value')
            #plt.xlabel('batch')
            #plt.legend(['loss','acc'], loc='upper left')
            #plt.show()

            # Data Augmentation
            if params['augmentation']:
                from keras.preprocessing.image import ImageDataGenerator

                # Initialize Generator
                datagen = ImageDataGenerator(vertical_flip=True)

                # Fit parameters from data
                datagen.fit(X_train)

                # Fit new data
                architecture.fit_generator(datagen.flow(X_train, Y_train, batch_size=params['batch']))

            model_json = architecture.to_json()
            with open(title + '_model.json', "w") as json_file:
                json_file.write(model_json)
            architecture.save_weights(title + '_weights.h5')
            file = open(title + '_seed.txt', 'a+')
            file.write('Repetition: ' + str(i+1) + ' , seed: ' + str(seed) + '\n')
            file.close()

            # Evaluate the architecture
            print('Evaluation metrics\n')
            scores = architecture.evaluate(X_test, Y_test, verbose=params['verbose'])

            Y_predicted = numpy.argmax(architecture.predict(X_test, batch_size=params['batch']), axis=1)
            test_results += Y_predicted.tolist()
            test_orig += numpy.argmax(Y_test, axis=1).tolist()

            matrix_orig.extend(test_orig)
            matrix_predicted.extend(Y_predicted)

            if params['problem_type'].lower() == 'classification':
                scores += self.classification_metrics(numpy.array(test_results), test_orig, numpy.unique(Y))
            else:
                scores += scores.tolist() + self.regression_metrics(numpy.array(test_results), test_orig)

            if params['use_new_data']:
                scores_new_data = architecture.evaluate(new_X, new_Y, verbose=params['verbose'])
                new_Y_predicted = numpy.argmax(architecture.predict(new_X, batch_size=params['batch']), axis=1)
                test_results_new_data += new_Y_predicted.tolist()
                test_orig_new_data += numpy.argmax(new_Y, axis=1).tolist()

                if params['problem_type'].lower() == 'classification':
                    scores_new_data += self.classification_metrics(numpy.array(test_results_new_data), test_orig_new_data, numpy.unique(Y))
                else:
                    scores_new_data += self.regression_metrics(numpy.array(test_results_new_data), test_orig_new_data)

                repetitions_scores_new_data.append(scores_new_data)
                
            import csv

            with open(title + '_test_output.csv', 'w+') as file:
                wr = csv.writer(file)
                wr.writerow(test_results)
            file.close()

            repetitions_scores.append(scores)

        global_score_mean = numpy.mean(repetitions_scores, axis=0)
        global_score_std = numpy.std(repetitions_scores, axis=0)

        if params['problem_type'].lower() == 'classification':
            architecture.metrics_names += ["sklearn_acc", "tn", "fp", "fn", "tp", "precision", "recall",
                                           "specificity", "f1", "auc_roc", "k"]
        else:
            architecture.metrics_names += ["mae", "r2"]


        file = open(title + '_results.txt', 'w')
        for count in range(len(architecture.metrics_names)):
            file.write(str(architecture.metrics_names[count]) + ": " + str(
                numpy.around(global_score_mean[count], decimals=4)) + chr(177) +str(numpy.around(global_score_std[count],decimals=4))+ '\n')
        file.close()

        matrix = self.compute_confusion_matrix(matrix_orig, matrix_predicted)
        numpy.savetxt(title + '_total_confusion_matrix.txt', matrix, fmt='% 4d')
        if params['problem_type'].lower() == 'classification':
            total_metrics_names = ["acc", "tn", "fp", "fn", "tp", "recall",
                                   "specificity", "precision", "f1", "Negative predictive value", "False positive rate",
                                   "False negative rate", "False discovery rate"]
            total_metrics_scores = self.compute_total_classification_metrics(matrix)
            file = open(title + '_total_results.txt', 'w')
            for i in range(len(total_metrics_names)):
                file.write(str(total_metrics_names[i]) + ": " + str(
                    numpy.around(total_metrics_scores[i], decimals=4)) + '\n')
            file.close()

        if params['use_new_data']:
            global_score_mean_new_data = numpy.mean(repetitions_scores_new_data, axis=0)
            global_score_std_new_data = numpy.std(repetitions_scores_new_data, axis=0)

            file = open(title + '_results_new_data.txt', 'w')
            for count in range(len(architecture.metrics_names)):
                file.write(str(architecture.metrics_names[count]) + ": " + str(
                    numpy.around(global_score_mean_new_data[count], decimals=4)) + chr(177) + str(
                    numpy.around(global_score_std_new_data[count], decimals=4)) + '\n')
            file.close()