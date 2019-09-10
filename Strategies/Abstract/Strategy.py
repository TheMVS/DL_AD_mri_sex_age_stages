import abc

class Strategy(abc.ABCMeta('ABC', (object,), {'__slots__': ()})):
    'Common base class for all strategies'
    __name = None
    __params = None

    def __init__(self, name, params):
        'Constructor'
        self.__name = name
        self.__params = params

    # Getters

    def get_name(self):
        return self.__name

    def get_params(self):
        return self.__params

    # Setters

    # Other methods

    @abc.abstractmethod
    def fit(self, model, data, new_data):
        'Trains the model with new data'
        pass

    def predict(self, model, data):
        'Predicts classes for new data'
        predictions = model.predict(data['X'])

    def prepare_data(self, X, is_vgg16):
        'Prepares data for VGG16 and Mobile net'
        import cv2
        import numpy as np
        aux = X
        if len(X[0].shape) < 4:
            MEAN_VALUE = np.array([103.939, 116.779, 123.68])  # BGR
            aux = [cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) for img in X]
            aux[:] = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in aux]
            if is_vgg16:
                aux[:] = [img - MEAN_VALUE for img in aux]
        aux = np.array(aux)
        return aux

    def encode_output(self, Y, vocab_size):
        from keras.utils import to_categorical
        import numpy as np
        ylist = list()
        for output in Y:
            encoded = to_categorical(output, num_classes=vocab_size)
            ylist.append(encoded)
        y = np.array(ylist)
        y = y.reshape(Y.shape[0], 1, vocab_size)
        return y

    def multiclass_confusion_matrix(self,Y_test,Y_predicted,labels):
        import numpy as np
        from sklearn.metrics import confusion_matrix

        matrix = confusion_matrix(Y_test, Y_predicted, labels=labels)
        fp = matrix.sum(axis=0) - np.diag(matrix)
        fn = matrix.sum(axis=1) - np.diag(matrix)
        tp = np.diag(matrix)
        tn = matrix.sum() - (fp + fn + tp)

        fp = np.sum(fp)
        fn = np.sum(fn)
        tp = np.sum(tp)
        tn = np.sum(tn)

        return tn, fp, fn, tp


    def classification_metrics(self, Y_predicted, Y_test, labels):
        from sklearn.metrics import confusion_matrix, accuracy_score,log_loss, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score

        if len(labels) <= 2:
            tn, fp, fn, tp = confusion_matrix(Y_test,Y_predicted,labels=labels).ravel()
        else:
            tn, fp, fn, tp = self.multiclass_confusion_matrix(Y_test, Y_predicted, labels=labels)

        specificity = float(tn) / float(tn + fp)

        acc = accuracy_score(Y_test, Y_predicted)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*(precision*recall)/(precision+recall)

        if len(labels) == 2:
            roc = roc_auc_score(Y_test,Y_predicted)
        else:
            roc = 0.5

        k = cohen_kappa_score(Y_test,Y_predicted,labels=labels)

        return [acc, tn, fp, fn, tp, precision, recall, specificity, f1, roc, k]

    def regression_metrics(self, Y_predicted, Y_test):
        from sklearn.metrics import r2_score,mean_absolute_error
        import numpy as np

        Y_test = np.argmax(Y_test, axis=1)

        r2 = r2_score(Y_test,Y_predicted)
        mae = mean_absolute_error(Y_test,Y_predicted)

        return [mae,r2]

    def compute_confusion_matrix(self, test_orig, test_predicted):
        import numpy as np

        num_classes = len(np.unique(test_orig))
        matrix = np.zeros((num_classes,num_classes), int)

        for t1, t2 in zip(test_orig,test_predicted):
            matrix[t1,t2] += 1

        return matrix

    def compute_total_classification_metrics(self, cnf_matrix):
        import numpy as np

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        F1 = 2 * (PPV * TPR) / (PPV + TPR)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)
        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)

        return [ACC, TN, FP, FN, TP, TPR, TNR, PPV, F1, NPV, FPR, FNR, FDR]