import abc



class Model(abc.ABCMeta('ABC', (object,), {'__slots__': ()})):
    'Common base class for all models'
    __name = None
    __params = None
    __model = None  # The Keras Model
    __sklearn_model = None

    def __init__(self, name, params):
        'Constructor'
        self.__name = name
        self.__params = params
        self.set_model()
        self.set_sklearn_model()

    # Getters

    def get_name(self):
        return self.__name

    def get_params(self):
        return self.__params

    def get_model(self):
        self.__model = self.create_model(self.__params)
        return self.__model

    def get_sklearn_model(self):
        self.set_sklearn_model()
        return self.__sklearn_model

    # Setters

    def set_model(self):
        'Creates de specified model'
        if (self.__params['model_path'] is None and self.__name.lower() != 'distillery_network'):
            self.__model = self.create_model(self.get_params())
        else:
            if (self.__name.lower() == 'distillery_network'):
                self.load_model(self.__params['model_path'])
                self.__model.load_weights(self.__params['distillery_weight_path'])
            else:
                self.load_model(self.__params['model_path'])
                if self.__params['weight_path'] is not None:
                    self.__model.load_weights(self.__params['weight_path'])

    def set_sklearn_model(self):
        'Creates de specified model'
        if self.__params['use_sklearn']:
            if (self.__params['sklearn_model'].lower() == 'linear_regression'):
                from sklearn import linear_model
                self.__sklearn_model = linear_model.LinearRegression()
            elif (self.__params['sklearn_model'].lower() == 'xgboost'):
                from sklearn import svm
                if (self.__params['problem_type'].lower() == 'regression'):
                    from xgboost import XGBRegressor
                    self.__sklearn_model = XGBRegressor()
                else:
                    from xgboost import XGBClassifier
                    self.__sklearn_model = XGBClassifier()
            else:
                from sklearn import svm
                if (self.__params['problem_type'].lower() == 'regression'):
                    self.__sklearn_model = svm.SVR(tol=self.__params['min_delta'])
                else:
                    self.__sklearn_model = svm.SVC(tol=self.__params['min_delta'])
    # Other methods

    @abc.abstractmethod
    def get_architecture(self, params):
        'Model Architecture'
        pass

    def create_model(self, params):
        'Creates specified model given some params'
        from keras.utils import multi_gpu_model

        model = self.get_architecture(params)

        print(model.summary())

        if params['device'].lower() == 'gpu':
            model = multi_gpu_model(model, gpus=params['gpus'])

        if (params['problem_type'].lower() == 'regression'):
            model.compile(loss='mse', optimizer=params['optimizer'], metrics=params['metrics'])
        else:
            model.compile(loss='binary_crossentropy', optimizer=params['optimizer'],
                          metrics=params['metrics'])

        print(model.summary())

        return model

    def load_model(self, model_file_path):
        'Loads model parameters and weights path'
        from keras.models import model_from_json
        self.__model = model_from_json(model_file_path)
