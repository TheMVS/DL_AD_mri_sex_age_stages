# -*- coding: utf-8 -*-


class Program():
    'Common base class for all programs'
    __problem = None
    __evaluation_problem = None
    __model = None
    __distillery_model = None
    __strategy = None

    def __init__(self, params):
        'Constructor'
        # params : dictionary
        self.set_problem(params)
        self.set_evaluation_problem(params)
        element = self.__problem.get_data()['X']
        element = element[0]
        params['input_shape'] = element.shape
        self.set_strategy(params)
        self.set_model(params)
        params['distillery_input_shape'] = self.get_model().get_model().output_shape[1:]  # TODO: check
        self.set_distillery_model(params)

    # Getters

    def get_problem(self):
        return self.__problem

    def get_evaluation_problem(self):
        return self.__evaluation_problem

    def get_model(self):
        return self.__model

    def get_distillery_model(self):
        return self.__distillery_model

    def get_strategy(self):
        return self.__strategy

    # Setters

    def set_strategy(self, params):
        if params['use_sklearn']:
            if params['strategy_type'].lower() == 'kfold':
                from Strategies.Sklearn_KFold import Sklearn_KFold
                self.__strategy = Sklearn_KFold(params=params)
            elif params['strategy_type'].lower() == 'loo':
                from Strategies.Sklearn_LOO import Sklearn_LOO
                self.__strategy = Sklearn_LOO(params=params)
            else:
                from Strategies.Sklearn_HoldOut import Sklearn_HoldOut
                self.__strategy = Sklearn_HoldOut(params=params)
        else:
            if params['strategy_type'].lower() == 'kfold':
                from Strategies.KFold import KFold
                self.__strategy = KFold(params=params)
            elif params['strategy_type'].lower() == 'loo':
                from Strategies.LOO import LOO
                self.__strategy = LOO(params=params)
            else:
                from Strategies.HoldOut import HoldOut
                self.__strategy = HoldOut(params=params)

    def set_model(self, params):
        'Return a list with all classes'
        if params['model_type'].lower() == 'full':
            from Models.Full_Connected import Full_Connected
            self.__model = Full_Connected(params=params)
        elif params['model_type'].lower() == 'convolutional':
            from Models.Convolutional import Convolutional
            self.__model = Convolutional(params=params)
        elif params['model_type'].lower() == 'vgg16':
            from Models.VGG16 import VGG16
            self.__model = VGG16(params=params)
        elif params['model_type'].lower() == 'mobile':
            from Models.Mobile_Net import Mobile_Net
            self.__model = Mobile_Net(params=params)
        elif params['model_type'].lower() == 'mobilev2':
            from Models.Mobile_NetV2 import Mobile_NetV2
            self.__model = Mobile_NetV2(params=params)
        elif params['model_type'].lower() == 'resnet':
            from Models.ResNet import ResNet
            self.__model = ResNet(params=params)
        elif params['model_type'].lower() == 'unet':
            from Models.UNet import UNet
            self.__model = UNet(params=params)
        else:
            from Models.Custom_Network import Custom_Network
            self.__model = Custom_Network(params=params)

    def set_distillery_model(self, params):
        if params['use_distillery']:
            from Models.Distillery_Network import Distillery_Network
            self.__distillery_model = Distillery_Network(params=params)

    def set_problem(self, params):
        'Sets specified data'
        if params['dataset_name'].lower() == 'oasis':
            from Problems.OASIS import OASIS
            self.__problem = OASIS()
        else:
            from Problems.ADNI import ADNI
            self.__problem = ADNI()

    def set_evaluation_problem(self, params):
        if params['use_new_data']:
            if params['new_dataset_name'].lower() == 'oasis':
                from Problems.OASIS import OASIS
                self.__evaluation_problem = OASIS()
            else:
                from Problems.ADNI import ADNI
                self.__evaluation_problem = ADNI()
        else:
            return None


    # Other methods
    def transform_ditillery_data(self, model, X):
        converted = []
        import numpy as np
        for element in X:
            shape = element.shape
            input = np.array(element)
            input = input.reshape(1, shape[0], shape[1], shape[2])
            converted.append(model.predict(input, batch_size=1)[0])
        return {'X': np.array(converted), 'Y': self.get_problem().get_data()['Y']}

    def main(self, params):
        if params['use_new_data']:
            self.get_strategy().fit(self.get_model(), self.get_problem().get_data(),
                                    self.get_evaluation_problem().get_data())
            if params['use_distillery']:
                self.get_strategy().fit(self.get_model(),
                                        self.transform_ditillery_data(self.get_distillery_model().get_model(),
                                                                      self.get_problem().get_data()['X']), self.transform_ditillery_data(self.get_distillery_model().get_model(),
                                                                      self.get_evaluation_problem().get_data()['X']))

        else:
            self.get_strategy().fit(self.get_model(), self.get_problem().get_data(), None)
            if params['use_distillery']:
                self.get_strategy().fit(self.get_model(),
                                        self.transform_ditillery_data(self.get_distillery_model().get_model(),
                                                                      self.get_problem().get_data()['X']), None)


def read_parameters():
    import Config

    problem_type = Config.PROBLEM_TYPE
    dataset_name = Config.DATASET_NAME
    model_type = Config.MODEL_TYPE
    strategy_type = Config.STRATEGY_TYPE

    model_path = Config.MODEL_PATH
    weight_path = Config.WEIGHT_PATH

    repetitions = Config.REPETITIONS
    folds = Config.FOLDS
    holdout_test_split = Config.HOLDOUT_TEST_SPLIT
    validation_split = Config.VALIDATION_SPLIT

    epochs = Config.EPOCHS
    batch = Config.BATCH

    dense_layer_list = Config.DENSE_LAYERS_LIST
    dense_dimension_list = Config.DENSE_DIMENSION_LIST

    conv_layer_list = Config.CONV_LAYERS_LIST
    conv_dimension_list = Config.CONV_DIMENSION_LIST

    custom_layer_list = Config.CUSTOM_LAYERS_LIST
    custom_dimension_list = Config.CUSTOM_DIMENSION_LIST

    use_distillery = Config.USE_DISTILLERY
    distillery_model_path = Config.DISTILLERY_MODEL_PATH
    distillery_weight_path = Config.DISTILLERY_WEIGHT_PATH

    last_layers = Config.LAST_LAYERS

    use_dropout = Config.USE_DROPOUT
    dropout_prob = Config.DROPOUT_PROB

    activation = Config.ACTIVATION
    optimizer = Config.OPTIMIZER

    device = Config.DEVICE
    gpus = Config.GPUS

    metrics = Config.METRICS
    loss = Config.LOSS

    min_delta = Config.MIN_DELTA
    patience = Config.PATIENCE

    augmentation = Config.AUGMENTATION

    use_sklearn = Config.USE_SKLEARN
    sklearn_model = Config.SKLEARN_MODEL

    verbose = Config.VERBOSE

    use_new_data = Config.NEW_DATASET
    new_dataset_name = Config.NEW_DATASET_NAME

    params = {'problem_type': problem_type, 'dataset_name': dataset_name, 'model_type': model_type,
              'strategy_type': strategy_type, 'model_path': model_path, 'weight_path': weight_path,
              'repetitions': repetitions, 'folds': folds, 'holdout_test_split': holdout_test_split, 'validation_split': validation_split,
              'epochs': epochs, 'batch': batch, 'dense_layer_list': dense_layer_list,
              'dense_dimension_list': dense_dimension_list, 'conv_layer_list': conv_layer_list,
              'conv_dimension_list': conv_dimension_list, 'custom_layer_list': custom_layer_list,
              'custom_dimension_list': custom_dimension_list, 'use_distillery': use_distillery,
              'distillery_model_path': distillery_model_path,
              'distillery_weight_path': distillery_weight_path,
              'last_layers': last_layers, 'use_dropout': use_dropout, 'dropout_prob': dropout_prob,
              'activation': activation, 'optimizer': optimizer, 'device': device, 'gpus': gpus, 'metrics': metrics,
              'loss': loss, 'min_delta': min_delta, 'patience': patience, 'verbose': verbose,
              'augmentation': augmentation, 'interactive': False,
              'test': False, 'use_sklearn': use_sklearn, 'sklearn_model': sklearn_model,
              'use_new_data': use_new_data, 'new_dataset_name': new_dataset_name}
    return params


def read_arguments(argv, params):
    error_text = 'Usage: Program.py [options...]\n \n -> Options:\n \t-i\t:\tLaunch shell for interactive mode.\n \t-t\t:\tExecute unitary tests.\n \nTry option -h for help\n'

    try:
        opts, args = getopt.getopt(argv, "hit", ['device='])
    except getopt.GetoptError:
        print(error_text)
        sys.exit()

    for opt, arg in opts:
        if opt == '-h':
            print(error_text)
            sys.exit()
        elif opt == "-t":
            params['test'] = True
        elif opt == "-i":
            params['interactive'] = True


def launch_shell(params):  # TODO: check if none was inserted
    help_message = 'Commands:\n \thelp: prompts help menu.\n \tadd <layer> <dimension>: adds layer to model.\n \tremove <layer> <position>: removes layer of given position according the set of layers of that kind.\n \tmodel: print model.\n \texit: quit shell.'
    params['model_type'] = 'custom'
    print('Running interactive mode:')
    problem_type = input("Problem type (regression or classification): ")
    if problem_type:
        params['problem_type'] = problem_type
    else:
        print('Using problem type from config file.')
    dataset_name = input("Choose data set (ADNI or OASIS): ")
    if dataset_name:
        params['dataset_name'] = dataset_name
    else:
        print('Using dataset from config file.')
    strategy_type = input("Choose evaluation strategy (KFold or HoldOut): ")
    if strategy_type:
        params['strategy_type'] = strategy_type
        if strategy_type.lower() == "kfold":
            folds = input("Set number of folds: ")
            if folds:
                params['folds'] = int(folds)
            else:
                print('Using folds from config file.')
        elif strategy_type.lower() == "holdout":
            split = input("Set test split [0,1]: ")
            if split:
                params['holdout_test_split'] = float(split)
            else:
                print('Using test split from config file.')
            split = input("Set validation split [0,1]: ")
            if split:
                params['validation_split'] = float(split)
            else:
                print('Using validation split from config file.')
        else:
            print('Invalid Strategy. Using evaluation strategy from config file.')
    else:
        print('Using evaluation strategy from config file.')
    data_augmentation = input("Use data augmentation? (Yes,True): ")
    if data_augmentation.lower() == 'True':
        params['augmentation'] = True
    elif data_augmentation.lower() == 'False':
        params['augmentation'] = False
    else:
        print('Using value from config file.')
    seed = input("Set seed: ")
    if seed:
        params['seed'] = seed
    else:
        print('Using seed from config file.')
    json_path = input("Insert model .json if needed: ")
    if json_path:
        params['model_path'] = json_path
        h5_path = input("Insert model weight .h5 file if needed: ")
        if h5_path:
            params['weight_path'] = h5_path
    else:
        print('Starting shell')

        from Shell.Macro import Macro
        macro = Macro.getInstance()
        layer_list = []
        dimension_list = []
        print('------------------------ Shell ------------------------')
        print(help_message)
        print('-------------------------------------------------------')
        command = ''
        while (command.lower().replace(" ", "") != 'exit'):
            command = input('>> ').lower()
            macro.read_prompt(command, layer_list, dimension_list)
            macro.run(layer_list, dimension_list)

        params['custom_layer_list'] = layer_list
        params['custom_dimension_list'] = dimension_list

        print('')
        print('Leaving Shell')
        print('')

        distillery = input("Use distillery (y/n): ")
        if distillery == 'y':
            params['use_distillery'] = True

            json_path = input("Insert distillery model .json: ")
            if json_path:
                params['distillery_model_path'] = json_path
            h5_path = input("Insert distillery model weight .h5: ")
            if h5_path:
                params['distillery_weight_path'] = h5_path


if __name__ == '__main__':

    parameters = read_parameters()

    import sys, getopt

    if (len(sys.argv) > 0):
        read_arguments(sys.argv[1:], parameters)

    if parameters['interactive']:
        launch_shell(parameters)

    program = Program(parameters)
    program.main(parameters)

    if parameters['test']:
        import unittest
        from io import StringIO
        from Tests.Test import Test

        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream)
        result = runner.run(unittest.makeSuite(Test))
        print('Tests run ', result.testsRun)
        print('Errors ', result.errors)
        print(result.failures)
        stream.seek(0)
        print('Test output\n', stream.read())
