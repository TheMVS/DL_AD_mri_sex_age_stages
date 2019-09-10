from Models.Abstract.Model import Model


class Convolutional(Model):

    def __init__(self, params):
        'Constructor'
        super(Convolutional, self).__init__('Convolutional', params)

    def get_architecture(self, params):
        from keras import Input, models
        from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, BatchNormalization, Flatten

        initial_model = Input(params['input_shape'])
        model = initial_model

        dimension_convs = params['conv_dimension_list']
        conv_layers = params['conv_layer_list']

        for i in range(len(conv_layers)):
            if(conv_layers[i].lower() == 'conv'):
                model = Conv2D(params['batch'], kernel_size=dimension_convs[i],
                               strides=(1, 1),
                               activation=params['activation'])(model)
            elif(conv_layers[i].lower() == 'maxpool'):
                model = MaxPooling2D(pool_size=dimension_convs[i])(model)
            elif (conv_layers[i].lower() == 'drop'):
                model = Dropout(dimension_convs[i])(model)

        model = BatchNormalization(axis=-1)(model)
        model = Flatten()(model)

        # Filter dropouts and dense layers
        dimension_dense = params['dense_dimension_list']
        layers_list = params['dense_layer_list']

        for i in range(len(dimension_dense)):
            if (layers_list[i].lower() == 'dense'):
                model = Dense(dimension_dense[i], activation=params['activation'])(model)
            elif (layers_list[i].lower() == 'drop'):
                model = Dropout(dimension_dense[i])(model)


        model = models.Model(initial_model, model)

        return model