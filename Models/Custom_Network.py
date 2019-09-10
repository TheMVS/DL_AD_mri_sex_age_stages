from Models.Abstract.Model import Model


class Custom_Network(Model):

    def __init__(self, params):
        'Constructor'
        super(Custom_Network, self).__init__('Custom_Network', params)

    def get_architecture(self, params):
        from keras import models
        from keras.layers import Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten, Conv2D, Conv3D, MaxPooling3D, Input, AveragePooling3D, AveragePooling2D, Softmax

        initial_model = Input(params['input_shape'])
        model = initial_model

        dimensions = params['custom_dimension_list']
        layers = params['custom_layer_list']

        for i in range(len(layers)):
            if (layers[i].lower() == 'conv3'):
                model = Conv3D(params['batch'], kernel_size=dimensions[i],
                               strides=(1, 1,1),
                               activation=params['activation'])(model)
            elif (layers[i].lower() == 'maxpool3'):
                model = MaxPooling3D(pool_size=dimensions[i])(model)
            elif (layers[i].lower() == 'avgpool3'):
                model = AveragePooling3D(pool_size=dimensions[i])(model)
            elif (layers[i].lower() == 'conv2'):
                model = Conv2D(params['batch'], kernel_size=dimensions[i],
                               strides=(1, 1),
                               activation=params['activation'])(model)
            elif (layers[i].lower() == 'maxpool2'):
                model = MaxPooling2D(pool_size=dimensions[i])(model)
            elif (layers[i].lower() == 'avgpool2'):
                model = AveragePooling2D(pool_size=dimensions[i])(model)
            elif (layers[i].lower() == 'drop'):
                model = Dropout(dimensions[i])(model)
            elif (layers[i].lower() == 'flatten'):
                model = Flatten()(model)
            elif (layers[i].lower() == 'norm'):
                model = BatchNormalization(axis=-1)(model)
            elif (layers[i].lower() == 'dense'):
                model = Dense(dimensions[i], activation=params['activation'])(model)
            elif (layers[i].lower() == 'softmax'):
                model = Softmax()(model)

        model = models.Model(initial_model, model)

        return model
