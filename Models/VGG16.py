from Models.Abstract.Model import Model


class VGG16(Model):

    def __init__(self, params):
        'Constructor'
        super(VGG16, self).__init__('VGG16', params)

    def get_architecture(self, params):
        from keras import models
        from keras.applications import vgg16
        from keras.layers import Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten, Conv2D, Conv3D, MaxPooling3D, AveragePooling3D, Softmax, AveragePooling2D

        # input_shape = params['input_shape'] #TODO: check_dims

        initial_model = vgg16.VGG16()  # TODO: tal vez incluir image_data_format='channels_last'

        for i in range(len(initial_model.layers) - 1 - params['last_layers']):
            initial_model.layers[i].trainable = False

        model = initial_model.layers[
            len(initial_model.layers) - 1 - params['last_layers']].output  # assuming you want the 3rd layer from the last

        if not params['use_sklearn']:

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
        else:
            model = Flatten()(model)

        model = models.Model(initial_model.input, model)

        return model
