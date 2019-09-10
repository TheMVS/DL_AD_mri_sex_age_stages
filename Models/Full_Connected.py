from Models.Abstract.Model import Model


class Full_Connected(Model):

    def __init__(self, params):
        'Constructor'
        super(Full_Connected, self).__init__('Full_Connected', params)

    def get_architecture(self, params):
        from keras import Input, models
        from keras.layers import Dense, Dropout, Flatten

        initial_model = Input(params['input_shape'])
        model = Flatten()(initial_model)

        # Filter dropouts and dense layers
        dimension_list = params['dense_dimension_list']
        layers_list = params['dense_layer_list']

        for i in range(len(dimension_list)):
            if(layers_list[i].lower() == 'dense'):
                model = Dense(dimension_list[i], activation=params['activation'])(model)
            elif(layers_list[i].lower() == 'drop'):
                model = Dropout(dimension_list[i])(model)

        model = models.Model(initial_model, model)

        return model
