from Shell.Abstract.Command import Command


class Add_Layer(Command):
    def execute(self, layer_list, dimension_list, params):
        (layer, value) = params
        layer_list.append(layer)
        if layer == 'dense':
            dimension_list.append(int(value))
        elif layer != 'flatten' and layer != 'norm':
            dimension_list.append(tuple(map(int, value[1:-1].split(','))))
        else:
            dimension_list.append(value)