from Shell.Abstract.Command import Command


class Remove_Layer(Command):

    def execute(self, layer_list, dimension_list, params):
        (layer, value) = params
        value = int(value)
        ocurrencies = 0
        for i in range(len(layer_list)):
            if layer_list[i] == layer:
                ocurrencies += 1
            if ocurrencies == value:
                del layer_list[i]
                del dimension_list[i]
                break