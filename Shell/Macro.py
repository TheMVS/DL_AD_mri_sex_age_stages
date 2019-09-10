from Shell import Singleton
from Shell.Add_Layer import Add_Layer
from Shell.Remove_Layer import Remove_Layer


class Macro:

    __commands = []
    __params = []
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Macro.__instance == None:
            Macro()
        return Macro.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Macro.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Macro.__instance = self

    def read_prompt(self,command,layer_list,dimension_list):
        help_message = 'Commands:\n \thelp: prompts help menu.\n \tadd <layer> <dimension>: adds layer to model.\n \tremove <layer> <position>: removes layer of given position according the set of layers of that kind.\n \tmodel: print model.\n \texit: quit shell.'
        inputs = command.split()
        if len(inputs) > 0:
            if inputs[0] == 'help':
                print(help_message)
            elif inputs[0] == 'model':
                for i in range(len(layer_list)):
                    print(layer_list[i], dimension_list[i])
            else:
                if inputs[0] == 'add':
                    if len(inputs) == 3:
                        self.add_command(Add_Layer())
                        self.add_params((inputs[1], inputs[2]))
                    elif len(inputs) == 2:
                        self.add_command(Add_Layer())
                        self.add_params((inputs[1], None))
                    else:
                        print(help_message)
                if inputs[0] == 'remove':
                    if len(inputs) == 3:
                        self.add_command(Remove_Layer())
                        self.add_params((inputs[1], inputs[2]))
                    else:
                        print(help_message)
        else:
            print(help_message)

    def add_command(self, command):
        self.__commands.append(command)

    def add_params(self,params):
        self.__params.append(params)

    def run(self,layer_list, dimension_list):
        for i in range(len(self.__commands)):
            self.__commands[i].execute(layer_list, dimension_list, self.__params[i])
            del self.__commands[i]
            del self.__params[i]
