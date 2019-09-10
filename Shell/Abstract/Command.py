import abc


class Command(abc.ABCMeta('ABC', (object,), {'__slots__': ()})):

    @abc.abstractmethod
    def execute(self, layer_list, dimension_list, params): pass
