from unittest import TestCase


class Test(TestCase):
    """
    Test the whole project
    """

    def test_macro_add(self):
        """
        Test adding layers Macro
        """
        from Shell.Macro import Macro

        list_inputs = ['add conv2 (2,2)', 'add maxpool3 (2,2,1)', 'add dense 2']
        layer_list = []
        dimension_list = []
        macro = Macro.getInstance()
        for input in list_inputs:
            macro.read_prompt(input, layer_list, dimension_list)
            macro.run(layer_list, dimension_list)

        self.assertEqual(layer_list, ['conv2', 'maxpool3', 'dense'])
        self.assertEqual(dimension_list, [(2, 2), (2, 2, 1), 2])

    def test_macro_remove(self):
        """
        Test removing layers Macro
        """
        from Shell.Macro import Macro

        list_inputs = ['add conv2 (2,2)', 'add dense 4', 'add dense 2', 'add maxpool3 (2,2,1)', 'add dense 2']
        layer_list = []
        dimension_list = []
        macro = Macro.getInstance()
        for input in list_inputs:
            macro.read_prompt(input, layer_list, dimension_list)
            macro.run(layer_list, dimension_list)

        macro.read_prompt('remove dense 2', layer_list, dimension_list)
        macro.run(layer_list, dimension_list)

        self.assertEqual(layer_list, ['conv2', 'dense', 'maxpool3', 'dense'])
        self.assertEqual(dimension_list, [(2, 2), 4, (2, 2, 1), 2])

    def test_oasis(self):
        """
        Test oasis dataset
        """
        from Problems.OASIS import OASIS
        oasis = OASIS()
        self.assertEqual(oasis.get_classes(), [0, 1])
        self.assertEqual(oasis.get_name(), 'OASIS')
        self.assertEqual(oasis.get_data()['X'].shape, (436, 256, 256, 1)) # 436 images, 256x256 pixels, 1 colour channel
        self.assertEqual(oasis.get_data()['Y'].shape, (436,))

    def test_adni(self):
        """
        Test adni dataset
        """
        from Problems.ADNI import ADNI
        adni= ADNI()
        self.assertEqual(adni.get_classes(), [0, 1])
        self.assertEqual(adni.get_name(), 'ADNI')
        self.assertEqual(adni.get_data()['X'].shape, (1743, 256, 256, 1)) # 436 images, 256x256 pixels, 1 colour channel
        self.assertEqual(adni.get_data()['Y'].shape, (1743,))