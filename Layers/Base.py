
class BaseLayer:
    '''
    This is the base layer every other layer inherit some basic properties from to determine if a layer is
    trainable or not and if it is in the testing phase or training phase.
    '''
    def __init__(self):
        self.trainable = False

        self.testing_phase = False