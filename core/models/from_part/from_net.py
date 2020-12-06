class FromNet:
    def __init__(self, base):
        self.base = base

    def forward(self):
        return self.base.forward()

