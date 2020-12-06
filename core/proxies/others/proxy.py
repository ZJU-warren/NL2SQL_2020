class ModuleProxy:
    def __init__(self, total, predict_mode=False, batch=1000):
        self.total = total
        self.mode = predict_mode
        self.batch = batch

        # epoch init
        self.start = 0
        self.batch_size = self.total // self.batch

    def run_a_epoch(self):
        if self.mode:
            data_index = list(range(0, self.total))
            self.forward(data_index)

        else:
            while True:
                # calculate the start, end of this batch
                end = min(self.total, self.start + self.batch_size)
                print('[%d, %d)' % (self.start, end))
                data_index = list(range(self.start, end))

                # forward
                self.forward(data_index)

                # update for next batch
                self.start = end % self.total

                if self.start == 0:
                    break

    def forward(self, data_index):
        pass
