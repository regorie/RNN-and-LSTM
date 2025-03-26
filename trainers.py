import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0
        self.iter_per_epoch = None


    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // batch_size


        self.ppl_list = []
        self.eval_interval = eval_interval
        self.iter_per_epoch = max_iters

        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):

            for iters in tqdm(range(max_iters)):
                shuffled_idx = np.arange(data_size)
                np.random.shuffle(shuffled_idx)
                xs = xs[shuffled_idx]
                ts = ts[shuffled_idx]
                batch_x, batch_t = xs[:batch_size], ts[:batch_size]


                # 기울기를 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                grads = model.backward()

                optimizer.update(grads, model.params)
                self.ppl_list.append(loss)
                #total_loss += loss
                #loss_count += 1

                # record loss
                #if iters % eval_interval == 0:
                #    ppl = total_loss / loss_count
                #    elapsed_time = time.time() - start_time
                    #print('| 에폭 %d |  반복 %d / %d | 퍼플렉서티 %.2f'
                    #      % (self.current_epoch + 1, iters + 1, max_iters, ppl))
                    #print("time taken: ", elapsed_time)
                #    self.ppl_list.append(float(ppl))
                #    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, file_name, ylim=None, interval=1, x_label='epochs', y_label='perplexity'):
        x = np.arange(0, len(self.ppl_list), interval)

        if ylim is not None:
            plt.ylim(*ylim)

        plt.plot(np.arange(len(x)), np.array(self.ppl_list)[x])

        plt.xlabel(x_label, fontsize=18)
        plt.ylabel(y_label, fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_name)
        else:
            plt.show()