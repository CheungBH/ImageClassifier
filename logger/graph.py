#-*-coding:utf-8-*-
import matplotlib.pyplot as plt
import os


class GraphSaver:
    def __init__(self, save_dir, metrics, phases=("train", "val")):
        self.save_dir = os.path.join(save_dir, "graph")
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics = metrics
        self.phases = phases

    def draw_graph(self, epoch_ls, train_ls, val_ls, name):
        ln1, = plt.plot(epoch_ls, train_ls, color='red', linewidth=3.0, linestyle='--')
        ln2, = plt.plot(epoch_ls, val_ls, color='blue', linewidth=3.0, linestyle='-.')
        plt.title("{}".format(name))
        plt.legend(handles=[ln1, ln2], labels=['train_{}'.format(name), 'val_{}'.format(name)])
        ax = plt.gca()
        ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
        ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
        plt.savefig(os.path.join(self.save_dir, "{}.jpg".format(name)))
        plt.cla()

    def process(self, epoch_ls, best_recorder):
        for idx in range(len(self.metrics)):
            self.draw_graph(epoch_ls, best_recorder["train"][idx], best_recorder["val"][idx], self.metrics[idx])

