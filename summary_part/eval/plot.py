import matplotlib.pyplot as plt
import numpy


class Create_fi:

    def __init__(self):
        self.loss_list = None
        self.x_list = []
        # Nomal, TF-IDF, EM-TF-IDFの類似度が入る配列
        self.similarity = [[],[],[]]

    def cros_loss(self, loss, first=False):
        if first:
            self.loss_list = numpy.zeros(len(loss))
        self.loss_list += loss

    def loss_plot(self, size, filename, wordvec_size, ylim=None):
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list / size, label=filename, lw=0.8)
        plt.legend(loc='upper right')
        plt.xlabel('iterations (wordvec_size and hidden_size = ' + str(wordvec_size) + ')')
        plt.ylabel('loss')
        # plt.show()

    def similist(self, similarity, i):
        self.similarity[i].append(similarity)

    def xlist(self, x):
        self.x_list.append(x)
        print(self.x_list)
        print(self.similarity)

    def simi_plot(self, ylim=None):
        for i, simi in enumerate(self.similarity):
            if ylim is not None:
                plt.ylim(*ylim)
            if i == 0:
                label = '5-gram'
                mark = "o"
            elif i == 1:
                label = 'TF-IDF'
                mark = "x"
            else:
                label = 'EM-TF-IDF'
                mark = "<"
            plt.plot(numpy.array(self.x_list), numpy.array(simi), marker=mark, label=label, lw=0.8)
            plt.legend(loc='upper left')

        plt.xlabel('wordvec_size and hidden_size')
        plt.ylabel('similarity')
        # plt.show()

    def save(self, file_name):
        plt.savefig(file_name + '.png')
        plt.close()
