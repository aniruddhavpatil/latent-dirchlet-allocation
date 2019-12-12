import os
from glm import GLM
import numpy as np
import random
from matplotlib import pyplot as plt
import csv

def read_data(dataset):
    # Read data
    data = {}
    data_dir_path = os.path.join(os.getcwd(), dataset)
    index_file = None
    for file in os.listdir(data_dir_path):
        if file == 'index.csv':
            index_file = open(os.path.join(data_dir_path, 'index.csv'), 'r').readlines()
        else:
            file_words = open(os.path.join(data_dir_path, file), 'r').read().split()
            data[file] = file_words

    # Read labels
    labels = {}
    for line in index_file:
        line = line.strip().split(',')
        labels[line[0]] = int(line[1])

    # Build vocabulary
    V = {}
    word_count = 0
    for file in data:
        for word in data[file]:
            word_count += 1
            if word not in V:
                V[word] = 1
            else:
                V[word] += 1

    D = []
    w_n = []
    d_n = []
    z_n = []
    i = 0
    for file in data:
        file_v = [0] * len(V)
        for word in data[file]:
            w_n.append(V[word])
            d_n.append(i)
            z_n.append(np.random.randint(0,20))
            file_v[list(V.keys()).index(word)] += 1
        i+=1
        file_v.append(labels[file])
        D.append(file_v)
    return np.asarray(D), word_count, V, w_n, d_n, z_n


class Dataset:
    def __init__(self, dataset_name):
        self.name = dataset_name

    def read(self, dataset_name):
        D, word_count, V, w_n, d_n, z_n = read_data(dataset_name)
        self.data, self.labels = D[:, :-1], D[:, -1]
        self.word_count = word_count
        self.V = V
        self.w = np.asarray(w_n)
        self.d = np.asarray(d_n)
        self.z = np.asarray(z_n)


class CollapsedGibbsSampler:
    def __init__(self, dataset):
        self.K = 20
        if dataset == 'artificial':
            self.K = 2
        self.V = dataset.V
        self.alpha = 5 / self.K
        self.beta = 0.01
        self.n_iters = 500
        self.data = dataset.data
        self.word_count = dataset.word_count
        self.doc_count, self.vocab_size = self.data.shape
        self.n = np.arange(self.word_count)
        self.d = dataset.d
        self.w = dataset.w
        self.z = dataset.z
        # self.d = np.random.randint(0, self.doc_count, size=self.word_count)
        # self.w = np.random.randint(0, self.vocab_size, size=self.word_count)
        # self.z = np.random.randint(0, self.K, size=self.word_count)
        random.shuffle(self.n)
        self.Cd = np.zeros((self.doc_count, self.K))
        self.Ct = np.zeros((self.K, self.vocab_size))

        for index in range(len(self.w)):
            self.Ct[self.z[index] , self.w[index]] += 1
            self.Cd[self.d[index] , self.z[index]] += 1

        # print(self.Cd, self.Ct)
        self.P = np.zeros(self.K)

    def sample(self):
        for i in range(self.n_iters):
            print('Iteration number:', i)
            for n in range(self.word_count):
                index = self.n[n]
                word = self.w[index]
                topic = self.z[index]
                doc = self.d[index]
                self.Cd[doc, topic] -= 1
                self.Cd[doc, topic] = 0 if self.Cd[doc, topic] < 0 else self.Cd[doc, topic]

                self.Ct[topic, word] -= 1
                self.Ct[topic, word] = 0 if self.Ct[topic, word] < 0 else self.Ct[topic, word]
                self.P = [((self.Ct[k, word] + self.beta) * (self.Cd[doc, k] + self.alpha)) /
                          ((self.vocab_size * self.beta + np.sum(self.Ct[k, :])) *
                           (self.K * self.alpha + np.sum(self.Cd[doc, :]))) for k in range(self.K)]
                self.P = np.asarray(self.P)
                self.P /= np.sum(self.P)
                topic = np.random.choice(self.K, 1, p=self.P)
                self.z[index] = topic
                self.Cd[doc, topic] += 1
                self.Ct[topic, word] += 1

        words = []
        word_array = list(self.V.keys())
        for i in self.Ct:
            word_idx = sorted(range(len(i)), key=lambda x: i[x], reverse=True)[:5]
            words.append([word_array[i] for i in word_idx])

        with open('topicwords.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(words)
        # for i in words:
        #     print(i)
        return self.z, self.Cd, self.Ct


def createDataset(Cd, log_dataset, cgs):
    K = cgs.K
    alpha = cgs.alpha
    cgs_dataset = Dataset('cgs_' + log_dataset.name)
    data = [[
        (Cd[i, k] + alpha) / (K * alpha + np.sum(Cd[i, :]))
        for k in range(K)]
        for i in range(Cd.shape[0])
    ]
    data = np.asarray(data)
    cgs_dataset.data = data
    cgs_dataset.labels = log_dataset.labels
    return cgs_dataset

def plot(bow, cgs, dataset):

    std_bow = np.std(bow, axis=0)
    acc_bow = np.mean(bow, axis=0)
    std_bow /= len(bow) ** 0.5

    std_cgs = np.std(cgs, axis=0)
    acc_cgs = np.mean(cgs, axis=0)
    std_cgs /= len(cgs) ** 0.5

    plt.clf()
    plt.xlabel('Portion of training data used from the ' + dataset + ' dataset')
    plt.ylabel('Mean Accuracy')
    plt.title('Logistic Regression')
    plt.xticks([0.1 * x for x in range(1, 11)])
    plt.errorbar([x * 0.1 for x in range(1, 11)],
                 acc_bow,
                 std_bow,
                 label='BoW',
                 capsize=5,
                 elinewidth=1,
                 ecolor='red',
                 color='black'
                 )
    plt.errorbar([x * 0.1 for x in range(1, 11)],
                 acc_cgs,
                 std_cgs,
                 label='LDA',
                 capsize=5,
                 elinewidth=1,
                 ecolor='blue',
                 color='brown'
                 )
    plt.legend()
    plt.savefig(dataset + '.png')

if __name__ == '__main__':
    # dataset = 'artificial'
    # dataset = 'artificial'
    random.seed(270)
    for dataset in ['20newsgroups']:
        bow_dataset = Dataset(dataset)
        bow_dataset.read(dataset)
        logistic_bow = GLM('logistic', 0.01, bow_dataset)
        logistic_bow.evaluate()

        log_dataset = Dataset(dataset)
        log_dataset.read(dataset)
        cgs = CollapsedGibbsSampler(log_dataset)
        _, Cd, _ = cgs.sample()
        cgs_dataset = createDataset(Cd, log_dataset, cgs)
        logistic_cgs = GLM ('logistic', 0.01, cgs_dataset)
        logistic_cgs.evaluate()
        all_error_cgs = logistic_cgs.all_error
        all_error_bow = logistic_bow.all_error

        plot(all_error_bow, all_error_cgs, dataset)


