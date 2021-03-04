import numpy
from matplotlib import pyplot as plt
import os

EXP_DIR = 'experiments/bi'
OUT_DIR = 'out/acc/bi'

FNs = 'nt2_cutout'
MSG = 'nt2_cutout'
train_acc=[49.997216, 50.02088, 50.485802, 54.56431, 59.08686, 61.528396, 62.866091, 63.802895, 64.546214, 65.313196, 66.286192, 66.687778, 67.186804, 67.974666, 68.10064, 68.441676, 68.956013, 69.325585, 69.590061, 70.085607, 70.421075, 70.81779, 70.980651, 71.385718, 71.750418, 71.769905, 72.190284, 72.783268, 72.907155, 73.182767,  73.3,  73.4,  73.8, 74.2]
test_acc=[49.969621, 49.969621, 53.150465, 57.95747, 65.11258, 66.161544, 67.775197, 65.058971, 55.997141, 68.550751, 72.274839, 71.152609, 72.46426, 66.98356, 71.329521, 71.695854, 71.32416, 72.671551, 70.096497, 71.149035, 73.091494, 73.213009, 72.006791, 69.966047, 73.302359, 74.515726, 70.814868, 73.187991, 70.518227, 72.5,  72.2,  71.658327,  70.6, 71]

train_loss=numpy.array([1.019221, 0.97472, 0.973878, 0.960823, 0.930509, 0.908547, 0.892166, 0.880761, 0.867106, 0.86006, 0.847706, 0.839716, 0.832095, 0.821428, 0.817737, 0.810673, 0.802846, 0.79695, 0.791061, 0.785499, 0.777537, 0.772215, 0.76586, 0.760071, 0.754224, 0.751233, 0.7435, 0.73437, 0.731037, 0.726345, 0.724, 0.721, 0.718, 0.71])-0.0
test_loss=[0.704459, 0.693506, 0.689755, 0.692785, 0.628882, 0.634083, 0.60628, 0.633471, 0.804599, 0.603262, 0.547977, 0.563568, 0.54368, 0.61281, 0.583598, 0.582377, 0.608032, 0.562793, 0.600985, 0.600157, 0.570333, 0.579149, 0.550868, 0.601351, 0.584025, 0.583546, 0.602254, 0.588483, 0.58496, 0.60809, 0.607, 0.613, 0.605, 0.615]
def main():
    epochs = range(0,len(train_acc))
    path=os.path.join(OUT_DIR,MSG)
    fig1, ax = plt.subplots()
    ax.plot(epochs, train_acc, label='train acc')
    ax.plot(epochs, test_acc, label='test acc')
    ax.legend()
    ax.set_title('top test acc: %.3f' % max(test_acc), fontsize=10)
    ax.set(xlabel='epochs', ylabel='acc')
    fig1.savefig(os.path.join(path, f"acc.png"))

    fig3, ax = plt.subplots()
    ax.plot(epochs, train_loss, label='train loss')
    ax.plot(epochs, test_loss, label='test loss')
    ax.legend()
    ax.set(xlabel='epochs', ylabel='loss')
    fig3.savefig(os.path.join(path, f"loss.png"))

    print("------------------------------------")
    print(MSG)
    print("train_acc:", train_acc[:30])
    print("test_acc:", test_acc[:30])
    print("train_loss:", train_loss[:30])
    print("test_loss:", test_loss[:30])
    print("------------------------------------")

    plt.clf()
    plt.cla()



# def save_results_old(epochs, train_acc, test_acc, train_loss, test_loss, path='./out/acc', top_test_acc=0.0, msg = ''):
#     length = min(len(train_acc),len(train_loss),len(train_acc),len(test_loss),len(epochs))
#     train_acc, train_loss, test_acc, test_loss, epochs = train_acc[:length], train_loss[:length], test_acc[:length], test_loss[:length], epochs[:length]
#
#     fig1, ax = plt.subplots()
#     ax.plot(epochs, train_acc)
#     ax.set(xlabel='epochs', ylabel='train acc')
#     fig1.savefig(os.path.join(path, f"train_acc.png"))
#
#     fig2, ax = plt.subplots()
#     ax.plot(epochs, test_acc)
#     ax.set(xlabel='epochs', ylabel='test acc')
#     ax.set_title('top test acc: %.3f' % top_test_acc, fontsize=10)
#     fig2.savefig(os.path.join(path, f"test_acc.png"))
#
#     fig3, ax = plt.subplots()
#     ax.plot(epochs, train_loss)
#     ax.set(xlabel='epochs', ylabel='train loss')
#     fig3.savefig(os.path.join(path, f"train_loss.png"))
#
#     fig4, ax = plt.subplots()
#     ax.plot(epochs, test_loss)
#     ax.set(xlabel='epochs', ylabel='test loss')
#     fig4.savefig(os.path.join(path, f"test_loss.png"))
#
#     plt.clf()
#     plt.cla()



if __name__ == '__main__':
    main()