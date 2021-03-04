from matplotlib import pyplot as plt
import os

EXP_DIR = 'experiments'
OUT_DIR = 'out/acc'
# FNs = ['s/img_all_bz26.out', 's/ci_all.txt']
# MSGs = ['img_all_bz26', 'ci_all']
# MODES = ['img', 'cif']

FNs = ['log']
MSGs = ['log']
MODES = ['cif']

def main():
    for fn, msg, mode in zip(FNs, MSGs, MODES):
        epochs = []
        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []
        path = os.path.join(EXP_DIR,fn)
        max_test_acc = 0
        try:
            os.mkdir(os.path.join(OUT_DIR,msg))
        except:
            pass

        if mode in 'cif':
            with open(path, 'r') as f:
                line = f.readline()
                while line:
                    try:
                        if 'Train_acc' in line:
                            train_acc.append(float(line.split('Train_acc')[1][1:11].strip()))
                        elif 'Train_loss' in line:
                            tmp_train_loss = float(line.split('Train_loss:')[1][1:11].strip())
                            train_loss.append(tmp_train_loss)
                        elif 'Valid_loss' in line:
                            tmp_test_loss = float(line.split('Valid_loss:')[1][1:11].strip())
                            test_loss.append(tmp_test_loss)
                        elif 'Valid_acc' in line:
                            tmp_test_acc = float(line.split('Valid_acc')[1][1:11].strip())
                            test_acc.append(tmp_test_acc)
                            if tmp_test_acc > max_test_acc:
                                max_test_acc = tmp_test_acc
                        elif 'Epoch:' in line:
                            epochs.append(int(line.split('Epoch')[1][1:].split('lr')[0].strip()))
                        else:
                            pass
                        line = f.readline()
                    except:
                        print(line)
                f.close()
        elif mode in 'img':
            with open(path, 'r') as f:
                line = f.readline()
                while line:
                    try:
                        if 'Train_acc' in line:
                            train_acc.append(float(line.split('Train_acc')[1][1:11].strip()))
                        elif 'TRAIN Step: 15000' in line:
                            tmp_train_loss = float(line.split('TRAIN Step: 15000 Objs:')[1][1:13].strip())
                            train_loss.append(tmp_train_loss)
                        elif 'VALID Step: 12000' in line:
                            tmp_test_loss = float(line.split('VALID Step: 12000 Objs:')[1][1:13].strip())
                            test_loss.append(tmp_test_loss)
                        elif 'Valid_acc_top1' in line:
                            tmp_test_acc = float(line.split('Valid_acc_top1')[1][1:11].strip())
                            test_acc.append(tmp_test_acc)
                            if tmp_test_acc > max_test_acc:
                                max_test_acc = tmp_test_acc
                        elif 'Epoch:' in line:
                            epochs.append(int(line.split('Epoch')[1][1:].split('lr')[0].strip()))
                        else:
                            pass
                        line = f.readline()
                    except:
                        print(line)
                f.close()

        save_results(epochs=epochs, train_acc=train_acc, test_acc=test_acc,
                     train_loss=train_loss, test_loss=test_loss,
                     path=os.path.join(OUT_DIR,msg), top_test_acc=max_test_acc)


def save_results(epochs, train_acc, test_acc, train_loss, test_loss, path='./out/acc', top_test_acc=0.0):
    length = min(len(train_acc),len(train_loss),len(train_acc),len(test_loss),len(epochs))
    train_acc, train_loss, test_acc, test_loss, epochs = train_acc[:length], train_loss[:length], test_acc[:length], test_loss[:length], epochs[:length]

    fig1, ax = plt.subplots()
    ax.plot(epochs, train_acc)
    ax.plot(epochs, test_acc)
    ax.set(xlabel='epochs', ylabel='acc')
    fig1.savefig(os.path.join(path, f"acc.png"))

    fig3, ax = plt.subplots()
    ax.plot(epochs, train_loss)
    ax.plot(epochs, test_loss)
    ax.set(xlabel='epochs', ylabel='loss')
    fig3.savefig(os.path.join(path, f"loss.png"))

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