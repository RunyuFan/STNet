import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from torch.autograd import Variable
# from resnet_lulc import ResNet18, ResNet34, ResNet50, ResNet101
import argparse
# from ResNext import resnext50_32x4d
# from MSDnet import MSDNet
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

import time
import torch
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from dataloader import Mydataset
import argparse
from thop import profile
from models.model_fuse import STNet, CheckinNet, ImgNet
import torchvision.models as models
# from CB_loss import CB_loss
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from Multimodel_train import MultiModalNet, FCViewer

# sklearn.metrics.cohen_kappa_score(y1, y2, labels=None, weights=None, sample_weight=None)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams["font.family"] = "Times New Roman"

def plot_confusion_matrix(cm,labels, title='Confusion Matrix of PDNet'):
    # font1 = {'family': 'Times New Roman',
    #          'size':50}
    # font2 = {'family': 'Times New Roman',
    #          'size':35}
    # font3 = {'family': 'Times New Roman'}
    plt.imshow(cm)   #  , interpolation='nearest', cmap=plt.cm.binary
    # plt.title(title,fontsize=80) # ,fontfamily='Times New Roman')
    # plt.colorbar().ax.tick_params(labelsize=50)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=45,fontsize=150) # ) # ,fontfamily='Times New Roman')
    plt.yticks(xlocations, labels,fontsize=150) # ,fontfamily='Times New Roman')
    plt.ylabel('True label',fontsize=150) # ,fontfamily='Times New Roman')
    plt.xlabel('Predicted label',fontsize=150) # ,fontfamily='Times New Roman')



def draw(y_true,y_pred,labels):
    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(y_true, y_pred)
    # np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(60, 60), dpi=300)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.3f" % (c,), color='red', fontsize=250, va='center', ha='center') # ) # ,fontfamily='Times New Roman') 50
        else:
            plt.text(x_val, y_val, 0, color='red', fontsize=250, va='center', ha='center') # ) # ,fontfamily='Times New Roman') 50
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15) # 0.15

    plot_confusion_matrix(cm_normalized, labels, title='Confusion matrix using both')
    # show confusion matrix
    plt.savefig('Confusion_STNet.png', format='png')
    plt.close()


class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def main(args):
    # Create model
    # if not os.path.exists(args.model_path):
    #     os.makedirs(args.model_path)

    train_txt='data\\train_6_10.txt'
    val_txt='data\\val_2_10.txt'
    test_txt='data\\test_2_10.txt'

    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),
                                                # torchvision.transforms.RandomHorizontalFlip(p=0.3),
                                                # torchvision.transforms.RandomVerticalFlip(p=0.3),
                                                # torchvision.transforms.RandomCrop(size=256),
                                                # torchvision.transforms.RandomRotation(180),
                                                torchvision.transforms.ToTensor()
                                                ])
    val_transform=torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),
                                                torchvision.transforms.ToTensor()])
    test_transform=torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),
                                                torchvision.transforms.ToTensor()])

    train_dataset=Mydataset(txt=train_txt,transform=train_transform)
    val_dataset=Mydataset(txt=val_txt,transform=test_transform)
    test_dataset=Mydataset(txt=test_txt,transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,pin_memory=True)
    val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False,pin_memory=True)

    print("Train numbers:{:d}".format(len(train_dataset)))
    print("Val numbers:{:d}".format(len(val_dataset)))
    print("Test numbers:{:d}".format(len(test_dataset)))


    # model=HRRS_model(n_class=2,dim=64)
    model2 = torch.load('.\\model-UIS\\Mixer_big-6-2-2.pth')  # best_val_model_multi_4_10_POI best_val_model_multi_4_10_Floor_nopretrain
    # STNet-8-1-1.pth
    # Mixer-6-2-2.pth
    # Mixer_base-6-2-2.pth
    # CheckinNet-8-1-1.pth
    # ImgNet-8-1-1.pth
    # Mixer_small-6-2-2.pth
    # inputs = torch.randn(1, 3, 256, 256).to(device)
    # in_msi = torch.randn(1, 3, 256, 256).to(device)
    # in_tpd = torch.randn(1, 120).to(device)
    # flops, params = profile(model2, (inputs, in_msi, in_tpd, ))
    # print('flops: ', flops, 'params: ', params)
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('model2 parameters:', sum(p.numel() for p in model2.parameters() if p.requires_grad))

    model2 = model2.to(device)
    # model3 = model3.to(device)
    print('start eval')

    best_acc_1 = 0.
    best_acc_2 = 0.
    best_acc_3 = 0.

    # model1.eval()
    model2.eval()
    # model3.eval()
    true_label = []
    pred_label = []
    num_class = 2
    classes = ('Others', 'UIS')  # ('住宅区', '公共服务区域', '商业区', '城市绿地', '工业区')
    class_correct2 = list(0. for i in range(num_class))
    class_total2 = list(0. for i in range(num_class))
    correct_prediction_2 = 0.
    total_2 = 0

    with torch.no_grad():
        for images, msi, checkin, labels in test_loader:
            images = images.to(device)
            # print(images.shape)
            labels = labels.to(device)
            msi = msi.to(device)

            checkin = checkin.to(device)
            checkin = checkin.clone().detach().float()

            _, pred_result = model2(images, msi, checkin)



            _2, pred = torch.max(pred_result, 1)
            pred_label.append(pred)
            true_label.append(labels)
            c2 = (pred == labels).squeeze()
            for label_idx in range(len(labels)):
                label = labels[label_idx]
                class_correct2[label] += c2[label_idx].item()
                class_total2[label] += 1
            total_2 += labels.size(0)
            # add correct
            correct_prediction_2 += (pred == labels).sum().item()

        t_l=torch.cat(true_label,dim=0)
        p_l=torch.cat(pred_label,dim=0)
        t_l=t_l.cpu().numpy()
        p_l = p_l.cpu().numpy()
    # for i in range(args.num_class):
    #     print('Model ResNet50 - Accuracy of %5s : %2f%%: Correct Num: %d in Total Num: %d' % (
    #         classes[i], 100 * class_correct1[i] / class_total1[i], class_correct1[i], class_total1[i]))
    # acc_1 = correct_prediction_1 / total_1
    # print("Total Acc Model ResNet50: %.4f" % (correct_prediction_1 / total_1))
    print('----------------------------------------------------')
    for i in range(2):
        print('Model - Accuracy of %5s : %2f%%: Correct Num: %d in Total Num: %d' % (
            classes[i], 100 * class_correct2[i] / class_total2[i], class_correct2[i], class_total2[i]))
    acc_2 = correct_prediction_2 / total_2
    print("Total Acc Model: %.4f" % (correct_prediction_2 / total_2))
    print('----------------------------------------------------')

    print(t_l, p_l)

    # with open("test_true.txt", 'w') as f:
    #     for i in t_l:
    #         f.write(str(i))
    #         f.write('\n')
    # with open("test_pred_PDNet.txt", 'w') as f:
    #     for i in p_l:
    #         f.write(str(i))
    #         f.write('\n')

    # draw(t_l, p_l, classes)
    print(cohen_kappa_score(t_l, p_l))
    print(accuracy_score(t_l, p_l))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=2, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    # parser.add_argument("--net", default='ResNet50', type=str)
    # parser.add_argument("--depth", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    # parser.add_argument("--num_workers", default=2, type=int)
    # parser.add_argument("--model_name", default='lulc-6-fintune-GID', type=str)
    # parser.add_argument("--model_path", default='./model', type=str)
    # parser.add_argument("--pretrained", default=False, type=bool)
    # parser.add_argument("--pretrained_model", default='./model/ResNet50.pth', type=str)
    args = parser.parse_args()

    main(args)
