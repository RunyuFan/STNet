import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.io as io
import torchvision
from torchvision import transforms as T
import numpy as np
def MyLoader(path,type):
    if type=='img':
        return Image.open(path).convert('RGB')
    elif type=='vector':
        return np.load(path, allow_pickle=True)
    elif type=='msi':
        return io.loadmat(path)['msi']


class Mydataset(Dataset):
    def __init__(self,txt,transform=None, target_transform=None, loader=MyLoader):
        with open(txt,'r') as fh:
            file=[]
            for line in fh:
                line=line.strip('\n')
                line=line.rstrip()
                words=line.split()
                file.append((words[0],words[1],words[2], int(words[3]))) # 路径1 路径2 路径3 路径4 路径5 标签


        self.file=file
        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader


    def __getitem__(self,index):

        hrs,lrs,checkin,label=self.file[index]

        hrs_f=self.loader(hrs,type='img')
        msi=self.loader(lrs,type='msi')
        checkin_f = np.array(self.loader(checkin, type='vector'))
        # print(checkin_f)

        if self.transform is not None:
            hrs_f=self.transform(hrs_f)
            msi=torch.from_numpy(msi*1.0)[4:, :, :]
            # msi = self.transform(msi)
            checkin_f=torch.from_numpy(checkin_f).reshape(120, 1)*1.0
        # print(hrs_f.shape, msi.shape,hpi_f.shape,sv_f.shape,checkin_f.shape,floor_f.shape)

        return hrs_f, msi, checkin_f, label

    def __len__(self):
        return len(self.file)


if __name__ == "__main__":
    test_transform=torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),
                                                torchvision.transforms.ToTensor()])
    test_dataset=Mydataset(txt='data\\test_5_10.txt',transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False,pin_memory=True)
    for step,(x1,x2,x3, label) in enumerate(test_loader):
        print(x1.shape, x2.shape, x3.shape, label.shape)
