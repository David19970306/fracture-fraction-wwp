from __future__ import absolute_import
import torch
import time
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms,datasets
from lib.utils.datamanager import datamanager
from lib.models.DenseNet169 import  Densenet169,simpleNet
from torch.optim import SGD,Adam
from lib.utils.loss import CrossEntropyLoss2d

class AverageMeter(object):

    def __init__(self):
        super(AverageMeter, self).__init__()
        self.reset()
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.cnt=0

    def updata(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.cnt+=n
        self.avg=self.sum/self.cnt
class AccMeter(object):

    def __init__(self):
        super(AccMeter, self).__init__()
        self.reset()
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.cnt=0

    def updata(self,val,n=1):
        self.val=val
        self.sum+=val
        self.cnt+=n
        self.avg=self.sum/self.cnt


def MUCRlib(use_gpu=False):
    use_gpu=torch.cuda.is_available()
    training=True

    record_file_path="record_file.txt"
    record_file=open(record_file_path,'a+')

    record_file.write("<<<<------ New lib:\n")

    Transform=transforms.Compose([transforms.Resize([320,320]),
                                  transforms.RandomHorizontalFlip(0.25),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    #trainfilePath="E:\\Datasets\\MURA-v1.1\\train_labeled_studies.csv"
    #testfilePath="E:\\Datasets\\MURA-v1.1\\valid_labeled_studies.csv"

    #dataset = {"train": datamanager(root="E:\\Datasets\\MURA-v1.1\\train_labeled_studies_test.csv", transform=Transform),
     #         "valid": datamanager(root="E:\\Datasets\\MURA-v1.1\\train_labeled_studies_test.csv", transform=Transform)}
    key_str="SHOULDER"
    dataset={"train":datamanager(root="drive/My Drive/MURA/train_labeled_studies.csv",key_str=key_str,transform=Transform),
            "valid": datamanager(root="drive/My Drive/MURA/valid_labeled_studies.csv",key_str=key_str,transform=Transform)}

    if (not dataset["train"].flag) or (not dataset["valid"].flag):
        print("No data")
        return 0
    dataloader={split:DataLoader(dataset=dataset[split],batch_size=8,shuffle=True)
                for split in ["train","valid"]}

    model=Densenet169()
    
    if use_gpu:
        model.cuda()
        print("use_gpu")
        record_file.write("use_gpu\n")
    else:
        print("use_cpu")
        record_file.write("use_cpu\n")
    optimizer=Adam(model.base.classifier.parameters(),lr=1e-4,eps=1e-8,weight_decay=1e-8)
    #optimizer=SGD(model.base.classifier.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-5)
    time_start=time.time()
    best_acc=0.001
    best_epoch = 0
    best_model_state_dict = model.state_dict()
    epochs=20
    for epoch in range(epochs):
        print("############epoch: {}/{}  ##########".format(epoch,epochs))
        record_file.write("############epoch: {}/{}  ##########\n".format(epoch,epochs))
        time_temt=time.time()
        if training:
            model.train(True)
            criterion=CrossEntropyLoss2d().cuda()
            train_acc=train(model=model,dataloader=dataloader['train'],criterion=criterion,optimizer=optimizer,epoch=epoch,use_gpu=use_gpu,record_file=record_file)
            time_end = time.time() - time_temt
            print("Train Using time: {}".format(time_end))
            record_file.write("Train Using time: {}\n".format(time_end))
        test_acc=test(model=model,dataloader=dataloader['valid'],criterion=criterion,epoch=epoch,use_gpu=use_gpu,record_file=record_file)
        time_end=time.time()-time_temt
        print("Epoch Using time: {}".format(time_end))
        record_file.write("Epoch Using time: {}\n".format(time_end))
        if best_acc<test_acc.avg:
            best_acc=test_acc.avg
            best_epoch=epoch
            best_model_state_dict=model.state_dict()
            torch.save(best_model_state_dict, "best_model_state.pkl")
            torch.save(model,"best_model.pkl")


    print("best epoch:{}".format(best_epoch))
    record_file.write("best epoch:{}\n".format(best_epoch))
    model_state=model.state_dict()
    torch.save(model_state,"end_model_state.pkl")
    torch.save(model,"end_model.pkl")
    time_end=time.time()-time_start
    print("overtime:{}".format(time_end))
    record_file.write("overtime:{}\n lib end --------->>>> \n".format(time_end))
    record_file.close()
    return 0

def train(model,dataloader,criterion,optimizer,epoch,use_gpu,record_file):
    print("------------training----------")
    record_file.write("------------training----------\n")
    train_acc=AccMeter()
    train_losses = AverageMeter()
    for batch,data in enumerate(dataloader,0):
        X,y=data
        y = y.float()
        if use_gpu:
            X,y=Variable(X.cuda()),Variable(y.cuda())
        else:
            X,y=Variable(X),Variable(y)
        y_pred=model(X).squeeze()#必须保证y与y_pred的shape相同
        isbig=(y_pred>0.5)
        isEque=(isbig.float()==y)

        optimizer.zero_grad()

        #print(type(y.data)," ",type(y_pred.data))
        #print("y.shape:",y.shape,"y_pred.shape:",y_pred.shape)
        loss=criterion(y_pred,y)
        loss.backward()
        optimizer.step()
        #print("loss.data[0]:",loss.data[0],"   X.size(0):",X.size(0))
        lossdata=loss.data[0]
        #print("lossdata_type:",type(lossdata),"lossdata:",lossdata)
        nsample=X.size(0)
        train_losses.updata(val=lossdata,n=nsample)
        train_acc.updata(val=torch.sum(isEque).data[0],n=nsample)
        if batch%200 == 0:
            #print("Batch: {}, Train Loss: {: .4f}, Train Acc: {: .4f} ".format(batch,train_losses.avg,train_acc.avg))
            print("...Batch: {}, Train Loss: {: .4f}, Train Acc: {: .4f} ".format(batch, train_losses.avg, train_acc.avg))
        if batch%600==0:
            record_file.write("...Batch: {}, Train Loss: {: .4f}, Train Acc: {: .4f} \n".format(batch, train_losses.avg, train_acc.avg))


    print("Epoch: {}  Train Loss: {: .4f}           Train Acc: {: .4f}".format(epoch, train_losses.avg, train_acc.avg))
    record_file.write("Epoch: {}  Train Loss: {: .4f}           Train Acc: {: .4f} __ {: .4f}  {: .4f}\n".format(epoch, train_losses.avg, train_acc.avg,train_losses.avg, train_acc.avg))

    return  train_acc

def test(model,dataloader,criterion,epoch,use_gpu,record_file):
    print("------------validing----------")
    record_file.write("------------validing----------\n")
    test_losses = AverageMeter()
    test_acc=AccMeter()
    for batch,data in enumerate(dataloader,0):
        X,y=data
        y=y.float()
        if use_gpu:
            X,y=Variable(X.cuda()),Variable(y.cuda())
        else:
            X,y=Variable(X),Variable(y)
        y_pred=model(X).squeeze()
        isbig=(y_pred>0.5)
        isEque=(isbig.float()==y)

        loss = criterion(y_pred, y)
        lossdata = loss.data[0]
        nsample = X.size(0)
        test_losses.updata(val=lossdata,n= nsample)
        test_acc.updata(val=torch.sum(isEque).data[0], n=nsample)
    print("Epoch: {} Valid Loss: {:.4f}           Valid Acc: {:.4f}".format(epoch,test_losses.avg,test_acc.avg))
    record_file.write("Epoch: {} Valid Loss: {:.4f}           Valid Acc: {:.4f} __ {:.4f}  {:.4f}\n".format(epoch,test_losses.avg,test_acc.avg,test_losses.avg,test_acc.avg))

    return test_acc

MUCRlib()

