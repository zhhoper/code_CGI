import argparse
parser = argparse.ArgumentParser()
parser.add_argument('network', help='network structure')
parser.add_argument('savePath', help='folder to save the results')
parser.add_argument('imgSize', type=int, default=256, help='image size')
parser.add_argument('maxSize', type=int, default=300, help='maximum size of resized images')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--epoch', type=int, help='number of epoch to train')
parser.add_argument('--wd', type=float, help='weight decay')
args = parser.parse_args()

import sys
sys.path.append('utils')
sys.path.append('model')

# load network structure
import importlib
network = importlib.import_module(args.network)

import os
import numpy as np
import time

import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch
from tensorboardX import SummaryWriter

# load module defined by myself
from defineCriterion import *
from loadData_CGI import *
from defineHelp import *


# set random seed for all possible randomness
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


baseFolder = '../CGI/'

criterion = mask_loss()

my_network = network.HourglassNet()
my_network.cuda()
gradientLayer = gradientLayer_color()
gradientLayer.cuda()

transformer = transforms.Compose([cropImg(output_size=args.imgSize, 
    maxSize=args.maxSize), ToTensor()])
transformer_test = transforms.Compose([testTransfer(output_size=args.imgSize), ToTensor()])

trainLoaderHelper = CGI(dataFolder = os.path.join(baseFolder, 'images'), 
	albedoFolder=os.path.join(baseFolder, 'images'),
	shadingFolder=os.path.join(baseFolder, 'shading'),
	normalFolder=os.path.join(baseFolder, 'normal'),
	maskFolder=os.path.join(baseFolder, 'mask'),
	fileListName=os.path.join(baseFolder, 'training.list'),
	#fileListName=os.path.join(baseFolder, 'debug.list'),
	transform = transformer)

testLoaderHelper = CGI(dataFolder=os.path.join(baseFolder, 'images'),
	albedoFolder=os.path.join(baseFolder, 'images'),
	shadingFolder=os.path.join(baseFolder, 'shading'),
	normalFolder=os.path.join(baseFolder, 'normal'),
	maskFolder=os.path.join(baseFolder, 'mask'),
	fileListName=os.path.join(baseFolder, 'testing.list'),
	transform = transformer_test)

trainLoader = torch.utils.data.DataLoader(trainLoaderHelper, 
	batch_size=10, shuffle=True, num_workers=5)

testLoader = torch.utils.data.DataLoader(testLoaderHelper, 
	batch_size=10, shuffle=False, num_workers=5)


def getLoss(output_albedo, output_shading, output_normal, albedo, shading, normal, mask):
    '''
    	compute the loss for the mini-batch
        contains: l1 loss for albedo, normal and shading
                  and their gradient
    '''
    output_albedo_grad = gradientLayer.forward(output_albedo)
    albedo_grad = gradientLayer.forward(albedo)
    output_shading_grad = gradientLayer.forward(output_shading)
    shading_grad = gradientLayer.forward(shading)
    output_normal_grad = gradientLayer.forward(output_normal)
    normal_grad = gradientLayer.forward(normal)
    
    imgMask = mask
    # repeat gradMask for color gradient
    mask_R = mask[:,0,:,:].unsqueeze(1)
    mask_G = mask[:,1,:,:].unsqueeze(1)
    mask_B = mask[:,2,:,:].unsqueeze(1)
    gradMask = torch.cat((mask_R, mask_R, mask_G, mask_G, mask_B, mask_B), dim=1)
    
    loss_albedo = criterion(imgMask, torch.abs(albedo - output_albedo))
    loss_shading = criterion(imgMask, torch.abs(shading - output_shading))
    
    loss_albedo_grad = criterion(gradMask, torch.abs(albedo_grad - output_albedo_grad))
    loss_shading_grad = criterion(gradMask, torch.abs(shading_grad - output_shading_grad))
    
    loss_normal = criterion(imgMask, torch.abs(output_normal - normal))
    loss_normal_grad = criterion(gradMask, torch.abs(normal_grad - output_normal_grad))
    
    return loss_albedo, loss_shading, loss_albedo_grad, loss_shading_grad, loss_normal, loss_normal_grad

def networkForward(data, Testing=False):
    '''
        given data, compute the loss for the network
        return loss in a list
    '''
    # get output from the network
    inputs, albedo, shading, normal, mask = data
    inputs, albedo, shading, normal, mask = \
            Variable(inputs.cuda(), volatile=Testing).float(), \
            Variable(albedo.cuda(), volatile=Testing).float(), \
            Variable(shading.cuda(), volatile=Testing).float(), \
            Variable(normal.cuda(), volatile=Testing).float(), \
            Variable(mask.cuda(), volatile=Testing).float()
    output_albedo, output_normal, output_shading  = my_network(inputs)

    # compute the loss
    loss_albedo, loss_shading, loss_albedo_grad, loss_shading_grad, \
        loss_normal, loss_normal_grad = \
        getLoss(output_albedo, output_shading, output_normal,
            albedo, shading, normal, mask)
    
    # append all the losses
    loss = {}
    loss['albedo'] = loss_albedo
    loss['albedo_grad'] = loss_albedo_grad
    loss['shading'] = loss_shading
    loss['shading_grad'] = loss_shading_grad
    loss['normal'] = loss_normal
    loss['normal_grad'] = loss_normal_grad
    loss['loss_list'] = [loss_albedo.data[0], loss_albedo_grad.data[0], loss_shading.data[0],
        loss_shading_grad.data[0], loss_normal.data[0], loss_normal_grad.data[0]]
    return loss, output_albedo, output_normal, output_shading


def main(savePath, lr=1e-3, weight_decay=0, total_epoch=100):
    begin_time = time.time()
    print 'learning rate is %.6f' % lr
    print 'weight decay is %.6f' % weight_decay
    print 'epoch is %05d' % total_epoch
    savePath = savePath + '_{:0.4f}_{:0.2f}_{:04d}'.format(lr, weight_decay, total_epoch)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    saveIntermedia = os.path.join(savePath, 'trained_model')
    if not os.path.exists(saveIntermedia):
        os.makedirs(saveIntermedia)
    writer = SummaryWriter(os.path.join(savePath, 'tensorboard'))

    monitor_count = 20 # every 20 iterations, output the current loss

    optimizer = optim.Adam(my_network.parameters(), lr = lr, weight_decay=weight_decay)
    fid = open(os.path.join(savePath, 'training.log'), 'w')
    fid_sep = open(os.path.join(savePath, 'training_sep.log'), 'w')
    fid_test = open(os.path.join(savePath, 'testing.log'), 'w')
    fid_test_sep = open(os.path.join(savePath, 'testing_sep.log'), 'w')
    print>>fid_sep, 'albedo, albedo_grad, shading, shading_grad, normal, normal_grad, lighting'
    print>>fid_test_sep, 'albedo, albedo_grad, shading, shading_grad, normal, normal_grad, lighting'

    numIte_training = 0
    numIte_testing = 0
    for epoch in range(total_epoch):
        # ---------------------------------------------------------
        # training
        # ---------------------------------------------------------
        running_loss = 0.0
        # loss for each indivisual component
        running_albedo_loss = 0.0
        running_shading_loss = 0.0
        running_normal_loss = 0.0

        my_network.train(True)
        epoch_time = time.time()
        loss_list = []
        training_loss_list = []
        
        for i , data in enumerate(trainLoader, 0):
            optimizer.zero_grad()
            loss_miniBatch, output_albedo, output_normal, output_shading  \
                    = networkForward(data, Testing=False)
            loss = loss_miniBatch['albedo'] + loss_miniBatch['albedo_grad'] + \
                    loss_miniBatch['shading'] + loss_miniBatch['shading_grad'] + \
                    loss_miniBatch['normal'] + loss_miniBatch['normal_grad']
            numLoss = len(loss_miniBatch['loss_list'])
            loss.backward()
            optimizer.step()
            loss_list.append(loss_miniBatch['loss_list'])
            training_loss_list.append(loss_miniBatch['loss_list'])
            running_loss += loss.data[0]

            running_albedo_loss  += loss_miniBatch['albedo'].data[0]
            running_shading_loss  += loss_miniBatch['shading'].data[0]
            running_normal_loss  += loss_miniBatch['normal'].data[0]

            if i%monitor_count == monitor_count - 1:
                numIte_training = numIte_training + 1
                training_loss_list = np.mean(np.array(training_loss_list), axis=0)
                print '[%d %5d] loss: ' % (epoch + 1, i+1),
                print loss_miniBatch
                print '%.4f '*numLoss % tuple(training_loss_list)
                print>>fid, '%d %5d ' % (epoch+1, i+1),
                print>>fid, '%.4f '*numLoss % tuple(training_loss_list)
                training_loss_list = []

                writer.add_scalar('train/loss_albedo', running_albedo_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_shading', running_shading_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_normal', running_normal_loss/monitor_count, numIte_training)
                writer.add_image('trainImage_albedo', output_albedo[0:5,:,:,:], numIte_training)
                writer.add_image('trainImage_shading', output_shading[0:5,:,:,:]/20.0, numIte_training)
                writer.add_image('trainImage_normal', (output_normal[0:5,:,:,:]+1)/2, numIte_training)

                running_loss = 0
                running_albedo_loss = 0
                running_shading_loss = 0
                running_normal_loss = 0
                running_light_loss = 0

        loss_list = np.mean(np.array(loss_list), axis=0)
        print >> fid_sep, '%0.6f '* numLoss % tuple(loss_list)
        print '%0.6f '* numLoss % tuple(loss_list)
        # ---------------------------------------------------------
        # validation 
        # ---------------------------------------------------------
        my_network.train(False)
        test_loss = 0
        # loss for each component
        test_albedo_loss = 0
        test_shading_loss = 0
        test_normal_loss = 0

        count = 0
        test_loss_list = []
        for i, data in enumerate(testLoader, 0):
            loss_miniBatch, output_albedo, output_normal, output_shading \
                    = networkForward(data, Testing=True)
            loss = loss_miniBatch['albedo'] + loss_miniBatch['albedo_grad'] + \
                    loss_miniBatch['shading'] + loss_miniBatch['shading_grad'] + \
                    loss_miniBatch['normal'] + loss_miniBatch['normal_grad']
            numLoss = len(loss_miniBatch['loss_list'])
            test_loss_list.append(loss_miniBatch['loss_list'])
            test_loss += loss.data[0]
            test_albedo_loss += loss_miniBatch['albedo']
            test_normal_loss += loss_miniBatch['normal']
            test_shading_loss += loss_miniBatch['shading']
            count = count + 1
        test_loss_list = np.mean(np.array(test_loss_list), axis=0)
        print '[%d  ] test loss: ' % (epoch+1),
        print '%.4f '*numLoss % tuple(test_loss_list)
        print>>fid_test, '%d ' % (epoch+1),
        print>>fid_test, '%.4f '*numLoss % tuple(test_loss_list)
        
        writer.add_scalar('test/loss_albedo', test_albedo_loss/count, epoch)
        writer.add_scalar('test/loss_shading', test_shading_loss/count, epoch)
        writer.add_scalar('test/loss_normal', test_normal_loss/count, epoch)
        writer.add_image('testImage_albedo', output_albedo[0:5,:,:,:], epoch)
        writer.add_image('testImage_shading', output_shading[0:5,:,:,:]/20.0, epoch)
        writer.add_image('testImage_normal', (output_normal[0:5,:,:,:]+1)/2, epoch)
    
        print >> fid_test_sep, '%0.6f '* numLoss % tuple(test_loss_list)
        print '%0.6f '* numLoss % tuple(test_loss_list)
    
        tmp_saveName = os.path.join(saveIntermedia, 'trained_model_{:02d}.t7'.format(epoch))
        my_network.cpu()
        torch.save(my_network.state_dict(), tmp_saveName)
        my_network.cuda()
        print 'this epoch cost %s seconds' %(time.time() - epoch_time)
    
    my_network.cpu()
    torch.save(my_network, os.path.join(savePath, 'trained_model.t7'))
    print 'time used for training is %s' % (time.time() - begin_time)
    print('Finished training')
    fid.close()
    fid_test.close()
    fid_sep.close()
    fid_test_sep.close()
    writer.close()

if __name__ == '__main__':
		savePath = args.savePath
		lr = 1e-3
		wd = 0
		total_epoch = 100
		if not args.lr is None:
				lr = args.lr
		if not args.wd is None:
				wd = args.wd
		if not args.epoch is None:
				total_epoch = args.epoch
		main(savePath, lr, wd, total_epoch)
