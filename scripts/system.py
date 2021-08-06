#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle
import rospy
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from std_msgs.msg import Float64MultiArray, Float64

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        """
        This imitates alexnet. 
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2) #入力チャンネル数は1, 出力チャンネル数は96 
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        #self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1 = nn.Linear(50176, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.fc4 = nn.Linear(10 + 4, 14)
        self.fc5 = nn.Linear(14, 1) # output is 1 dim scalar probability
        """
        self.conv1 = nn.Conv2d(3, 4, 3, 2, 1)
        self.cbn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3, 2, 1)
        self.cbn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3, 2, 1)
        self.cbn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3, 2, 1)
        self.cbn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, 3, 2, 1)
        self.cbn5 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8 + 4, 12)
        self.fc6 = nn.Linear(12, 1) # output is 1 dim scalar probability
        """
        self.conv1 = nn.Conv2d(1, 4, 3, 2, 1)
        self.cbn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3, 2, 1)
        self.cbn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3, 2, 1)
        self.cbn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3, 2, 1)
        self.cbn4 = nn.BatchNorm2d(32)
        #self.conv5 = nn.Conv2d(32, 64, 3, 2, 1)
        #self.cbn5 = nn.BatchNorm2d(64)
        #self.fc1 = nn.Linear(128, 64)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8 + 4, 12)
        self.fc5 = nn.Linear(12, 1) # output is 1 dim scalar probability
        """

    # depth encording without concate grasp point
    def forward(self, x, y):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.cbn1(x)
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv2(x))
        x = self.cbn2(x)
        x = F.relu(self.conv3(x))
        x = self.cbn3(x)
        x = F.relu(self.conv4(x))
        x = self.cbn4(x)
        x = F.relu(self.conv5(x))
        x = self.cbn5(x)
        x = x.view(-1, self.num_flat_features(x))
        #depth_data =depth_data.view(depth_data.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        z = torch.cat((x, y), dim=1)
        z = F.relu(self.fc5(z))
        z = self.fc6(z)
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.cbn1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.cbn2(x)
        x = F.relu(self.conv3(x))
        x = self.cbn3(x)
        #x = F.relu(self.conv4(x))
        #x = self.cbn4(x)
        #x = F.relu(self.conv5(x))
        #x = self.cbn5(x)
        x = x.view(-1, self.num_flat_features(x))
        #depth_data =depth_data.view(depth_data.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        z = torch.cat((x, y), dim=1)
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        """
        return z
   
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class WashSystem():
    def __init__(self):
        # PAST NUM
        self.PAST_STATE_NUM = 2   # FIX
        self.PAST_INPUT_NUM = 1   # FIX
        self.PAST_NUM = max(self.PAST_STATE_NUM, self.PAST_INPUT_NUM)

        # INPUT DIM
        self.STATE_DIM = 2
        self.INPUT_DIM = 2
        self.DELTA_STEP = 5   # FIX

        # HyperParameters for NeuralNetwork
        self.INPUT_NN_DIM = self.STATE_DIM * self.PAST_STATE_NUM + self.INPUT_DIM * self.PAST_INPUT_NUM
        self.OUTPUT_NN_DIM = self.STATE_DIM
        self.TRAIN_TEST_RATIO = 0.8   # FIX
        self.BATCH_SIZE = 1000   # FIX

        # real data
        self.past_states = []
        self.past_inputs = []
        self.now_input = [0.7, 0]

        # train parameters
        self.LOOP_NUM = 1

    def predict_callback(self, msg):
        # store past states data

        # lpf for state and input

        # optimize input ??

        # online training 
        # train
        losses = 0
        for i in range(LOOP_NUM):
            x_ = Variable(np.array(X).astype(np.float32).reshape(batch_num, 6))
            t_ = Variable(np.array(Y).astype(np.float32).reshape(batch_num, 2))

            self.train_optimizer.zero_grad()
            outputs = self.model(x_)
            loss = self.criterion(outputs.view_as(t_), t_)
            loss.backward()
            self.train_optimizer.step()
            losses += loss.data

            writer = SummaryWriter(log_dir)
            running_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), tensorboard_cnt) #(epoch + 1) * i)
            if i % 100 == 99: 
                #print('[%d, %5d] loss: %.3f' %
                #      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            tensorboard_cnt += 1

    def test(self):
        for data in testloader:
            depth_data, grasp_point, labels = data
            outputs = self.model(depth_data, grasp_point)
            # lossのgrasp_point偏微分に対してoptimaizationする．
            depth_data.requires_grad(False)
            grasp_point.requires_grad(True)
            loss = self.criterion(outputs.view_as(labels), labels)
            loss.backward()
            self.train_optimizer.step()       

    def realtime_feedback(self):
        rospy.init_node('prediction')
        rospy.Subscriber('/dish_state', Float64MultiArray, self.predict_callback, queue_size=1)
        #rospy.Subscriber('/end effector pos ?', Float64MultiArray, self.predict_callback, queue_size=1)
        #rospy.Subscriber('/force? ', Float64MultiArray, self.predict_callback, queue_size=1)
        
        pub = rospy.Publisher('/hough_pointcloud', PointCloud2, queue_size=100)
        rospy.Rate(30)
   
if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda signal, frame: sys.exit(0))

    # init arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", nargs='?', default=False, const=True, help="train NN")
    parser.add_argument("--action", "-a", default=2, help="0:simulate 1:realtime feedback with simulate 2:realtime feedback with real robot")
    #parser.add_argument("--model", "-m", default='../log/diabolo_system/mymodel.h5', help="which model do you use")
    parser.add_argument("--online_training", "-o", nargs='?', default=False, const=True, help="online training")                
    args = parser.parse_args()

    # parse
    train_flag = int(args.train)   # parse train
    action = int(args.action)   # parse action
    #model_file = args.model   # which model
    online_training_ = args.online_training   # which model
    
    ws = WashSystem()
    
    # train model or load model
    if train_flag:
        ws.load_data(LOG_FILES)
        ws.arrange_data()
        ws.make_model()
        print('[Train] start')        
        ws.train(loop_num=500)
        ws.save_model()
    #else:
    #    ws.load_data(LOG_FILES)
    #    ws.arrange_data()
    #    ws.make_model()
    #    print('[Train] pass')                
    #    ws.load_model(log_file=model_file)
    #    print('load model from {}'.format(model_file))
        
    # test
    print('[Test] start')            
    ws.test()

    # action
    if action == 0:
        print('[Simulate] start')
        #ws.simulate_offline(simulate_loop_num=300)
    elif action == 1:
        print('[RealtimeFeedback] start with simulate')                
        ws.realtime_feedback(simulate=True, online_training=False)
    elif action == 2:
        print('[RealtimeFeedback] start with real robot')                
        ws.realtime_feedback(simulate=False, online_training=online_training_)     
