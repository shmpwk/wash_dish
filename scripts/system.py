#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, signal, sys
import pickle
import yaml
import argparse
import rospy
import datetime
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from std_msgs.msg import Float64MultiArray, Float64
#from buffer import SimpleBuffer
from network import *

class MyDataset(Dataset):
    def __init__(self, file_path):
        self.datanum = 0
        self.seq_num = -1
        self.seq_length = 30
        self.time_steps = 5
        self.buffer_size = 2
        self.batch_size = 2
        self.input_rarm = []
        self.input_larm = []
        self.state_point = []
        self.state_rforce = []
        self.state_lforce = []
        input_rarm_key = "r_arm_controller.yaml"
        input_larm_key = "l_arm_controller.yaml"
        state_point_key = "bboxes.yaml"
        state_rforce_key = "r_force.yaml"
        state_lforce_key = "l_force.yaml"
        #state_point_key = point.yaml  # To do

        # data buffer
        #buf = Simplebuffer(self.seq_length, buffer_size, input_rarm_shape, input_larm_shape, state_rforce_shape, state_lforce_shape, state_point_shape, device=torch.device('cuda'))
        #input_rarm_buffer, input_larm_buffer, state_rforce_buffer, state_lforce_buffer, state_point_buffer = buf.load(

        for dir_name, sub_dirs, files in sorted(os.walk(file_path)):
            if sub_dirs != []:
                self.seq_num += 1 #next seq
            for file in sorted(files):
                if file == state_point_key:
                    with open(os.path.join(dir_name, file), 'rb') as obj:
                        obj_point = yaml.safe_load(obj)
                        if obj_point["boxes"] == []:
                            break
                        pos_x = obj_point["boxes"][0]["pose"]["position"]["x"]
                        pos_y = obj_point["boxes"][0]["pose"]["position"]["y"]
                        pos_z = obj_point["boxes"][0]["pose"]["position"]["z"]
                        ori_x = obj_point["boxes"][0]["pose"]["orientation"]["x"]
                        ori_y = obj_point["boxes"][0]["pose"]["orientation"]["y"]
                        ori_z = obj_point["boxes"][0]["pose"]["orientation"]["z"]
                        ori_w = obj_point["boxes"][0]["pose"]["orientation"]["w"]
                        dim_x = obj_point["boxes"][0]["dimensions"]["x"]
                        dim_y = obj_point["boxes"][0]["dimensions"]["y"]
                        dim_z = obj_point["boxes"][0]["dimensions"]["z"]
                        obj_point_arr = [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w, dim_x, dim_y, dim_z]
                        self.state_point = np.append(self.state_point, obj_point_arr)
                if file == input_rarm_key:
                    with open(os.path.join(dir_name, file), 'rb') as rarm:
                        input_rarm_controller = yaml.safe_load(rarm)
                        self.input_rarm = np.append(self.input_rarm, input_rarm_controller["desired"]["positions"], axis=0)
                        self.state_rforce = np.append(self.state_rforce, input_rarm_controller["desired"]["effort"], axis=0)
                if file == input_larm_key:
                    with open(os.path.join(dir_name, file), 'rb') as larm:
                        input_larm_controller = yaml.safe_load(larm)
                        self.input_larm = np.append(self.input_larm, input_larm_controller["desired"]["positions"], axis=0)
                        self.state_lforce = np.append(self.state_lforce, input_larm_controller["desired"]["effort"], axis=0)

                        self.datanum += 1

        #print(self.input_larm.shape) 1 dim (data_num * 7)
        self.input_rarm = self.input_rarm.reshape(self.seq_num, -1, 7)
        self.input_larm = self.input_larm.reshape(self.seq_num, -1, 7)
        self.state_rforce = self.state_rforce.reshape(self.seq_num, -1, 7)
        self.state_lforce = self.state_lforce.reshape(self.seq_num, -1, 7)
        self.state_point = self.state_point.reshape(self.seq_num, -1, 10)
        p_arm = np.append(self.input_rarm, self.input_larm, axis=2)
        f_arm = np.append(self.state_rforce, self.state_lforce, axis=2)
        arm = np.append(p_arm, f_arm, axis=2)
        self.dataset = np.append(arm, self.state_point, axis=2)
        print(self.dataset.shape) # now (dataset num, seq_length, z_input_dim) # (time_step, data_dim)??

    def __len__(self):
        #return self.datanum #should be dataset size / batch size ? 
        #print((self.seq_length - self.time_steps) * self.seq_num) #/ self.batch_size)
        return (self.seq_length - self.time_steps) * self.seq_num #/ self.batch_size # len(self.dataset) - self.time_steps  (30-5)/2*4=50 

    def __getitem__(self, idx):
        j = int(idx  / (self.seq_length - self.time_steps))
        dataset = self.dataset[j,idx%(self.seq_length-self.time_steps):idx%(self.seq_length-self.time_steps)+self.time_steps,:]
        label = self.dataset[j,(idx+self.time_steps)/self.seq_length,:]
        dataset = torch.from_numpy(np.array(dataset)).float()
        label = torch.from_numpy(np.array(label)).float()
        return dataset, label  # dataset.shape = (time_steps, 38), label.shape = (1, 38)
        
        #p_rarm = self.input_rarm[idx]
        #p_larm = self.input_larm[idx]
        #f_rarm = self.state_rforce[idx]
        #f_larm = self.state_lforce[idx]
        #obj = self.state_point
        #p_rarm = torch.from_numpy(np.array(p_rarm)).float()
        #p_larm = torch.from_numpy(np.array(p_larm)).float()
        #f_rarm = torch.from_numpy(np.array(f_rarm)).float()
        #f_larm = torch.from_numpy(np.array(f_larm)).float()
        #obj = torch.from_numpy(np.array(obj)).float()
        #type(p_rarm)
        #return p_rarm, p_larm

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
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # real data
        self.past_states = []
        self.past_inputs = []
        self.now_input = [0.7, 0]

        # train parameters
        self.LOOP_NUM = 1
        self.time_steps = 5

        self.epoch = 1
        self.sample_num = 16 # ? image size related something
        self.batch_size = 2 #batch_size is 2 but want to change seq_size to data_length
        self.z_input_dim = 38 #same as z_dim? -> no, angle vector and obj point
        self.data_shape = self.z_input_dim # same as input_size??
        self.x_input_size = 24 #angle vector(7*2 dim), obj point(10 dim)
        self.data_length = 5
        self.lrG = 0.0002
        self.lrD = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999    

    def load_data(self, datasets):
        train_dataloader = torch.utils.data.DataLoader(
            datasets, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        #tmp = next(iter(train_dataloader))
        return train_dataloader

    def make_model(self):
        self.model = Net()
        self.model = self.model.to(self.DEVICE)
        #self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.MSELoss()
        self.train_optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        #summary(self.model, [(3, 128, 128), (4,)])

        # networks init
        self.G = Lstm(inputDim=self.z_input_dim, hiddenDim=4, outputDim=self.z_input_dim) #(38,4,38)
        self.D = Discriminator(input_dim=self.x_input_size, output_dim=1, input_size=self.x_input_size) #(24,1,)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))
        self.G.cuda()
        self.D.cuda()
        self.BCE_loss = nn.BCELoss().cuda()

        #self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        #self.sample_z_ = self.sample_z_.cuda()
        
    def predict_callback(self, msg):
        # store past states data

        # lpf for state and input

        # optimize input ??

        # online training 
        # train
        losses = 0
        for i in range(self.LOOP_NUM):
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

    def pretrain(self, train_dataloader):
        now = datetime.datetime.now()
        tensorboard_cnt = 0
        log_dir = 'data/loss/loss_' + now.strftime('%Y%m%d_%H%M%S')
        for epoch in range(10):
            # training
            running_loss = 0.0
            training_accuracy = 0.0
            for i, data in enumerate(train_dataloader, 0):
                self.G_optimizer.zero_grad()
                z, t = data
                z = z.float().to(self.DEVICE)
                t = t.float().to(self.DEVICE)
                # data_next(t) =G(data(0:t-1))
                data_next = self.G(z)
                loss = self.criterion(data_next, t)
                loss.backward()
                self.G_optimizer.step()
                running_loss += loss.item()

                training_accuracy += np.sum(np.abs((data_next.data.cpu() - t.data.cpu()).numpy()) < 0.1)
                writer = SummaryWriter(log_dir)
                writer.add_scalar("Loss/train", loss.item(), tensorboard_cnt) #(epoch + 1) * i)
                if i % 100 == 99: 
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
                tensorboard_cnt += 1

            print('%d loss: %.3f, training_accuracy: %.5f' % (
                epoch + 1, running_loss, training_accuracy))

    def train(self, train_dataloader):
        now = datetime.datetime.now()  
        log_dir = './Data/loss/loss_' + now.strftime('%Y%m%d_%H%M%S')
        tensorboard_cnt = 0
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()

        for epoch in range(2):
            self.G.train()
            for iter, x_ in enumerate(train_dataloader, 0):
                if iter == train_dataloader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, 30, self.z_input_dim))
                x_, z_ = x_.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_)
                print(G_.shape)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()
                
                if ((iter + 1) % 10) == 0:
                    with torch.no_grad():
                        tot_num_samples = min(self.sample_num, self.batch_size)
                        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
                        #display_process(self.train_hist, self.G, image_frame_dim, self.sample_z_)
                        #display.clear_output(wait=True)
                        #display.display(pl.gcf())
                        #plt.close()

                writer = SummaryWriter(log_dir)
                D_running_loss += D_loss.item()
                G_running_loss += G_loss.item()
                writer.add_scalar("Loss/train", D_loss.item(), tensorboard_cnt) #(epoch + 1) * i)
                writer.add_scalar("Loss/train", G_loss.item(), tensorboard_cnt) #(epoch + 1) * i)
                if i % 100 == 99:    # 2 ミニバッチ毎に表示する
                    print('[%d, %5d] D_loss: %.3f' %
                          (epoch + 1, i + 1, D_running_loss / 100))
                    print('[%d, %5d] G_loss: %.3f' %
                          (epoch + 1, i + 1, G_running_loss / 100))
                    running_loss = 0.0
                tensorboard_cnt += 1
        #writer.flush()  
        #plt.close()
        print("Training finish!")

    #def train(self, train_dataloader):
    #    for epoch in range(2):
    #        losses = 0
    #        for i, data in enumerate(train_dataloader, 0):
    #            input_rarm, input_larm = data
    #            x_ = input_rarm
    #            t_ = Variable(np.array(Y).astype(np.float32).reshape(batch_num, 2))

    #            self.train_optimizer.zero_grad()
    #            outputs = self.model(x_)
    #            loss = self.criterion(outputs.view_as(t_), t_)
    #            loss.backward()
    #            self.train_optimizer.step()
    #            losses += loss.data

    #            writer = SummaryWriter(log_dir)
    #            running_loss += loss.item()
    #            writer.add_scalar("Loss/train", loss.item(), tensorboard_cnt) #(epoch + 1) * i)
    #            if i % 100 == 99: 
    #                #print('[%d, %5d] loss: %.3f' %
    #                #      (epoch + 1, i + 1, running_loss / 100))
    #                running_loss = 0.0
    #            tensorboard_cnt += 1

    def test(self):
        for data in self:
            depth_data, grasp_point, labels = data
            outputs = self.model(depth_data, grasp_point)
            # lossのgrasp_point偏微分に対してoptimaizationする．
            depth_data.requires_grad(False)
            grasp_point.requires_grad(True)
            loss = self.criterion(outputs.view_as(labels), labels)
            loss.backward()
            self.train_optimizer.step()       

    def realtime_feedback(self, simulate=False, online_training=False):
        rospy.init_node('prediction')
        rospy.Subscriber('/dish_state', Float64MultiArray, self.predict_callback, queue_size=1)
        #rospy.Subscriber('/end effector pos ?', Float64MultiArray, self.predict_callback, queue_size=1)
        #rospy.Subscriber('/force? ', Float64MultiArray, self.predict_callback, queue_size=1)
        
        pub = rospy.Publisher('/hough_pointcloud', PointCloud2, queue_size=100)
        rospy.Rate(30)

    def save_model(self):
        now = datetime.datetime.now()   
        lstm_filename = 'Data/trained_gan_model/model_' + now.strftime('%Y%m%d_%H%M%S') + '.pth'
        lstm_model_path = lstm_filename      
        # GPU save   
        ## Save only parameter   
        torch.save(self.model.state_dict(), lstm_model_path) 
        ## Save whole model
        #torch.save(self.model(), lstm_model_path)   
        #torch.save(self.ae(), encoder_model_path)   
        # CPU save  
        #torch.save(self.model.to('cpu').state_dict(), model_path)
        print("Finished Saving model")    

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda signal, frame: sys.exit(0))

    # init arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", nargs='?', default=True, const=True, help="train NN")
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
    FILE_PATH = "Data/seq_data/wash_dish"
    
    # train model or load model
    if train_flag:
        datasets = MyDataset(FILE_PATH)
        for i in range(len(datasets)):
            img = datasets[i]
            #print(type(img))
        train_dataloader = ws.load_data(datasets)
        #ws.arrange_data()
        ws.make_model()
        print('[Train] start')        
        ws.pretrain(train_dataloader)
        ws.train(train_dataloader)
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
    #ws.test()

    # action
    #if action == 0:
    #    print('[Simulate] start')
    #    #ws.simulate_offline(simulate_loop_num=300)
    #elif action == 1:
    #    print('[RealtimeFeedback] start with simulate')                
    #    ws.realtime_feedback(simulate=True, online_training=False)
    #elif action == 2:
    #    print('[RealtimeFeedback] start with real robot')                
    #    ws.realtime_feedback(simulate=False, online_training=online_training_)     
