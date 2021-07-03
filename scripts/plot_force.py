#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def load_data():
    dataset_path = os.path.expanduser('~/Data/force')

    key = '.pkl'
    for dir_name, sub_dirs, files in os.walk(dataset_path):
        for f in files:
            load_file = os.path.join(dir_name, f)
            if key ==f[-len(key):]:
                data = pickle.load(open(load_file, 'rb'))
                return data

def plot_data(datatype):
    fig0 = plt.figure()
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    fig5 = plt.figure()
    ax0 = fig0.add_subplot(111, xlabel="time", ylabel="force_lx")
    ax1 = fig1.add_subplot(111, xlabel="time", ylabel="force_ly")
    ax2 = fig2.add_subplot(111, xlabel="time", ylabel="force_lz")
    ax3 = fig3.add_subplot(111, xlabel="time", ylabel="force_rx")
    ax4 = fig4.add_subplot(111, xlabel="time", ylabel="force_ry")
    ax5 = fig5.add_subplot(111, xlabel="time", ylabel="force_rz")
    ax0.plot(datatype[:,0])
    ax1.plot(datatype[:,1])
    ax2.plot(datatype[:,2])
    ax3.plot(datatype[:,3])
    ax4.plot(datatype[:,4])
    ax5.plot(datatype[:,5])
    fig0.savefig('force0.png')
    fig1.savefig('force1.png')
    fig2.savefig('force2.png')
    fig3.savefig('force3.png')
    fig4.savefig('force4.png')
    fig5.savefig('force5.png')

    #fig, axs = plt.subplots(8, 1)
    #print(len(datatype))
    #datatype = np.array(datatype)
    ##exec("axs[0].plot(self.%s['force'][datatype[2]])" % (datatype[0]))
    #axs[0].set_xlabel('time',fontsize="small")
    #axs[0].set_ylabel("force",fontsize="small")
    #axs[0].grid(True)
    ##plot all arm joints
    #for i in datatype:
    #    #exec("axs[i+1].plot(np.transpose(self.%s[datatype[1]][datatype[2]])[i])" % (datatype[0]))
    #    axs[i+1].set_xlabel('time',fontsize="small")
    #    axs[i+1].set_ylabel(name,fontsize="small")
    #    axs[i+1].grid(True)
    #plt.show()

def main():
    data = load_data()
    plot_data(data)

if __name__ == "__main__":
    main()
