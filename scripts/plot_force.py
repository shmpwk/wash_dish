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
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel="time", ylabel="force_x")
    ax.plot(datatype)
    fig.savefig('testforce.png')

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
