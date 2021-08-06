#!/usr/bin/env python


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

    def predict_callback(self, msg):
        pass


if __name__=="__main__":
    try:
        rospy.init_node('prediction')
        rospy.Subscriber('/dish_state', Float64MultiArray, self.predict_callback, queue_size=1)
        pub = rospy.Publisher('/hough_pointcloud', PointCloud2, queue_size=100)
        rospy.spin()
    except rospy.ROSInterruptException: pass
