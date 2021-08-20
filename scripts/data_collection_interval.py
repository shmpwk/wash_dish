#!/usr/bin/env python
import rospy 
from std_srvs.srv import Trigger, TriggerRequest

def main():
    try:
        rospy.init_node('data_collection_interval')
        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            rospy.wait_for_service("/seq_data_collection/save_request")
            try:
                service = rospy.ServiceProxy("/seq_data_collection/save_request", Trigger)
                req = TriggerRequest()
                service(req)

            except rospy.ServiceException, e:
                print "Service call failed: %s"%e
            r.sleep()

    except rospy.ROSInterruptException: pass

if __name__=="__main__":
    main()
