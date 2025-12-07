#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose
import math

class Pub():
    def __init__(self):
        self.r = rospy.Rate(0.1)

        self.pose = Pose()
        self.init_value()
        self.ang_pub = rospy.Publisher('/jointangles', Pose, queue_size=1)
        
        print("^^^^^^^^^^^^^^^^^^")
        while not rospy.is_shutdown():
            self.ang_pub.publish(self.pose)
            self.r.sleep()
    def init_value(self):
        self.pose.position.x = 0/180.0 * math.pi
        self.pose.position.y = 0/180.0 * math.pi
        self.pose.position.z = 120/180.0 * math.pi
        self.pose.orientation.x = 0/180.0 * math.pi
        self.pose.orientation.y = 90/180.0 * math.pi
        self.pose.orientation.z = 0/180.0 * math.pi
        self.pose.orientation.w = 36

    def get(self, msg):
        pass
    def exc(self):
        self.ang_pub.publish(self.pose)

        
	rospy.sleep(1)

if __name__ == "__main__": 
    rospy.init_node('Top',anonymous=False)

    try:
        my = Pub()
        my.exc()
        # rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("cam_goal node terminated.")
