import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
import cv2

def convert_depth_image(ros_image):
    cv_bridge = CvBridge()
    try:
        depth_image = cv_bridge.imgmsg_to_cv2(ros_image, desired_encoding='passthrough')
    except CvBridgeError, e:
        print(e)
    depth_array = np.array(depth_image, dtype=np.float32)
    np.save("depth_img.npy", depth_array)
    rospy.loginfo(depth_array)
    #To save image as png
    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imwrite("depth_img.png", depth_colormap)
    #Or you use 
    # depth_array = depth_array.astype(np.uint16)
    # cv2.imwrite("depth_img.png", depth_array)


def pixel2depth():
    rospy.init_node('pixel2depth',anonymous=True)
    rospy.Subscriber("/pepper/camera/depth/image_raw", Image,callback=convert_depth_image, queue_size=1)
    rospy.spin()

if __name__ == '__main__':
    pixel2depth()
