import lcm
import numpy as np
import sys
sys.path.append("/home/unitree/go1_gym/go1_gym_deploy")
from lcm_types.camera_message_lcmt import camera_message_lcmt
#from realsense import A1RealSense


def realsense_callback(channel, msg):
    img_str = camera_message_lcmt.decode(msg)
    utf_str = img_str.data
    img = np.frombuffer(utf_str, dtype=np.uint16)
    
    import imageio
    counter = 0
    while counter < 5:
        print(f"received message from: {channel}")
        print(img.shape)
        imageio.imwrite(f"test_{counter}.png", img.reshape((64,64)))
        counter += 1

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
subscription = lc.subscribe("realsense_test", realsense_callback)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass

lc.unsubsribe(subscription)
