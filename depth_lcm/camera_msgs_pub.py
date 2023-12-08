import lcm
import threading
import time
import select

import numpy as np
import sys
sys.path.append("/home/nano/walk-these-ways")
from go1_gym_deploy.lcm_types.camera_message_rect_wide import camera_message_rect_wide
from go1_gym_deploy.lcm_types.camera_message_lcmt import camera_message_lcmt
from realsense import Go1RealSense


class UnitreeLCMInspector:
    def __init__(self, lc):
        self.lc = lc

        self.cam_sub = self.lc.subscribe("realsense_test", self.realsense_callback)

    def realsense_callback(self, channel, msg):
        img_str = camera_message_lcmt.decode(msg)
        img = np.fromstring(img_str.data, dtype=np.uint8)
        print(img)

    def _rect_camera_cb(self, channel, data):

        # message_types = [camera_message_rect_front, camera_message_rect_front_chin, camera_message_rect_left,
        #                  camera_message_rect_right, camera_message_rect_rear_down]
        # image_shapes = [(200, 200, 3), (100, 100, 3), (100, 232, 3), (100, 232, 3), (200, 200, 3)]

        message_types = [camera_message_rect_wide, camera_message_rect_wide, camera_message_rect_wide, camera_message_rect_wide, camera_message_rect_wide]
        image_shapes = [(116, 100, 3), (116, 100, 3), (116, 100, 3), (116, 100, 3), (116, 100, 3)]

        cam_name = channel.split("_")[-1]

        cam_id = self.camera_names.index(cam_name) + 1

        print(channel, message_types[cam_id - 1])
        msg = message_types[cam_id - 1].decode(data)

        img = np.fromstring(msg.data, dtype=np.uint8)
        img = np.flip(np.flip(
            img.reshape((image_shapes[cam_id - 1][2], image_shapes[cam_id - 1][1], image_shapes[cam_id - 1][0])),
            axis=0), axis=1).transpose(1, 2, 0)

        if cam_id == 1:
            self.camera_image_front = img
        elif cam_id == 2:
            self.camera_image_bottom = img
        elif cam_id == 3:
            self.camera_image_left = img
        elif cam_id == 4:
            self.camera_image_right = img
        elif cam_id == 5:
            self.camera_image_rear = img
        else:
            print("Image received from camera with unknown ID#!")

        print(f"f{1. / (time.time() - self.ts[cam_id - 1])}: received py from {cam_name}!")
        self.ts[cam_id-1] = time.time()

        from PIL import Image
        im = Image.fromarray(img)
        im.save(f"{cam_name}_image.jpeg")

    def publish_depth(self):
        msg = camera_message_lcmt()
        cam = Go1RealSense()
        cam.find_camera()
        cam.pipeline.start()
        print("pipeline started")
        cam.continue_execution = True

        cam.streaming()
        print("start publishing")
        while True:
            cam.streaming()
            depth_img = cam.get_depth_frame().reshape(1,-1)
            img_str = depth_img.tobytes()
            msg.data = img_str

            print(f"depth img: {depth_img}")

            self.lc.publish("depth_image", msg.encode())

    def poll(self, cb=None):
        t = time.time()
        try:
            while True:
                timeout = 0.01
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                if rfds:
                    # print("message received!")
                    self.lc.handle()
                    # print(f'Freq {1. / (time.time() - t)} Hz'); t = time.time()
                else:
                    continue
                    # print(f'waiting for message... Freq {1. / (time.time() - t)} Hz'); t = time.time()
                #    if cb is not None:
                #        cb()
        except KeyboardInterrupt:
            pass

    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        self.run_thread.start()

if __name__ == "__main__":
    import lcm

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=254")
    # lc = lcm.LCM()
    print("init")
    insp = UnitreeLCMInspector(lc)
    insp.publish_depth()