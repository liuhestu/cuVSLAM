import depthai as dai
import time

def extract_ns(timestamp):
    if hasattr(timestamp, "total_seconds"):
        return int(timestamp.total_seconds() * 1e9)
    return None

pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.Camera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
imu = pipeline.create(dai.node.IMU)
imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 200)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)

cam_out = pipeline.create(dai.node.XLinkOut)
cam_out.setStreamName("cam")
cam.out.link(cam_out.input)

imu_out = pipeline.create(dai.node.XLinkOut)
imu_out.setStreamName("imu")
imu.out.link(imu_out.input)

with dai.Device(pipeline) as device:
    q_cam = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_imu = device.getOutputQueue(name="imu", maxSize=4, blocking=False)
    
    for i in range(5):
        frame = q_cam.get()
        print("Camera device ts:", extract_ns(frame.getTimestampDevice()))
        # print("Camera host ts:", extract_ns(frame.getTimestamp()))
        try:
            imu_msg = q_imu.get()
            for packet in imu_msg.packets:
                print("IMU device ts:", extract_ns(packet.getTimestampDevice()))
        except Exception as e:
            pass

