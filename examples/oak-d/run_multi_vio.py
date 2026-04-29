#!/usr/bin/env python3
"""
Multi OAK-D VIO for Rerun 0.22.1 – Stable main-process visualization.
"""

import os, sys, time, signal
from datetime import timedelta
from multiprocessing import Process, Queue, set_start_method
from collections import deque

import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation
import cuvslam as vslam
import h5py
import rerun as rr
import rerun.blueprint as rrb

# ---------- ROS2 (optional) ----------
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from builtin_interfaces.msg import Time
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# ==========  Configuration ==========
FPS = 30
RESOLUTION = (1280, 720)
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_NS = 35 * 1e6
IMU_JITTER_THRESHOLD_NS = 6 * 1e6
IMU_FREQUENCY = 200
BORDER_TOP, BORDER_BOTTOM, BORDER_LEFT, BORDER_RIGHT = 10, 10, 10, 10
CM_TO_METERS = 100

IMU_GYROSCOPE_NOISE_DENSITY = 6.0673370376614875e-03
IMU_GYROSCOPE_RANDOM_WALK = 3.6211951458325785e-05
IMU_ACCELEROMETER_NOISE_DENSITY = 3.3621979208052800e-02
IMU_ACCELEROMETER_RANDOM_WALK = 9.8256589971851467e-04

# ==========  Helper Functions ==========
def oak_transform_to_pose(extr):
    arr = np.array(extr)
    rot = arr[:3,:3]; t = arr[:3,3] / CM_TO_METERS
    return vslam.Pose(rotation=Rotation.from_matrix(rot).as_quat(), translation=t)

def set_cuvslam_camera(params):
    cam = vslam.Camera()
    cam.distortion = vslam.Distortion(vslam.Distortion.Model.Polynomial, params["distortion"])
    cam.focal = (params["intrinsics"][0][0], params["intrinsics"][1][1])
    cam.principal = (params["intrinsics"][0][2], params["intrinsics"][1][2])
    cam.size = params["resolution"]
    cam.rig_from_camera = oak_transform_to_pose(params["extrinsics"])
    cam.border_top = BORDER_TOP; cam.border_bottom = BORDER_BOTTOM
    cam.border_left = BORDER_LEFT; cam.border_right = BORDER_RIGHT
    return cam

def get_stereo_calib(calib, res):
    s = {"left":{},"right":{}}
    for side, sock in zip(["left","right"], [dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C]):
        s[side]["resolution"] = res
        s[side]["intrinsics"] = calib.getCameraIntrinsics(sock, res[0], res[1])
        s[side]["extrinsics"] = calib.getCameraExtrinsics(sock, dai.CameraBoardSocket.CAM_A)
        s[side]["distortion"] = calib.getDistortionCoefficients(sock)[:8]
    return s

def get_imu_calib(extr):
    imu = vslam.ImuCalibration()
    t = np.array(extr)[:3,3] / CM_TO_METERS
    imu.rig_from_imu = vslam.Pose(rotation=np.array([0.,0.,0.,1.]), translation=t)
    imu.gyroscope_noise_density = IMU_GYROSCOPE_NOISE_DENSITY
    imu.gyroscope_random_walk = IMU_GYROSCOPE_RANDOM_WALK
    imu.accelerometer_noise_density = IMU_ACCELEROMETER_NOISE_DENSITY
    imu.accelerometer_random_walk = IMU_ACCELEROMETER_RANDOM_WALK
    imu.frequency = IMU_FREQUENCY
    return imu

def ts_to_ns(ts):
    if ts is None: return None
    if hasattr(ts, 'total_seconds'): return int(ts.total_seconds()*1e9)
    if hasattr(ts, 'seconds') and hasattr(ts, 'microseconds'): return int(ts.seconds*1e9 + ts.microseconds*1000)
    if isinstance(ts, (int,float)): return int(ts*1e9)
    try: return int(ts.total_seconds()*1e9)
    except: return None

def color_from_id(uid): return [(uid*17)%256, (uid*31)%256, (uid*47)%256]

# ==========  ROS2 Publisher ==========
class CameraPosePublisher(Node):
    def __init__(self, cid):
        super().__init__(f'vio_camera_{cid}')
        self.pub = self.create_publisher(PoseStamped, f'cam_{cid}/pose', 10)
        self.cid = cid
    def publish_pose(self, ts_ns, trans, quat_xyzw):
        msg = PoseStamped()
        sec, nsec = int(ts_ns//1e9), int(ts_ns%1e9)
        stamp = Time(); stamp.sec=sec; stamp.nanosec=nsec
        msg.header.stamp = stamp
        msg.header.frame_id = f'cam_{self.cid}_odom'
        msg.pose.position.x = float(trans[0])
        msg.pose.position.y = float(trans[1])
        msg.pose.position.z = float(trans[2])
        msg.pose.orientation.x = float(quat_xyzw[0])
        msg.pose.orientation.y = float(quat_xyzw[1])
        msg.pose.orientation.z = float(quat_xyzw[2])
        msg.pose.orientation.w = float(quat_xyzw[3])
        self.pub.publish(msg)

# ==========  VIO Process (NO RERUN, sends data via Queue) ==========
def vio_process(camera_id, device_id, num_cameras, vis_queue, traj_queue):
    if ROS2_AVAILABLE:
        rclpy.init(args=None)
        ros_node = CameraPosePublisher(camera_id)
    else:
        ros_node = None

    infos = dai.Device.getAllAvailableDevices()
    target = next((d for d in infos if d.deviceId == device_id), None)
    if target is None:
        print(f"[Cam {camera_id}] Device {device_id} not found.")
        traj_queue.put((camera_id, []))
        return

    device = dai.Device(target)
    calib = device.readCalibration()
    stereo = get_stereo_calib(calib, RESOLUTION)
    cams = [set_cuvslam_camera(stereo["left"]), set_cuvslam_camera(stereo["right"])]
    rig = vslam.Rig(cams)
    imu_extr = calib.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_A)
    rig.imus = [get_imu_calib(imu_extr)]

    odom_cfg = vslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        rectified_stereo_camera=False,
        odometry_mode=vslam.Tracker.OdometryMode.Inertial
    )
    slam_cfg = vslam.Tracker.SlamConfig()
    tracker = vslam.Tracker(rig, odom_cfg, slam_config=slam_cfg)

    pipeline = dai.Pipeline(device)
    lcam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=FPS)
    rcam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=FPS)
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(seconds=0.5/FPS))
    lo = lcam.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)
    ro = rcam.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)
    lo.link(sync.inputs["left"]); ro.link(sync.inputs["right"])
    sq = sync.out.createOutputQueue()

    imu_node = pipeline.create(dai.node.IMU)
    imu_node.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], IMU_FREQUENCY)
    imu_node.setBatchReportThreshold(1); imu_node.setMaxBatchReports(10)
    iq = imu_node.out.createOutputQueue()

    pipeline.start()

    frame_id = 0
    prev_cts = None; last_imu_ts = None; last_sent_ts = None
    imu_buf = []; imu_count = 0
    traj_full = []

    try:
        while pipeline.isRunning():
            mg = sq.get()
            lf, rf = mg["left"], mg["right"]
            ts_dev = None
            for m in ('getTimestampDevice','getTimestamp'):
                if hasattr(mg, m):
                    ts_dev = getattr(mg, m)()
                    break
            if ts_dev is None: continue
            ts_ns = ts_to_ns(ts_dev)
            if ts_ns is None: continue

            ims = iq.tryGetAll()
            for im in ims:
                for pkt in im.packets:
                    accel = pkt.acceleroMeter; gyro = pkt.gyroscope
                    ts = None
                    for m in ('getTimestampDevice','getTimestamp','getTimestampHost'):
                        if hasattr(pkt, m): ts = getattr(pkt, m)(); break
                    if ts is None:
                        for sf in ('acceleroMeter','gyroscope'):
                            if hasattr(pkt, sf):
                                so = getattr(pkt, sf)
                                for a in ('tsDevice','timestamp','timestampDevice'):
                                    if hasattr(so, a): ts = getattr(so, a); break
                            if ts is not None: break
                    if ts is None: continue
                    its = ts_to_ns(ts)
                    if its is None: continue
                    imu_buf.append((its, accel, gyro))
            imu_buf.sort(key=lambda x: x[0])

            while imu_buf and imu_buf[0][0] <= ts_ns:
                its, accel, gyro = imu_buf.pop(0)
                if last_sent_ts is not None and its <= last_sent_ts: continue
                if last_imu_ts is not None and (its - last_imu_ts) <= 0: continue
                last_imu_ts = its
                meas = vslam.ImuMeasurement()
                meas.timestamp_ns = its
                meas.linear_accelerations = np.array([accel.x, accel.y, accel.z], dtype=np.float32)
                meas.angular_velocities = np.array([gyro.x, gyro.y, gyro.z], dtype=np.float32)
                tracker.register_imu_measurement(0, meas)
                imu_count += 1
                last_sent_ts = its

            prev_cts = ts_ns; imu_count = 0; frame_id += 1

            if frame_id <= WARMUP_FRAMES: continue

            left_img = lf.getCvFrame(); right_img = rf.getCvFrame()
            odom_est, slam_est = tracker.track(ts_ns, (left_img, right_img))
            pose = slam_est if slam_est is not None else odom_est.world_from_rig
            if pose is None:
                print(f"[Cam {camera_id}] Tracking failed at {frame_id}")
                continue

            trans = pose.translation.copy(); quat = pose.rotation.copy()
            traj_full.append((ts_ns, trans, quat))

            # 序列化特征点
            raw_obs = tracker.get_last_observations(0)
            obs_list = [(o.u, o.v, o.id) for o in raw_obs] if raw_obs else []

            if ros_node:
                ros_node.publish_pose(ts_ns, trans, quat)
                rclpy.spin_once(ros_node, timeout_sec=0)

            # 发送给主进程
            vis_queue.put({
                'camera_id': camera_id,
                'timestamp_ns': ts_ns,
                'left_image': left_img,
                'translation': trans,
                'rotation': quat,
                'observations': obs_list
            })
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop(); device.close()
        if ros_node: ros_node.destroy_node(); rclpy.shutdown()
        traj_queue.put((camera_id, traj_full))
        print(f"[Camera {camera_id}] Finished. {len(traj_full)} poses.")

# ==========  Main Process ==========
def main():
    set_start_method('spawn', force=True)

    # 启动 Rerun Viewer（主进程）
    rr.init("multi_oak_vio", spawn=True)
    time.sleep(2)

    infos = dai.Device.getAllAvailableDevices()
    if not infos:
        print("No OAK-D devices found.")
        return

    num_cameras = len(infos)
    print(f"Found {num_cameras} device(s):")
    for i, info in enumerate(infos):
        print(f"  {i}: {info.name} (deviceId: {info.deviceId})")

    # ---------- 动态蓝图：左侧纵列图像，右侧 3D ----------
    cam_views = []
    for i in range(num_cameras):
        cam_views.append(rrb.Spatial2DView(origin=f"cam_{i}/left", name=f"cam{i}-left"))
    left_col = rrb.Vertical(*cam_views)
    right_3d = rrb.Spatial3DView(name="3D")
    blueprint = rrb.Blueprint(rrb.Horizontal(left_col, right_3d))
    rr.send_blueprint(blueprint)

    # ---------- 偏移量：多个相机轨迹起点分离 ----------
    offsets = {}
    if num_cameras == 1:
        offsets[0] = np.zeros(3)
    elif num_cameras == 2:
        offsets[0] = np.array([-0.5, 0.0, 0.0])
        offsets[1] = np.array([ 0.5, 0.0, 0.0])
    elif num_cameras == 3:
        offsets[0] = np.array([-0.5, -0.5, 0])
        offsets[1] = np.array([ 0.5, -0.5, 0])
        offsets[2] = np.array([ 0.0,  0.5, 0])
    else:
        offsets[0] = np.array([-0.5, -0.5, 0])
        offsets[1] = np.array([ 0.5, -0.5, 0])
        offsets[2] = np.array([-0.5,  0.5, 0])
        offsets[3] = np.array([ 0.5,  0.5, 0])

    # ---------- 启动子进程 ----------
    vis_queue = Queue()
    traj_queue = Queue()
    processes = []

    def handler(s, f):
        print("\nShutting down...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)

    for cid, info in enumerate(infos):
        p = Process(target=vio_process,
                    args=(cid, info.deviceId, num_cameras,
                          vis_queue, traj_queue))
        p.start()
        processes.append(p)

    # ---------- 主可视化循环 ----------
    max_traj_len = 2000
    trajectories = {cid: deque(maxlen=max_traj_len) for cid in range(num_cameras)}
    viz_frame_id = 0
    last_traj_update = {cid: 0 for cid in range(num_cameras)}

    try:
        while any(p.is_alive() for p in processes):
            while not vis_queue.empty():
                data = vis_queue.get()
                cid = data['camera_id']
                ts_ns = data['timestamp_ns']
                left_img = data['left_image']
                trans = data['translation']
                quat = data['rotation']
                observations = data['observations']

                offset = offsets.get(cid, np.zeros(3))
                vis_trans = trans + offset
                trajectories[cid].append(vis_trans)

                rr.set_time_sequence("frame", viz_frame_id)
                viz_frame_id += 1

                prefix = f"cam_{cid}"

                # 1. 左目图像
                rr.log(f"{prefix}/left", rr.Image(left_img).compress(jpeg_quality=80))

                # 2. 当前位姿（带偏移）
                rr.log(f"{prefix}/world/rig", rr.Transform3D(
                    translation=vis_trans,
                    rotation=rr.Quaternion(xyzw=quat)
                ))

                # 3. 轨迹（每 10 帧更新一次，减少数据量）
                if len(trajectories[cid]) > 1 and viz_frame_id - last_traj_update[cid] >= 10:
                    traj_np = np.array(trajectories[cid])
                    rr.log(f"{prefix}/world/trajectory", rr.LineStrips3D([traj_np]))
                    last_traj_update[cid] = viz_frame_id

                # 4. 2D 特征点
                if observations:
                    pts = [[u, v] for (u, v, _) in observations]
                    cols = [color_from_id(oid) for (_, _, oid) in observations]
                    if pts:
                        rr.log(f"{prefix}/left/observations",
                               rr.Points2D(pts, radii=4, colors=cols))
            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            p.join()

    # ---------- 保存 HDF5 ----------
    all_traj = {}
    while not traj_queue.empty():
        cid, data = traj_queue.get()
        all_traj[cid] = data

    with h5py.File("trajectories.h5", "w") as f:
        for cid, data in all_traj.items():
            if not data:
                continue
            timestamps = np.array([d[0] for d in data])
            positions = np.array([d[1] for d in data])
            quats = np.array([d[2] for d in data])
            grp = f.create_group(f"cam_{cid}")
            grp.create_dataset("timestamps", data=timestamps)
            grp.create_dataset("positions", data=positions)
            grp.create_dataset("quaternions_xyzw", data=quats)
            print(f"Saved cam_{cid}: {len(timestamps)} poses.")

    print("All trajectories saved to trajectories.h5")
    print("Done.")

if __name__ == "__main__":
    main()