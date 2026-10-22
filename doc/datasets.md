# Datasets

| Dataset | Camera | Configs | Download |
|---|---|---|---|
| PanoAir (17 drone flights) | 360° equirectangular, two lenses | `config/panoair.yaml` | by hand, from its Google Drive |
| EuRoC MAV (11 sequences) | stereo pinhole | `config/euroc.yaml`, `config/euroc_stereo.yaml` | `scripts/python/download_dataset.py euroc` |
| TUM-VI, 512 x 512 export | stereo fisheye | `config/tum_vi.yaml`, `config/tum_vi_stereo.yaml` | `scripts/python/download_dataset.py tumvi` |

Build, run and evaluate: [usage.md](usage.md).

## Download

### EuRoC MAV and TUM-VI

```bash
python3 scripts/python/download_dataset.py euroc /data/EuRoC                   # all 11 sequences
python3 scripts/python/download_dataset.py tumvi /data/TUM-VI                  # the 20 indoor sequences
python3 scripts/python/download_dataset.py tumvi /data/TUM-VI room1 corridor4  # chosen sequences
```

Every sequence ends up as `SEQ/mav0` with both cameras, the IMU and the ground truth, prepared for `vio` (see
[Folder format](#folder-format)); a sequence that is already there is skipped, and an interrupted EuRoC download
resumes. EuRoC comes from its official archives (ETH Research Collection): only the 0.7-1.6 GB zip of each requested
sequence is fetched out of the room archives, about 11 GB for all 11. TUM-VI comes from the TUM server: the indoor
sequences are 1.1-10.3 GB each (rooms 1.4-1.8 GB), about 81 GB for all 20. The script needs Python 3 only.

Sequences downloaded some other way (the ASL folder format: `SEQ/mav0/cam0`, `cam1`, `imu0` and
`state_groundtruth_estimate0` or `mocap0`) are prepared with

```bash
python3 scripts/python/prepare_asl_dataset.py /data/EuRoC/MH_01_easy /data/TUM-VI/room1
```

### PanoAir

Download the flights `seq1` … `seq17` from the [PanoAir Google Drive](https://drive.google.com/drive/folders/1bpEvIG8duxgAJkuRhN4_eMlmg634q-1l)
(dataset: [github.com/YYWumm/PanoAir](https://github.com/YYWumm/PanoAir)). A flight needs `frame_sequences/panocam/`,
`frame_sequences/rgb.txt`, `frame_sequences/imu0.txt` and `SeqN_GT.txt`; the other image folders and the ROS bag are
not used. Then prepare every flight under the root:

```bash
# Writes cam_timestamps.txt, imu_data.csv and gt.txt into every seqN/frame_sequences/ (the images stay in panocam/)
python3 scripts/python/prepare_panoair.py /data/PanoAir
```

## Folder format

| File | Content |
|---|---|
| `images/` or `panocam/` | PNG images, processed in the order of their names (resized to `camera.width` x `camera.height` if needed) |
| `images1/` (stereo) | camera-1 PNG images; one pairs with the camera-0 image of the same file name (a camera-0 image without one is processed monocular) |
| `cam_timestamps.txt` | one image time per line [s], in the order of the images |
| `imu_data.csv` | a header line, then `timestamp,ax,ay,az,gx,gy,gz` [s, m/s^2, rad/s] |
| `gt.txt` (optional) | ground truth in TUM format (`time x y z qx qy qz qw`), shown by `vio gui` |

The IMU is read in single precision (as in the reference runs). `prepare_asl_dataset.py` writes `images/` (a link to
`cam0/data`), `images1/` (a link to `cam1/data`), `cam_timestamps.txt`, `imu_data.csv` and `gt.txt` (from
`state_groundtruth_estimate0` or `mocap0`) into `SEQ/mav0/`; the folder must allow symbolic links.

## Dataset settings

The EuRoC and TUM-VI configs hold the dataset's own calibration: `config/euroc.yaml` (`PINHOLE_RADTAN`),
`config/tum_vi.yaml` (`FISHEYE_EQUIDISTANT`), and as stereo `config/euroc_stereo.yaml`, `config/tum_vi_stereo.yaml`
(the monocular configs plus `camera1`, `rig.imu_from_camera1` and `feature_tracker.stereo_max_epipolar_error`).
Method settings differ from `config/panoair.yaml` only where their premise was the 360 drone camera, each with its
reason in the config: e.g. both datasets compensate the auto-exposure gain before the optical flow
(`feature_tracker.gain_compensation`), and TUM-VI, being hand-held, never starts at rest
(`initializer.excitation_accel_std: 0.0`) and counts only translation as motion when it decides to start
(`initializer.rotation_compensated_disparity: 1`). Outdoors, TUM-VI makes landmarks only of tracks whose depth the
window determines (`estimator.landmark_max_linearity`: clouds sit at infinity and drift).
