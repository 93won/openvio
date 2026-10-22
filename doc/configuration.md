# Configuration

`vio` reads one YAML file (`--config_path`); `config/panoair.yaml` explains every key in a comment.

| Section | Options struct | Used by |
|---|---|---|
| `camera` | `Camera` | every module (model, size, parameters, untracked regions) |
| `camera1` | `Camera` | stereo only: the second camera (same keys as `camera`; leave the section out for a monocular rig) |
| `imu` | `ImuNoise`, gravity | initializer, estimator, startup calibration |
| `rig` | `Rig` | camera-IMU extrinsics (`imu_from_camera`, stereo also `imu_from_camera1`), clock offset |
| `feature_tracker` | `FeatureTrackerOptions` | `FeatureTracker` |
| `landmark_tracker` | `LandmarkTrackerOptions` | `LandmarkTracker` (used when `estimator.landmark_dormant_time > 0`) |
| `initializer` | `InitializerOptions` | `Initializer`, `StartupCalibration` |
| `estimator` (+ `estimator.triangulation`) | `EstimatorOptions` (+ `TriangulationOptions`) | `Estimator` |

Rules:

- A key missing from the file keeps the default of its struct.
- `--section.key=value` on the command line replaces one key (nested: `--estimator.triangulation.max_depth=100`). The
  value is parsed by the same YAML reader as the file, so the same text gives the same number; a sequence is written
  as `[a, b]`. An override that names no option stops the run.
- Every options struct has `Check()`; the run stops with the violated condition when one fails.
- Values that were single precision in the reference runs are still rounded through float when read: the IMU noise
  densities and gravity, `rig.imu_from_camera` (and `rig.imu_from_camera1`), `camera.polar_exclusion_ratio`, the mask
  rectangles, and the tracker's quality level, minimum distance, rotation threshold and cluster ratio.

Stereo (`config/euroc_stereo.yaml`, `config/tum_vi_stereo.yaml`): a `camera1` section makes the rig stereo. The pair is
fixed: `rig.imu_from_camera1` gives camera 1 relative to camera 0, and the online extrinsic states move both cameras.
Every tracked camera-0 feature is matched into the camera-1 image of the same time stamp (optical flow from the camera-0
pixel, back to camera 0, then `feature_tracker.stereo_max_epipolar_error` [px] and both rays meeting in front of the
cameras); the matches add measurements of the same tracks. A camera-0 image without a camera-1 image is processed
monocular. The initializer uses camera 0 only. Both cameras need one sensor (a pinhole or fisheye model).
