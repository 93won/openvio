# Architecture

## Layers

```
apps/        vio (odometry, gui)
ui/          Viewer
pipeline/    OdometryPipeline (dataset, IMU/image order, TUM output, callbacks), ConfigLoader
odometry/    VisualInertialOdometry (one image in a fixed order), FrameReport
estimators/  Estimator, Initializer, StartupCalibration
feature/     FeatureTracker, LandmarkTracker
scene/       Camera, Frame (Feature), Keyframe
sensor/      camera models and their dispatch, Rig, IMU samples of an interval
geometry/    SO(3) and quaternions, triangulation, inverse depth
util/        logging, basic types
```

A module includes only modules below it. A header exists when another module includes it or when it is core geometry;
everything else lives in a .cc file (the estimator's internals are split over `estimator.cc`, `imu_propagation.cc`
and `visual_update.cc`; the initializer's preintegration and cost functions are inside `initializer.cc`).

## One image

`OdometryPipeline::Run()` feeds the IMU up to one sample past the image's latest exposure time
(`VisualInertialOdometry::CaptureTime()`), then calls `VisualInertialOdometry::ProcessImage()`:

1. `FeatureTracker` tracks the features of the previous frame (optical flow, forward-backward check, trackable region,
   robust rotation check, light clusters, per-cell cap) and detects new corners; it records why each lost feature was
   lost.
2. Once initialized, `LandmarkTracker` searches the landmarks the tracker lost at their pixels predicted by the
   `Estimator` and adds the matches to the frame under the landmark id.
3. Before the initialization, `StartupCalibration` and `Initializer` take the frame; on success the `Estimator`
   is initialized, its time offsets and the accel bias along gravity are set from the startup data, and the image ends.
4. Afterwards, `Estimator::ProcessImage()`: the observations enter the track database, the state is propagated to the
   exposure and cloned, tracks leaving the window update the clones, landmarks are updated, long tracks become new
   landmarks, the oldest clone leaves the window (a keyframe), the IMU buffer is trimmed.
5. `LandmarkTracker` stores a patch of every landmark updated with the image.

## Callbacks

| Callback | When | Data |
|---|---|---|
| `kInitializedCallback` | the estimator was initialized with the image | `odometry().report().initialization` |
| `kFrameCallback` | every processed image | `odometry().report()`, `odometry().estimator()`, `image()` |
| `kKeyframeCallback` | the oldest clone left the window with its final estimate | `keyframe()`: time, pose, 6x6 covariance, observations, rays |

The viewer registers `kFrameCallback`. A loop-closure component would register `kKeyframeCallback`.

## Determinism

The build fixes the rounding (`-ffp-contract=off`, `-march=x86-64-v3`), the solvers run on one thread (OpenCV's image
functions may use more, with results independent of the split), the rotation check uses a fixed random seed, and:

- Eigen blocks large matrix products by fixed cache sizes (`EIGEN_NO_CPUID`). Otherwise it reads the cache sizes of the
  core that runs the first product; the two core types of a hybrid CPU report different sizes, so the summation order
  and the last digits depended on where a run started.
- The initializer keeps all its Ceres parameter blocks in one array in a fixed order. Ceres orders the blocks of a
  covariance by memory address; with blocks scattered over the heap the initial covariance changed in the last digits
  (~1e-15) with any change of the memory allocations.

A build gives the same trajectory, byte for byte, on every run of a dataset.
