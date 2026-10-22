# Camera models

A camera model is a struct of static functions in `src/vio/sensor/` (no state; the image size and the parameters are
passed in). The `Camera` (`scene/camera.h`) holds the model id, the image size and the parameters, and calls the model
through the run-time dispatch of `sensor/camera_model.h` (a switch over `ALL_CAMERA_MODELS`). The code outside the
models never assumes a projection: it goes through `Camera`, which passes the parameters to every function.

| Member | Meaning |
|---|---|
| `kId`, `kName`, `kNumParameters` | identity (the name is `camera.model` in the config), number of parameters |
| `kNumOnlineParams` | leading parameters the estimator may refine online (0: all calibrated) |
| `kNumSensors` | sensors (lenses) that one image combines |
| `kWrapsHorizontally` | the image continues across its left and right edges |
| `PlainParams(params)` | sets the parameters of the model without its online corrections |
| `Project(width, height, params, point, d_pixel_d_params)` | pixel of a point in the camera frame (outside the image if the model cannot image it), optionally with the derivative w.r.t. the parameters |
| `Unproject(width, height, params, pixel)` | ray of a pixel |
| `ProjectionJacobian(width, height, params, point, J)` | d pixel / d point; false where it is undefined |
| `Residual(width, height, measured, predicted)` | measured minus predicted pixel (wrapped across a seam if the model has one) |
| `SensorOf(ray)` | the sensor that images a ray |
| `PixelsPerRadian(width, height, params)` | image scale at the image center (search windows, patch sizes, rest check) |
| `PixelToRay`, `RayToPixel` | single-precision ray and pixel (feature tracker, landmark tracker) |

How the model properties are used:

| Property | Used by |
|---|---|
| `kNumOnlineParams` | `estimator.camera_param_sigma > 0` adds that many states after the extrinsics (one prior sigma for all); a model without online parameters needs `camera_param_sigma: 0` |
| `kNumSensors` | per-sensor time offsets and sensor poses (`SensorOf`), the startup time-offset estimate of each sensor |
| `kWrapsHorizontally` | the tangent-plane patches of the landmark tracker sample across the left/right edge, else zero outside the image |
| `PlainParams` | the initializer and the startup calibration use `Camera::Plain()`, the model before the estimator refines its online parameters |

## Equirectangular (`EQUIRECTANGULAR`)

An equirectangular image stitched from two lenses: the front lens (sensor 0) looks along +z, the back lens (sensor 1)
sees the rays with z <= 0. Camera frame: x right, y down, z forward.

- `u = W (0.5 + atan2(x, z) / 2pi)`, `v = H (0.5 + asin(y / |p|) / pi)`; the residual wraps `u` into [-W/2, W/2].
- One parameter k, the lens-angle error of the stitching: a ray at angle theta from its lens axis is imaged at
  `theta + k theta (1 - (2 theta / pi)^2)` (zero at the axis and at the seam). k = 0 is the plain projection
  (`PlainParams`). k is online (`kNumOnlineParams = 1`); `camera.params` gives its start value.
- The derivative w.r.t. the point, the image scale and the tracking front end use the plain projection (k is small).
- The rig gives each lens its time offset (estimated).

## Pinhole with radial-tangential distortion (`PINHOLE_RADTAN`)

Parameters `[fx, fy, cx, cy, k1, k2, p1, p2]`, all calibrated. With `x = X / Z`, `y = Y / Z`, `r^2 = x^2 + y^2`:

- `x_d = x (1 + k1 r^2 + k2 r^4) + 2 p1 x y + p2 (r^2 + 2 x^2)`, `y_d = y (1 + k1 r^2 + k2 r^4) + p1 (r^2 + 2 y^2) + 2 p2 x y`
- `u = fx x_d + cx`, `v = fy y_d + cy`; points with Z <= 0 cannot be imaged. The ray of a pixel inverts the distortion
  by Gauss-Newton; the image scale is `sqrt(fx fy)`. One sensor; the image does not wrap.

## Fisheye with the equidistant projection (`FISHEYE_EQUIDISTANT`)

Parameters `[fx, fy, cx, cy, k1, k2, k3, k4]`, all calibrated. A point at angle `theta = atan2(sqrt(X^2 + Y^2), Z)` from
the optical axis is imaged at the radius `theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)` in
its azimuth: `x_d = theta_d X / r`, `y_d = theta_d Y / r`, `u = fx x_d + cx`, `v = fy y_d + cy`. Points beyond 90
degrees are imaged as long as the lens sees them. The ray of a pixel inverts the polynomial by Newton steps; the image
scale is `sqrt(fx fy)`. One sensor; the image does not wrap.

## Adding a model

1. A struct `XxxCameraModel : public CameraModelBase<XxxCameraModel>` with the members above, in its own header and
   source file under `src/vio/sensor/`.
2. A value in `enum class CameraModelId` and a line `ON_CAMERA_MODEL(XxxCameraModel)` in `ALL_CAMERA_MODELS`.
3. Its source file in `CMakeLists.txt`, and checks that `Project(Unproject(pixel))` returns the pixel and that the
   derivatives w.r.t. the point and the parameters match central differences.
