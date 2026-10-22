# References

Methods used in the code and the papers they follow. Each file also lists its references in its header comment.

## By function

| Code | Method | Reference |
|---|---|---|
| `geometry/lie_group.h` | JPL quaternions, rotation conventions | [Trawny 2005] |
| `geometry/triangulation.cc` `TriangulatePoint` | linear estimate: the point closest to all rays (least squares) | [Hartley 2004] Ch. 12 |
| `geometry/triangulation.cc` `InverseDepthProblem` | reprojection error in anchored inverse depth (alpha, beta, rho) | [Mourikis 2007] Appendix |
| `geometry/triangulation.cc` `MinimizeInverseDepth` | Levenberg-Marquardt damping of the Gauss-Newton matrix | [Marquardt 1963] |
| `geometry/inverse_depth.cc` | inverse-depth points | [Civera 2008] |
| `sensor/imu_interval.cc` | IMU samples of an interval, linear between samples | — |
| `estimators/imu_propagation.cc` `IntegrateMean` | IMU mean: JPL kinematics integrated with the classical Runge-Kutta method | [Trawny 2005], [Hairer 1993] Ch. II |
| `estimators/imu_propagation.cc` `IntegrateRotation` | closed-form integrals Xi_1..Xi_4 over an interval | [Yang 2020] |
| `estimators/imu_propagation.cc` `PropagateInterval` | error-state transition at first estimates, discrete noise | [Li 2013], [Yang 2020] |
| `estimators/imu_propagation.cc` `PropagateToImage` | covariance propagation; clone with the time-offset Jacobian | [Mourikis 2007] Sec. III-B, III-C; [Li 2014] |
| `estimators/visual_update.cc` `LinearizeMeasurement` | measurement model, Jacobians at first estimates | [Mourikis 2007] Sec. III-D; [Li 2013] |
| `estimators/visual_update.cc` `ProjectOutPoint`, `CompressRows` | projection onto the left nullspace of the point Jacobian; QR compression | [Mourikis 2007] Sec. III-E; [Golub 2013] §5.2 |
| `estimators/visual_update.cc` `InitializeLandmarks`, `UpdateLandmarks` | landmarks in the state next to the window; initialization from the QR-separated rows | [Li 2012] |
| `estimators/visual_update.cc` `InitializeLandmarks` | inverse-depth landmarks; linearity index of the depth | [Civera 2008] |
| `estimators/estimator.cc` `KalmanUpdate`, `ApplyCorrection` | Kalman update, small-angle quaternion correction | [Maybeck 1979] Ch. 5; [Trawny 2005] |
| `estimators/estimator.cc` `ChiSquaredUpperTail`, `ChiSquaredQuantile` | 95 % chi-square gate: closed forms of the tail for integer degrees of freedom | [Abramowitz 1964] §26.4 |
| `estimators/estimator.h` | sliding-window filter with nullspace-projected track updates | [Mourikis 2007] |
| `estimators/estimator.h` | online camera-IMU time offset | [Li 2014] |
| `estimators/initializer.cc` `Preintegration`, `ImuPreintegrationCost` | closed-form preintegration with piecewise-constant measurements, its residual | [Eckenhoff 2019] |
| `estimators/initializer.cc` `ChoosePoseTimes`, `SolveLinearStart`, `RefineStart` | moving start: linear system in the first IMU frame, maximum-likelihood refinement | [Geneva 2023] |
| `estimators/initializer.cc` `SolveGravityConstrainedLeastSquares` | gravity magnitude by a Lagrange multiplier; polynomial roots from the companion matrix | [Dong-Si 2012], [Edelman 1995] |
| `estimators/initializer.cc` `StartAtRest` | start at rest: gravity from the mean specific force, gyro bias from the mean rate | [Groves 2013] Ch. 5 |
| `feature/feature_tracker.h` | corner detection | [Shi 1994] |
| `feature/feature_tracker.h` | pyramidal optical flow (OpenCV) | [Lucas 1981] |
| `sensor/pinhole_radtan_camera_model.cc` | pinhole projection with radial and decentering (tangential) distortion | [Brown 1966], [Brown 1971] |
| `sensor/fisheye_equidistant_camera_model.cc` | equidistant fisheye projection with a polynomial in the incidence angle | [Kannala 2006] |

## Papers

| Key | Reference |
|---|---|
| [Abramowitz 1964] | M. Abramowitz, I. A. Stegun, "Handbook of Mathematical Functions", National Bureau of Standards, 1964 |
| [Brown 1966] | D. C. Brown, "Decentering Distortion of Lenses", Photogrammetric Engineering 32(3), 1966 |
| [Brown 1971] | D. C. Brown, "Close-Range Camera Calibration", Photogrammetric Engineering 37(8), 1971 |
| [Civera 2008] | J. Civera, A. J. Davison, J. M. M. Montiel, "Inverse Depth Parametrization for Monocular SLAM", IEEE T-RO 24(5), 2008 |
| [Dong-Si 2012] | T.-C. Dong-Si, A. I. Mourikis, "Estimator initialization in vision-aided inertial navigation with unknown camera-IMU calibration", IROS 2012 |
| [Eckenhoff 2019] | K. Eckenhoff, P. Geneva, G. Huang, "Closed-form preintegration methods for graph-based visual-inertial navigation", IJRR 38(5), 2019 |
| [Edelman 1995] | A. Edelman, H. Murakami, "Polynomial roots from companion matrix eigenvalues", Mathematics of Computation 64(210), 1995 |
| [Geneva 2023] | P. Geneva, G. Huang, "Revisiting Visual-Inertial Structure-from-Motion for Odometry and SLAM Initialization", IEEE RA-L 8(4), 2023 |
| [Golub 2013] | G. H. Golub, C. F. Van Loan, "Matrix Computations", 4th ed., Johns Hopkins University Press, 2013 |
| [Groves 2013] | P. D. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems", 2nd ed., Artech House, 2013 |
| [Hairer 1993] | E. Hairer, S. P. Norsett, G. Wanner, "Solving Ordinary Differential Equations I", 2nd ed., Springer, 1993 |
| [Hartley 2004] | R. Hartley, A. Zisserman, "Multiple View Geometry in Computer Vision", 2nd ed., Cambridge University Press, 2004 |
| [Kannala 2006] | J. Kannala, S. S. Brandt, "A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses", IEEE TPAMI 28(8), 2006 |
| [Li 2012] | M. Li, A. I. Mourikis, "Optimization-Based Estimator Design for Vision-Aided Inertial Navigation", RSS 2012 |
| [Li 2013] | M. Li, A. I. Mourikis, "High-precision, consistent EKF-based visual-inertial odometry", IJRR 32(6), 2013 |
| [Li 2014] | M. Li, A. I. Mourikis, "Online temporal calibration for camera-IMU systems: Theory and algorithms", IJRR 33(7), 2014 |
| [Lucas 1981] | B. D. Lucas, T. Kanade, "An Iterative Image Registration Technique with an Application to Stereo Vision", IJCAI 1981 |
| [Marquardt 1963] | D. W. Marquardt, "An Algorithm for Least-Squares Estimation of Nonlinear Parameters", SIAM J. Appl. Math. 11(2), 1963 |
| [Maybeck 1979] | P. S. Maybeck, "Stochastic Models, Estimation, and Control", Vol. 1, Academic Press, 1979 |
| [Mourikis 2007] | A. I. Mourikis, S. I. Roumeliotis, "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation", ICRA 2007 |
| [Shi 1994] | J. Shi, C. Tomasi, "Good Features to Track", CVPR 1994 |
| [Trawny 2005] | N. Trawny, S. I. Roumeliotis, "Indirect Kalman Filter for 3D Attitude Estimation", Tech. Rep. 2005-002, University of Minnesota, 2005 |
| [Yang 2020] | Y. Yang, B. P. W. Babu, C. Chen, G. Huang, L. Ren, "Analytic Combined IMU Integration (ACI^2) for Visual Inertial Navigation", ICRA 2020 |

## Dataset

| Data | Reference |
|---|---|
| PanoAir | Y. Wu, X. Zhang, Y. Du, T. Zhang, C. Li, S. Chen, G. Zhang, X. Xu, "PanoAir: A Panoramic Visual-Inertial SLAM with Cross-Time Real-World UAV Dataset", arXiv:2604.00852, 2026, https://github.com/YYWumm/PanoAir |
| EuRoC MAV | M. Burri, J. Nikolic, P. Gohl, T. Schneider, J. Rehder, S. Omari, M. W. Achtelik, R. Siegwart, "The EuRoC micro aerial vehicle datasets", IJRR 35(10), 2016, doi:10.3929/ethz-b-000690084 (data) |
| TUM-VI | D. Schubert, T. Goll, N. Demmel, V. Usenko, J. Stückler, D. Cremers, "The TUM VI Benchmark for Evaluating Visual-Inertial Odometry", IROS 2018 |
