# OpenVIO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[![OpenVIO demo](doc/media/openvio_demo.png)](https://github.com/93won/openvio/releases/download/v1.0.0/openvio_demo.mp4)

**Release notes:** [v1.0.0](release_notes/v1.0.0.pdf)

## How to run

Dependencies:

- C++17, CMake >= 3.16
- OpenCV >= 4.0, Eigen3 >= 3.3, libpng
- Ceres Solver and spdlog (included in `thirdparty/`)
- GLFW >= 3.3 and OpenGL for the viewer (optional)

Build, get a dataset, run and evaluate: [doc/usage.md](doc/usage.md)

## Benchmark Datasets

```bibtex
@article{wu2026panoair,
  title   = {PanoAir: A Panoramic Visual-Inertial SLAM with Cross-Time Real-World UAV Dataset},
  author  = {Wu, Yiyang and Zhang, Xiaohu and Du, Yanjin and Zhang, Tongsu and Li, Chujun and Chen, Siyang and
             Zhang, Guoyi and Xu, Xiangpeng},
  journal = {arXiv preprint arXiv:2604.00852},
  year    = {2026}
}

@article{burri2016euroc,
  title   = {The EuRoC micro aerial vehicle datasets},
  author  = {Burri, Michael and Nikolic, Janosch and Gohl, Pascal and Schneider, Thomas and Rehder, Joern and
             Omari, Sammy and Achtelik, Markus W and Siegwart, Roland},
  journal = {The International Journal of Robotics Research},
  volume  = {35},
  number  = {10},
  pages   = {1157--1163},
  year    = {2016}
}

@inproceedings{schubert2018tumvi,
  title     = {The TUM VI Benchmark for Evaluating Visual-Inertial Odometry},
  author    = {Schubert, David and Goll, Thore and Demmel, Nikolaus and Usenko, Vladyslav and St{\"u}ckler, J{\"o}rg and
               Cremers, Daniel},
  booktitle = {2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {1680--1687},
  year      = {2018}
}
```
