// SPDX-License-Identifier: MIT
/**
 * @file      camera_model.cc
 * @brief     Run-time dispatch of the camera models.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-05
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/sensor/camera_model.h"

#include "vio/sensor/equirectangular_camera_model.h"
#include "vio/sensor/fisheye_equidistant_camera_model.h"
#include "vio/sensor/pinhole_radtan_camera_model.h"

#include <stdexcept>

namespace vio {
namespace {

[[noreturn]] void UnknownModel(CameraModelId model_id) {
  throw std::invalid_argument("unknown camera model id " + std::to_string(static_cast<int>(model_id)));
}

}  // namespace

CameraModelId CameraModelFromName(const std::string& name) {
#define ON_CAMERA_MODEL(CameraModel) \
  if (name == CameraModel::kName) return CameraModel::kId;
  ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
  return CameraModelId::kInvalid;
}

std::string CameraModelName(CameraModelId model_id) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel) \
  case CameraModel::kId:             \
    return CameraModel::kName;
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      return "INVALID";
  }
}

int CameraModelNumParameters(CameraModelId model_id) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel) \
  case CameraModel::kId:             \
    return CameraModel::kNumParameters;
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      return -1;
  }
}

int CameraModelNumOnlineParams(CameraModelId model_id) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel) \
  case CameraModel::kId:             \
    return CameraModel::kNumOnlineParams;
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      return -1;
  }
}

int CameraModelNumSensors(CameraModelId model_id) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel) \
  case CameraModel::kId:             \
    return CameraModel::kNumSensors;
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      return -1;
  }
}

bool CameraModelWrapsHorizontally(CameraModelId model_id) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel) \
  case CameraModel::kId:             \
    return CameraModel::kWrapsHorizontally;
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      UnknownModel(model_id);
  }
}

void CameraModelPlainParams(CameraModelId model_id, std::vector<double>* params) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel)            \
  case CameraModel::kId:                        \
    CameraModel::PlainParams(params->data());   \
    return;
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      UnknownModel(model_id);
  }
}

Eigen::Vector2d CameraModelProject(CameraModelId model_id, int width, int height, const std::vector<double>& params,
                                   const Eigen::Vector3d& point, double* d_pixel_d_params) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel) \
  case CameraModel::kId:             \
    return CameraModel::Project(width, height, params.data(), point, d_pixel_d_params);
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      UnknownModel(model_id);
  }
}

Eigen::Vector3d CameraModelUnproject(CameraModelId model_id, int width, int height, const std::vector<double>& params,
                                     const Eigen::Vector2d& pixel) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel) \
  case CameraModel::kId:             \
    return CameraModel::Unproject(width, height, params.data(), pixel);
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      UnknownModel(model_id);
  }
}

bool CameraModelProjectionJacobian(CameraModelId model_id, int width, int height, const std::vector<double>& params,
                                   const Eigen::Vector3d& point, Eigen::Matrix<double, 2, 3>* J) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel) \
  case CameraModel::kId:             \
    return CameraModel::ProjectionJacobian(width, height, params.data(), point, J);
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      UnknownModel(model_id);
  }
}

Eigen::Vector2d CameraModelResidual(CameraModelId model_id, int width, int height, const Eigen::Vector2d& measured,
                                    const Eigen::Vector2d& predicted) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel) \
  case CameraModel::kId:             \
    return CameraModel::Residual(width, height, measured, predicted);
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      UnknownModel(model_id);
  }
}

int CameraModelSensorOf(CameraModelId model_id, const Eigen::Vector3d& ray) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel) \
  case CameraModel::kId:             \
    return CameraModel::SensorOf(ray);
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      UnknownModel(model_id);
  }
}

double CameraModelPixelsPerRadian(CameraModelId model_id, int width, int height, const std::vector<double>& params) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel) \
  case CameraModel::kId:             \
    return CameraModel::PixelsPerRadian(width, height, params.data());
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      UnknownModel(model_id);
  }
}

Eigen::Vector3f CameraModelPixelToRay(CameraModelId model_id, int width, int height, const std::vector<double>& params,
                                      const cv::Point2f& pixel) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel) \
  case CameraModel::kId:             \
    return CameraModel::PixelToRay(width, height, params.data(), pixel);
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      UnknownModel(model_id);
  }
}

cv::Point2f CameraModelRayToPixel(CameraModelId model_id, int width, int height, const std::vector<double>& params,
                                  const Eigen::Vector3f& ray) {
  switch (model_id) {
#define ON_CAMERA_MODEL(CameraModel) \
  case CameraModel::kId:             \
    return CameraModel::RayToPixel(width, height, params.data(), ray);
    ALL_CAMERA_MODELS
#undef ON_CAMERA_MODEL
    default:
      UnknownModel(model_id);
  }
}

}  // namespace vio
