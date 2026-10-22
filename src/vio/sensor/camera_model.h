// SPDX-License-Identifier: MIT
/**
 * @file      camera_model.h
 * @brief     Camera model identifiers, the list of models and their run-time dispatch.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-05
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * Every camera model is a struct with static functions (no state): the image size and the model parameters are passed
 * in, and the code outside the models never assumes a projection. A model provides
 *   kId, kName, kNumParameters          identity and number of parameters
 *   kNumOnlineParams                    leading parameters the estimator may refine online (0: all fixed)
 *   kNumSensors                         sensors (lenses) that one image combines; SensorOf tells which one saw a ray
 *   kWrapsHorizontally                  the image continues across its left and right edges
 *   PlainParams                         the parameters without their online corrections (the initializer's model)
 *   Project / Unproject                 double precision (estimation)
 *   ProjectionJacobian                  d pixel / d point
 *   Residual                            measured minus predicted pixel (e.g. wrapped across a seam)
 *   PixelsPerRadian                     image scale at the image center
 *   PixelToRay / RayToPixel             single precision (tracking front end)
 * A new model: one struct, one ON_CAMERA_MODEL line below, one enum value.
 */

#pragma once

#include <Eigen/Core>
#include <opencv2/core/types.hpp>

#include <string>
#include <vector>

namespace vio {

enum class CameraModelId {
  kInvalid = -1,
  kEquirectangular = 0,
  kPinholeRadtan = 1,
  kFisheyeEquidistant = 2,
};

// Every model, as ON_CAMERA_MODEL(ModelStruct); the dispatch functions expand it into a switch
#define ALL_CAMERA_MODELS                      \
  ON_CAMERA_MODEL(EquirectangularCameraModel) \
  ON_CAMERA_MODEL(PinholeRadtanCameraModel)   \
  ON_CAMERA_MODEL(FisheyeEquidistantCameraModel)

// Base of every camera model
template <typename CameraModel>
struct CameraModelBase {
  static bool HasValidParams(const std::vector<double>& params) {
    return static_cast<int>(params.size()) == CameraModel::kNumParameters;
  }
};

CameraModelId CameraModelFromName(const std::string& name);
std::string CameraModelName(CameraModelId model_id);
int CameraModelNumParameters(CameraModelId model_id);
int CameraModelNumOnlineParams(CameraModelId model_id);
int CameraModelNumSensors(CameraModelId model_id);
bool CameraModelWrapsHorizontally(CameraModelId model_id);

// Set the parameters to the model without its online corrections
void CameraModelPlainParams(CameraModelId model_id, std::vector<double>* params);

/**
 * @brief Pixel of a point in the camera frame (outside the image for a point the model cannot image)
 * @param d_pixel_d_params Output 2 x kNumParameters derivative, column-major (null: not computed)
 */
Eigen::Vector2d CameraModelProject(CameraModelId model_id, int width, int height, const std::vector<double>& params,
                                   const Eigen::Vector3d& point, double* d_pixel_d_params = nullptr);

/**
 * @brief Unit ray of a pixel in the camera frame
 */
Eigen::Vector3d CameraModelUnproject(CameraModelId model_id, int width, int height, const std::vector<double>& params,
                                     const Eigen::Vector2d& pixel);

/**
 * @brief Derivative of the pixel w.r.t. the point
 * @return false where the derivative is undefined (or the point cannot be imaged)
 */
bool CameraModelProjectionJacobian(CameraModelId model_id, int width, int height, const std::vector<double>& params,
                                   const Eigen::Vector3d& point, Eigen::Matrix<double, 2, 3>* J);

Eigen::Vector2d CameraModelResidual(CameraModelId model_id, int width, int height, const Eigen::Vector2d& measured,
                                    const Eigen::Vector2d& predicted);
int CameraModelSensorOf(CameraModelId model_id, const Eigen::Vector3d& ray);
double CameraModelPixelsPerRadian(CameraModelId model_id, int width, int height, const std::vector<double>& params);
Eigen::Vector3f CameraModelPixelToRay(CameraModelId model_id, int width, int height, const std::vector<double>& params,
                                      const cv::Point2f& pixel);
cv::Point2f CameraModelRayToPixel(CameraModelId model_id, int width, int height, const std::vector<double>& params,
                                  const Eigen::Vector3f& ray);

}  // namespace vio
