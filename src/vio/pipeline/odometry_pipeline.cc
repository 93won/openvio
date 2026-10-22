// SPDX-License-Identifier: MIT
/**
 * @file      odometry_pipeline.cc
 * @brief     Implements the odometry pipeline.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-05
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/pipeline/odometry_pipeline.h"

#include "vio/geometry/lie_group.h"
#include "vio/util/logging.h"

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>
#include <png.h>

#include <algorithm>
#include <csetjmp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <sstream>

namespace vio {
namespace {

namespace fs = std::filesystem;

/**
 * 8-bit grayscale image of a PNG file, decoded with libpng the way cv::imread(path, cv::IMREAD_GRAYSCALE) does it:
 * 16-bit samples keep their high byte, alpha is dropped, a palette is expanded, and color becomes gray with
 * png_set_rgb_to_gray(0.299, 0.587). Reading with libpng directly keeps OpenCV's image-codec module, and the many
 * libraries it loads (GDAL, GDCM, ...), out of the process. Empty when the file cannot be decoded.
 */
cv::Mat ReadGrayPng(const std::string& path) {
  cv::Mat image;
  std::vector<png_bytep> rows;
  FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return image;
  }
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = (png != nullptr) ? png_create_info_struct(png) : nullptr;
  png_infop end_info = (png != nullptr) ? png_create_info_struct(png) : nullptr;
  if (info == nullptr || end_info == nullptr || setjmp(png_jmpbuf(png)) != 0) {
    // libpng reported an error (or could not start): no image
    image.release();
  } else {
    png_init_io(png, file);
    png_read_info(png, info);
    png_uint_32 width = 0;
    png_uint_32 height = 0;
    int bit_depth = 0;
    int color_type = 0;
    png_get_IHDR(png, info, &width, &height, &bit_depth, &color_type, nullptr, nullptr, nullptr);
    if (bit_depth == 16) {
      png_set_strip_16(png);
    }
    png_set_strip_alpha(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
      png_set_palette_to_rgb(png);
    }
    if ((color_type & PNG_COLOR_MASK_COLOR) == 0 && bit_depth < 8) {
      png_set_expand_gray_1_2_4_to_8(png);
    }
    png_set_rgb_to_gray(png, 1, 0.299, 0.587);
    png_set_interlace_handling(png);
    png_read_update_info(png, info);
    image.create(static_cast<int>(height), static_cast<int>(width), CV_8UC1);
    rows.resize(height);
    for (png_uint_32 y = 0; y < height; ++y) {
      rows[y] = image.ptr<uchar>(static_cast<int>(y));
    }
    png_read_image(png, rows.data());
    png_read_end(png, end_info);
  }
  png_destroy_read_struct(&png, &info, &end_info);
  std::fclose(file);
  return image;
}

struct Dataset {
  std::vector<fs::path> images;
  std::vector<fs::path> images1;  // Stereo: camera-1 image of every camera-0 image (empty path: none)
  std::vector<double> image_times;
  std::vector<ImuSample> imu;
};

// PNG files of a directory, sorted by name
std::vector<fs::path> ListImages(const fs::path& directory) {
  std::vector<fs::path> images;
  std::error_code error;
  for (const auto& entry : fs::directory_iterator(directory, error)) {
    if (entry.is_regular_file() && entry.path().extension() == ".png") {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

bool ReadDataset(const fs::path& root, bool stereo, Dataset* dataset) {
  // Prepared datasets link the images as images/ (and the camera-1 images as images1/); raw sequences keep them in
  // panocam/
  dataset->images = ListImages(fs::exists(root / "images") ? root / "images" : root / "panocam");
  if (stereo) {
    // A camera-1 image pairs with the camera-0 image of the same name (time stamp); without one, the image is
    // processed monocular
    size_t pairs = 0;
    for (const fs::path& image : dataset->images) {
      const fs::path image1 = root / "images1" / image.filename();
      const bool exists = fs::exists(image1);
      dataset->images1.push_back(exists ? image1 : fs::path());
      pairs += exists ? 1 : 0;
    }
    if (pairs == 0) {
      LOG_ERROR("Dataset {}: no camera-1 image in images1/ (stereo config)", root.string());
      return false;
    }
    LOG_INFO("Stereo: {} of {} images have a camera-1 image", pairs, dataset->images.size());
  }

  std::ifstream times(root / "cam_timestamps.txt");
  double t;
  while (times >> t) {
    dataset->image_times.push_back(t);
  }

  // The IMU is read in single precision, as in the reference runs (D9 quirk)
  std::ifstream imu(root / "imu_data.csv");
  std::string line;
  std::getline(imu, line);
  while (std::getline(imu, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    double timestamp;
    float ax, ay, az, gx, gy, gz;
    if (row >> timestamp >> ax >> ay >> az >> gx >> gy >> gz) {
      dataset->imu.push_back({timestamp, Eigen::Vector3d(gx, gy, gz), Eigen::Vector3d(ax, ay, az)});
    }
  }
  if (dataset->images.empty() || dataset->images.size() != dataset->image_times.size() || dataset->imu.size() < 2) {
    LOG_ERROR("Dataset {}: {} images, {} image times, {} IMU samples", root.string(), dataset->images.size(),
              dataset->image_times.size(), dataset->imu.size());
    return false;
  }
  return true;
}

template <typename Derived>
std::string Format(const Eigen::MatrixBase<Derived>& v, int precision = 6) {
  std::ostringstream text;
  text << std::setprecision(precision);
  for (int i = 0; i < v.size(); ++i) {
    text << (i > 0 ? "," : "") << v(i);
  }
  return text.str();
}

// Sums of the per-image statistics over a run
struct RunTotals {
  EstimatorStatistics sum;
  int images = 0;
  double landmarks_in_state_sum = 0.0;

  void Add(const EstimatorStatistics& s) {
    sum.candidates += s.candidates;
    sum.too_few_measurements += s.too_few_measurements;
    sum.triangulation_failed += s.triangulation_failed;
    sum.triangulation_ill_conditioned += s.triangulation_ill_conditioned;
    sum.triangulation_linear_depth += s.triangulation_linear_depth;
    sum.triangulation_refined_depth += s.triangulation_refined_depth;
    sum.triangulation_baseline += s.triangulation_baseline;
    sum.triangulation_not_finite += s.triangulation_not_finite;
    sum.projection_failed += s.projection_failed;
    sum.gate_rejected += s.gate_rejected;
    sum.accepted += s.accepted;
    sum.gate_tested += s.gate_tested;
    sum.nis_sum += s.nis_sum;
    sum.nis_accepted_sum += s.nis_accepted_sum;
    sum.landmarks_updated += s.landmarks_updated;
    sum.landmarks_gate_rejected += s.landmarks_gate_rejected;
    sum.landmarks_lost += s.landmarks_lost;
    sum.landmarks_initialized += s.landmarks_initialized;
    sum.landmarks_relinearized += s.landmarks_relinearized;
    sum.landmarks_dormant += s.landmarks_dormant;
    sum.landmarks_found_again += s.landmarks_found_again;
    sum.landmarks_removed += s.landmarks_removed;
    sum.landmark_life_sum += s.landmark_life_sum;
    sum.landmark_updates_sum += s.landmark_updates_sum;
    for (int k = 0; k < 6; ++k) {
      sum.landmark_life_histogram[k] += s.landmark_life_histogram[k];
    }
    sum.landmarks_in_state = std::max(sum.landmarks_in_state, s.landmarks_in_state);
    landmarks_in_state_sum += s.landmarks_in_state;
    ++images;
  }
};

// Rig calibration relative to the configured values (it moves only when estimated online)
std::string CalibrationText(const VisualInertialOdometryOptions& options, const Estimator& estimator) {
  const Rig& rig = estimator.rig();
  const Eigen::Vector3d dphi = LogSO3(rig.R_ItoC * options.rig.R_ItoC.transpose()) * 180.0 / M_PI;
  const Eigen::Vector3d dp = 100.0 * (rig.p_IinC - options.rig.p_IinC);
  std::ostringstream text;
  text << "time_offset=" << rig.time_offset << " sensor_time_offset=" << rig.sensor_time_offset
       << " extrinsic_rotation_deg=" << Format(dphi) << " extrinsic_translation_cm=" << Format(dp)
       << " camera_params=" << Format(Eigen::Map<const Eigen::VectorXd>(estimator.camera().params.data(),
                                                                           estimator.camera().params.size()));
  return text.str();
}

}  // namespace

OdometryPipeline::OdometryPipeline(const VisualInertialOdometryOptions& options, const std::string& dataset_path,
                                   const std::string& output_path)
    : options_(options), dataset_path_(dataset_path), output_path_(output_path) {}

void OdometryPipeline::AddCallback(CallbackType type, std::function<void()> callback) {
  callbacks_[type].push_back(std::move(callback));
}

void OdometryPipeline::Callback(CallbackType type) const {
  const auto it = callbacks_.find(type);
  if (it == callbacks_.end()) {
    return;
  }
  for (const auto& callback : it->second) {
    callback();
  }
}

bool OdometryPipeline::Run() {
  Dataset dataset;
  if (!ReadDataset(dataset_path_, options_.IsStereo(), &dataset)) {
    return false;
  }
  std::ofstream out;
  if (!output_path_.empty()) {
    out.open(output_path_);
    if (!out) {
      LOG_ERROR("Cannot open the trajectory output {}", output_path_);
      return false;
    }
    out << std::fixed << std::setprecision(9) << std::unitbuf;
  }
  odometry_ = std::make_unique<VisualInertialOdometry>(options_);
  const Estimator& estimator = odometry_->estimator();

  RunTotals totals;
  int initialized_index = -1;
  size_t next_imu = 0;
  stop_ = false;

  // The next image is decoded while this one is processed, as a camera driver delivers images
  auto read_async = [](const std::vector<fs::path>& images, size_t i) {
    return std::async(std::launch::async, ReadGrayPng, images[i].string());
  };
  const bool stereo = !dataset.images1.empty();
  std::future<cv::Mat> next_image = read_async(dataset.images, 0);
  std::future<cv::Mat> next_image1 = stereo ? read_async(dataset.images1, 0) : std::future<cv::Mat>();
  for (size_t i = 0; i < dataset.images.size() && !stop_; ++i) {
    // A live system processes an image once the IMU has passed its capture time
    const double image_time = dataset.image_times[i];
    while (next_imu < dataset.imu.size()) {
      const ImuSample& sample = dataset.imu[next_imu++];
      odometry_->AddImu(sample);
      if (odometry_->ToCameraClock(sample.timestamp) > odometry_->CaptureTime(image_time)) {
        break;
      }
    }
    cv::Mat image = next_image.get();
    cv::Mat image1 = stereo ? next_image1.get() : cv::Mat();
    if (i + 1 < dataset.images.size()) {
      next_image = read_async(dataset.images, i + 1);
      if (stereo) {
        next_image1 = read_async(dataset.images1, i + 1);
      }
    }
    if (image.empty() || (stereo && image1.empty() && !dataset.images1[i].empty())) {
      LOG_ERROR("Cannot read the image {}", (image.empty() ? dataset.images : dataset.images1)[i].string());
      return false;
    }
    if (image.cols != options_.camera.width || image.rows != options_.camera.height) {
      cv::resize(image, image, cv::Size(options_.camera.width, options_.camera.height));
    }
    if (!image1.empty() && (image1.cols != options_.camera1.width || image1.rows != options_.camera1.height)) {
      cv::resize(image1, image1, cv::Size(options_.camera1.width, options_.camera1.height));
    }
    image_ = image;
    if (!odometry_->ProcessImage(image_time, image, static_cast<int>(i), image1)) {
      LOG_ERROR("Odometry stopped at image {}: {}", i, odometry_->failure());
      Callback(kFrameCallback);
      return false;
    }

    const FrameReport& report = odometry_->report();
    if (report.initialized) {
      const InitializerResult& init = report.initialization;
      initialized_index = static_cast<int>(i);
      LOG_INFO("Initialized ({}) at image {}: t={:.9f} v={} q_GtoI={} bg={} ba={}", init.method, i, init.timestamp,
               Format(init.v_IinG, 9), Format(init.q_GtoI, 9), Format(init.gyro_bias, 9), Format(init.accel_bias, 9));
      Callback(kInitializedCallback);
    }
    if (report.has_state) {
      const EstimatorStatistics& stats = estimator.statistics();
      totals.Add(stats);
      const Eigen::Quaterniond q_ItoG(jpl::ToRotation(estimator.orientation()).transpose());
      const Eigen::Vector3d& p = estimator.position();
      if (out.is_open()) {
        out << estimator.timestamp() << ' ' << p.x() << ' ' << p.y() << ' ' << p.z() << ' ' << q_ItoG.x() << ' '
            << q_ItoG.y() << ' ' << q_ItoG.z() << ' ' << q_ItoG.w() << '\n';
      }
      if (estimator.marginalized_keyframe()) {
        Callback(kKeyframeCallback);
      }
      if (i % 100 == 0) {
        LOG_INFO("image={} features={} clones={} candidates={} accepted={} rows={} landmarks={} position={} "
                 "speed={:.3f} {}",
                 i, report.observations.size(), estimator.NumClones(), stats.candidates, stats.accepted,
                 stats.update_rows, stats.landmarks_in_state, Format(p), estimator.velocity().norm(),
                 CalibrationText(options_, estimator));
      }
    }
    Callback(kFrameCallback);
  }

  const EstimatorStatistics& s = totals.sum;
  const LandmarkTracker::Statistics& found = odometry_->landmark_tracker().statistics();
  auto by_loss = [](const int (&counts)[kNumTrackLosses]) {
    std::ostringstream text;
    for (int k = 0; k < kNumTrackLosses; ++k) {
      text << (k > 0 ? "," : "") << counts[k];
    }
    return text.str();
  };
  LOG_INFO("Run summary: initialized_image={} images_with_state={} candidates={} accepted={} too_few={} "
           "triangulation_failed={} (ill_conditioned={} linear_depth={} refined_depth={} baseline={} not_finite={}) "
           "projection_failed={} gate_rejected={} mean_nis_tested={:.3f} mean_nis_accepted={:.3f}",
           initialized_index, totals.images, s.candidates, s.accepted, s.too_few_measurements, s.triangulation_failed,
           s.triangulation_ill_conditioned, s.triangulation_linear_depth, s.triangulation_refined_depth,
           s.triangulation_baseline, s.triangulation_not_finite, s.projection_failed, s.gate_rejected,
           s.nis_sum / std::max(1, s.gate_tested), s.nis_accepted_sum / std::max(1, s.accepted));
  LOG_INFO("Landmarks: initialized={} updated={} gate_rejected={} lost={} max_in_state={} mean_in_state={:.1f} "
           "relinearized={} dormant={} found_again={} removed={} mean_life={:.2f} "
           "mean_updates={:.1f} life_histogram={},{},{},{},{},{}",
           s.landmarks_initialized, s.landmarks_updated, s.landmarks_gate_rejected, s.landmarks_lost,
           s.landmarks_in_state, totals.landmarks_in_state_sum / std::max(1, totals.images),
           s.landmarks_relinearized, s.landmarks_dormant, s.landmarks_found_again, s.landmarks_removed,
           s.landmark_life_sum / std::max(1, s.landmarks_removed),
           s.landmark_updates_sum / std::max(1, s.landmarks_removed), s.landmark_life_histogram[0],
           s.landmark_life_histogram[1], s.landmark_life_histogram[2], s.landmark_life_histogram[3],
           s.landmark_life_histogram[4], s.landmark_life_histogram[5]);
  LOG_INFO("Landmark tracker: searched={} matched={} conflicts={} added={} merged={} added_by_loss={} "
           "accepted_by_loss={} mean_offset={:.2f}",
           found.searched, found.matched, found.conflicts, found.added, found.merged, by_loss(found.added_by_loss),
           by_loss(found.accepted_by_loss), found.offset_sum / std::max(1, found.added));
  if (initialized_index >= 0) {
    LOG_INFO("Calibration: {}", CalibrationText(options_, estimator));
  }
  return initialized_index >= 0;
}

}  // namespace vio
