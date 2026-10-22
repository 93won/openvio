// SPDX-License-Identifier: MIT
/**
 * @file      vio.cc
 * @brief     Command-line tool: vio odometry | vio gui.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-05
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/pipeline/config_loader.h"
#include "vio/pipeline/odometry_pipeline.h"
#include "vio/util/logging.h"

#ifdef VIO_WITH_GUI
#include "vio/ui/viewer.h"
#endif

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

void PrintUsage() {
  std::cout << "Usage:\n"
               "  vio odometry --dataset_path DIR --config_path FILE [--output_path FILE] [--section.key=value ...]\n"
               "  vio gui --dataset_path DIR --config_path FILE [--output_path FILE] [--shots_path DIR] "
               "[--exit_at_end] [--section.key=value ...]\n"
               "\n"
               "  DIR holds images/ (or panocam/), cam_timestamps.txt, imu_data.csv (and gt.txt for the gui).\n"
               "  FILE (output) receives the trajectory in TUM format. --section.key=value overrides one option of\n"
               "  the config file, e.g. --estimator.max_landmarks=150.\n";
}

// Command-line arguments: the tool's own flags and the option overrides (keys with a dot)
struct Arguments {
  std::map<std::string, std::string> flags;
  std::vector<std::string> overrides;
};

bool ParseArguments(int argc, char** argv, Arguments* arguments) {
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) {
      std::cerr << "Unexpected argument " << arg << "\n";
      return false;
    }
    std::string key = arg.substr(2);
    std::string value;
    const size_t equal = key.find('=');
    if (equal != std::string::npos) {
      value = key.substr(equal + 1);
      key = key.substr(0, equal);
    } else if (key == "exit_at_end") {
      value = "1";
    } else if (i + 1 < argc) {
      value = argv[++i];
    } else {
      std::cerr << "Missing value of --" << key << "\n";
      return false;
    }
    if (key.find('.') != std::string::npos) {
      arguments->overrides.push_back(key + "=" + value);
    } else {
      arguments->flags[key] = value;
    }
  }
  return true;
}

std::string Flag(const Arguments& arguments, const std::string& key) {
  const auto it = arguments.flags.find(key);
  return (it == arguments.flags.end()) ? std::string() : it->second;
}

bool ReadOptions(const Arguments& arguments, vio::ConfigLoader* options) {
  const std::string config_path = Flag(arguments, "config_path");
  if (config_path.empty() || Flag(arguments, "dataset_path").empty()) {
    PrintUsage();
    return false;
  }
  return options->Read(config_path, arguments.overrides) && options->Check();
}

int RunOdometry(const Arguments& arguments) {
  vio::ConfigLoader options;
  if (!ReadOptions(arguments, &options)) {
    return 2;
  }
  vio::OdometryPipeline pipeline(options.odometry, Flag(arguments, "dataset_path"), Flag(arguments, "output_path"));
  return pipeline.Run() ? 0 : 1;
}

#ifdef VIO_WITH_GUI
// Ground-truth positions of a TUM trajectory
void ReadGroundTruth(const std::filesystem::path& path, std::vector<double>* times,
                     std::vector<Eigen::Vector3d>* positions) {
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    double t, x, y, z;
    if (row >> t >> x >> y >> z) {
      times->push_back(t);
      positions->emplace_back(x, y, z);
    }
  }
}

int RunGui(const Arguments& arguments) {
  vio::ConfigLoader options;
  if (!ReadOptions(arguments, &options)) {
    return 2;
  }
  const std::string dataset_path = Flag(arguments, "dataset_path");
  const std::string shots_path = Flag(arguments, "shots_path");
  const bool exit_at_end = !Flag(arguments, "exit_at_end").empty();

  vio::Viewer viewer(1920, 1080);
  if (!viewer.Initialize()) {
    LOG_ERROR("Cannot create the viewer window");
    return 1;
  }
  std::vector<double> gt_times;
  std::vector<Eigen::Vector3d> gt_positions;
  ReadGroundTruth(std::filesystem::path(dataset_path) / "gt.txt", &gt_times, &gt_positions);
  if (!gt_times.empty()) {
    viewer.SetGroundTruth(gt_times, gt_positions);
  }
  if (!shots_path.empty()) {
    std::filesystem::create_directories(shots_path);
  }
  auto screenshot = [&](const std::string& name) {
    viewer.RequestScreenshot((std::filesystem::path(shots_path) / name).string());
  };

  vio::OdometryPipeline pipeline(options.odometry, dataset_path, Flag(arguments, "output_path"));
  auto wall_start = std::chrono::steady_clock::now();
  double data_start = -1.0;
  pipeline.AddCallback(vio::OdometryPipeline::kFrameCallback, [&] {
    const vio::FrameReport& report = pipeline.odometry().report();
    viewer.Push(vio::ViewerFrame::FromOdometry(pipeline.odometry(), pipeline.image()));
    if (!shots_path.empty() && report.index % 300 == 0) {
      screenshot("viewer_" + std::to_string(report.index) + ".png");
    }
    // Render; hold while paused (one image per step), and pace to the data time in real-time mode
    if (data_start < 0.0) {
      data_start = report.timestamp;
    }
    while (true) {
      if (!viewer.RenderFrame()) {
        pipeline.Stop();
        return;
      }
      const auto now = std::chrono::steady_clock::now();
      if (viewer.IsPaused() && !viewer.ConsumeStepRequest()) {
        wall_start = now;
        data_start = report.timestamp;
        continue;
      }
      if (viewer.IsRealtime() &&
          report.timestamp - data_start > std::chrono::duration<double>(now - wall_start).count()) {
        continue;
      }
      if (!viewer.IsRealtime()) {
        wall_start = now;
        data_start = report.timestamp;
      }
      return;
    }
  });
  const bool ok = pipeline.Run();

  // Render once more so a screenshot requested with the last image is written, then save every plot tab
  if (!shots_path.empty()) {
    screenshot("viewer_final.png");
  }
  viewer.RenderFrame();
  if (!shots_path.empty()) {
    const char* tabs[4] = {"motion", "updates", "uncertainty", "biases"};
    for (int tab = 0; tab < 4; ++tab) {
      // The plots fit their axes in the frame after the tab first appears
      viewer.SelectPlotTab(tab);
      viewer.RenderFrame();
      viewer.RenderFrame();
      screenshot("final_" + std::string(tabs[tab]) + ".png");
      viewer.RenderFrame();
    }
  }
  while (!exit_at_end && viewer.RenderFrame()) {
  }
  return ok ? 0 : 1;
}
#endif

}  // namespace

int main(int argc, char** argv) {
  vio::InitLogging();
  if (argc < 2) {
    PrintUsage();
    return 2;
  }
  const std::string command = argv[1];
  if (command == "help" || command == "-h" || command == "--help") {
    PrintUsage();
    return 0;
  }
  Arguments arguments;
  if (!ParseArguments(argc, argv, &arguments)) {
    PrintUsage();
    return 2;
  }
  if (command == "odometry") {
    return RunOdometry(arguments);
  }
  if (command == "gui") {
#ifdef VIO_WITH_GUI
    return RunGui(arguments);
#else
    LOG_ERROR("This build has no viewer (GLFW was not found)");
    return 2;
#endif
  }
  LOG_ERROR("Unknown command '{}'", command);
  PrintUsage();
  return 2;
}
