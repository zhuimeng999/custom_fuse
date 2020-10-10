//
// Created by lucius on 10/10/20.
//

#ifndef CUSTOM_FUSE_LOAD_INPUTS_HPP
#define CUSTOM_FUSE_LOAD_INPUTS_HPP


#include "Options.hpp"
#include <vector>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>

struct ProjectInfo {
  std::vector<boost::filesystem::path> images;
  std::vector<boost::filesystem::path> depths;
  std::vector<boost::filesystem::path> probs;
  std::vector<Eigen::Matrix4f> extrs;
  std::vector<Eigen::Matrix3f> intrs;
  std::vector<std::vector<Eigen::Matrix3f>> relative_Rs;
  std::vector<std::vector<Eigen::Vector3f>> relative_Ts;
  std::vector<Eigen::Matrix3f> R_invs;
  std::vector<Eigen::Vector3f> Ts;
  std::vector<std::vector<std::pair<int, float>>> pair_info;
};

void load_project(ProjectInfo &project_info);


#endif //CUSTOM_FUSE_LOAD_INPUTS_HPP
