//
// Created by lucius on 10/10/20.
//

#include "load_inputs.hpp"
#include "Options.hpp"
#include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>
#include <fstream>

namespace fs = boost::filesystem;

void ReadPair(ProjectInfo &project_info, const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    BOOST_LOG_TRIVIAL(error) << "can not open output file";
    exit(EXIT_FAILURE);
  }
  uint32_t total_view_num = 0;
  in >> total_view_num;

  project_info.pair_info.resize(total_view_num);
  uint32_t ref_id;
  uint32_t src_num;
  uint32_t src_id;
  float src_score;
  for (auto i = 0; i < total_view_num; i++) {
    in >> ref_id >> src_num;
    if (ref_id != i) {
      BOOST_LOG_TRIVIAL(error) << "unmatch index in inpute file expect" << i << " got " << ref_id;
      exit(EXIT_FAILURE);
    }
    project_info.pair_info[ref_id].reserve(src_num);
    for (auto j = 0; j < src_num; j++) {
      in >> src_id >> src_score;
      project_info.pair_info[ref_id].emplace_back(src_id, src_score);
    }
  }
  std::string dummy;
  in >> dummy;
  if ((dummy != "") or !in.eof()) {
    BOOST_LOG_TRIVIAL(error) << "file size is too big";
    exit(EXIT_FAILURE);
  }
  in.close();
}

void load_project(ProjectInfo &project_info)
{
  fs::path proj_dir(options.inputs_dir);
  try
  {
    if (fs::exists(proj_dir))    // does p actually exist?
    {
      if (fs::is_directory(proj_dir))      // is p a directory?
      {
        for(const auto &it: fs::directory_iterator(proj_dir)){
          const auto &p = it.path();
          if(p.extension() == ".jpg"){
            project_info.images.push_back(p);
          }
        }
      }
      else{
        BOOST_LOG_TRIVIAL(fatal) << proj_dir << " exists, but is not a directory";
        exit(EXIT_FAILURE);
      }
    }
    else{
      BOOST_LOG_TRIVIAL(fatal) << proj_dir << " does not exist";
      exit(EXIT_FAILURE);
    }

    std::sort(project_info.images.begin(), project_info.images.end());
    for(const auto &it:project_info.images){
      const auto depth = it.parent_path()/(it.stem().string() + "_init.pfm");
      const auto prob = it.parent_path()/(it.stem().string() + "_prob.pfm");
      const auto cam = it.parent_path()/(it.stem().string() + ".txt");
      if(fs::is_regular_file(it) and fs::is_regular_file(prob) and fs::is_regular_file(cam)){
        project_info.depths.push_back(depth);
        project_info.probs.push_back(depth);
        {
          Eigen::Matrix4f extr;
          std::ifstream in(cam);
          assert(in.is_open());
          std::string signature;
          in >> signature;
          for(int j = 0; j < 4; j++){
            for(int k = 0; k < 4; k++){
              in >> extr(j, k);
            }
          }
          project_info.extrs.push_back(extr);
          in >> signature;
          Eigen::Matrix3f intr;
          for(int j = 0; j < 3; j++){
            for(int k = 0; k < 3; k++){
              in >>intr(j, k);
            }
          }
          project_info.intrs.push_back(intr);
          float dummy;
          in >> dummy >> dummy >> dummy >> dummy >> dummy;
          assert(in.eof());
          in.close();
        }
      } else {
        BOOST_LOG_TRIVIAL(fatal) << "can not find corrospond depth or prob file for image " << it;
        exit(EXIT_FAILURE);
      }
    }
    BOOST_LOG_TRIVIAL(info) << "project have " << project_info.images.size() << " views";
    BOOST_LOG_TRIVIAL(info) << "compute relative matrix ...";
    project_info.relative_Rs.resize(project_info.images.size());
    project_info.relative_Ts.resize(project_info.images.size());
    project_info.R_invs.resize(project_info.images.size());
    project_info.Ts.resize(project_info.images.size());
    for(auto i = 0; i < project_info.images.size(); i++) {
      project_info.relative_Rs[i].resize(project_info.images.size());
      project_info.relative_Ts[i].resize(project_info.images.size());
    }
    for(auto i = 0; i < project_info.images.size(); i++) {
      const auto ref_R = project_info.extrs[i].block<3, 3>(0, 0);
      const auto ref_T = project_info.extrs[i].block<3, 1>(0, 3);
      const auto ref_K = project_info.intrs[i];
      const auto ref_R_transpose = ref_R.transpose();
      const auto ref_K_inv = ref_K.inverse();

      project_info.R_invs[i] = ref_R_transpose*ref_K_inv;
      project_info.Ts[i] = ref_R_transpose*ref_T;
      for(auto j = 0; j < project_info.images.size(); j++){
        if(i == j){
          continue;
        }
        const auto src_R = project_info.extrs[j].block<3, 3>(0, 0);
        const auto src_T = project_info.extrs[j].block<3, 1>(0, 3);
        const auto src_K = project_info.intrs[j];

        const auto delta_R = src_R * ref_R_transpose;
        const auto delta_T = src_T - (delta_R * ref_T);
        const auto R = src_K * delta_R * ref_K_inv;
        const auto T = src_K * delta_T;
        project_info.relative_Rs[i][j] = R;
        project_info.relative_Ts[i][j] = T;
      }
    }

    if(options.pair_info.empty()){
      project_info.pair_info.resize(project_info.images.size());
      for(auto i = 0; i < project_info.images.size(); i++) {
        project_info.pair_info[i].reserve(project_info.images.size());
        for(int j = 0; j < project_info.images.size(); j++){
          project_info.pair_info[i].emplace_back(j, 0);
        }
      }
    } else {
      ReadPair(project_info, options.pair_info);
    }
    BOOST_LOG_TRIVIAL(info) << "compute relative matrix done";
  }
  catch (const fs::filesystem_error& ex)
  {
    BOOST_LOG_TRIVIAL(fatal) << ex.what();
    exit(EXIT_FAILURE);
  }
}
