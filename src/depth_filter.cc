//
// Created by lucius on 10/10/20.
//

#include "depth_filter.hpp"
#include "depth_filter_kernel.cuh"
#include <boost/log/trivial.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

struct ProblemDesc {
  cv::Mat ref_image;
  cv::Mat ref_depth;
  cv::Mat ref_prob;
  Eigen::Matrix3f R_inv;
  Eigen::Vector3f T_inv;

  std::vector<cv::Mat> src_images;
  std::vector<cv::Mat> src_depths;
  std::vector<Eigen::Matrix3f> src_Rs;
  std::vector<Eigen::Vector3f> src_Ts;
  std::vector<Eigen::Matrix3f> src_R_invs;
  std::vector<Eigen::Vector3f> src_T_invs;
};

void depth_filter_launch_gpu(const ProblemDesc &pd, cv::Mat &quality_image,
                         std::vector<PlyPoint> &points_3d, std::vector<cv::Mat> &project_images)
{
  ProblemDescGpu pdg;
  pdg.height = pd.ref_depth.rows;
  pdg.width = pd.ref_depth.cols;
  pdg.pair_num = pd.src_depths.size();

  pdg.ref_image = pd.ref_image.data;
  pdg.ref_depth = reinterpret_cast<float *>(pd.ref_depth.data);
  pdg.ref_prob = reinterpret_cast<float *>(pd.ref_prob.data);
  pdg.R_inv = (float *)pd.R_inv.data();
  pdg.T_inv = (float *)pd.T_inv.data();

  auto image_area = pdg.height*pdg.width;
  auto depth_image_size = image_area * sizeof(float);
  auto total_src_size = pd.src_depths.size() * depth_image_size;
  pdg.src_depths = reinterpret_cast<float *>(malloc(total_src_size));
  if(pdg.src_depths == nullptr){
    BOOST_LOG_TRIVIAL(fatal) << "can not allocate memory for src depths";
    exit(EXIT_FAILURE);
  }
  for(int i = 0; i < pd.src_depths.size(); i++){
    memcpy(&pdg.src_depths[i * image_area], pd.src_depths[i].data, depth_image_size);
  }

  static_assert(sizeof(Eigen::Matrix3f[5]) == 5*9*4, "Eigen Matrix does not fit requirement");
  static_assert(sizeof(Eigen::Vector3f[5]) == 5*3*4, "Eigen Matrix does not fit requirement");
  pdg.src_Rs = (float *)(pd.src_Rs.data());
  pdg.src_Ts = (float *)(pd.src_Ts.data());
  pdg.src_R_invs = (float *)(pd.src_R_invs.data());
  pdg.src_T_invs = (float *)(pd.src_T_invs.data());

  static_assert(sizeof(std::pair<int, int>) == 8, "std::pair<int, int> expect to 8");
  std::vector<std::pair<int, int>> valid_pixel;
  valid_pixel.reserve(image_area);
  for(auto h = 0; h < pdg.height; h++){
    for(auto w = 0; w < pdg.width; w++){
      assert(pd.ref_prob.at<float>(h, w) == ((float *)pd.ref_prob.data)[h * pdg.width + w]);
      if(pd.ref_prob.at<float>(h, w) >= options.prob_threshold){
        valid_pixel.emplace_back(h, w);
      }
    }
  }
  pdg.valid_pixel = reinterpret_cast<int *>(valid_pixel.data());
  pdg.valid_pixel_num = valid_pixel.size();

  cv::Mat points_3d_data(pdg.height, pdg.width, CV_32FC3);
#pragma omp critical
  depth_filter_kernel_gpu(pdg, reinterpret_cast<int *>(quality_image.data), reinterpret_cast<float *>(points_3d_data.data));

  for(const auto &it:valid_pixel){
    if(quality_image.at<int32_t>(it.first, it.second) >= options.num_view){
      points_3d.emplace_back();
      auto &point = points_3d.back();
      const auto & point_3d = points_3d_data.at<cv::Vec3f>(it.first, it.second);
      point.x = point_3d[0];
      point.y = point_3d[1];
      point.z = point_3d[2];
      const auto &rgb = pd.ref_image.at<cv::Vec3b>(it.first, it.second);
      point.b = rgb(0);
      point.g = rgb(1);
      point.r = rgb(2);
    }
  }
  free(pdg.src_depths);

}

void depth_filter_launch(const ProblemDesc &pd, cv::Mat &quality_image,
                         std::vector<PlyPoint> &points_3d, std::vector<cv::Mat> &project_images)
{
  const auto &ref_image = pd.ref_image;
  const auto &ref_depth = pd.ref_depth;
  const auto &ref_prob = pd.ref_prob;
  const auto image_size = ref_depth.size;

  quality_image.setTo(0);

  for(int src_id = 0; src_id < pd.src_depths.size(); src_id++){
    const auto &src_image = pd.src_images[src_id];
    const auto &src_depth = pd.src_depths[src_id];
    auto &project_image = project_images[src_id];

    const auto &R = pd.src_Rs[src_id];
    const auto &T = pd.src_Ts[src_id];
    const auto &R_inv = pd.src_R_invs[src_id];
    const auto &T_inv = pd.src_T_invs[src_id];

    for(int h = 0; h < image_size[0]; h++){
      for(int w = 0; w < image_size[1]; w++){
        if(ref_prob.at<float>(h, w) < options.prob_threshold){
          continue;
        }
        Eigen::Vector2f ref_xy(w + 0.5, h + 0.5);
        Eigen::Vector3f ref_pos = ref_depth.at<float>(h, w) * ref_xy.homogeneous();
        const Eigen::Vector3f src_pos = R * ref_pos + T;
        if(src_pos.z() <= 0.f){
          continue;
        }
        const Eigen::Vector2f src_xy = src_pos.hnormalized();

        auto image_h = src_xy.y() - 0.5f;
        auto image_w = src_xy.x() - 0.5f;
        if (image_h > 0.0f && image_w > 0.0f &&
            image_h < static_cast<float>(image_size[0] - 1) && image_w < static_cast<float>(image_size[1] - 1)){

          const float fh = std::floor(image_h);
          const float fw = std::floor(image_w);
          const float dh = image_h - fh;
          const float dw = image_w - fw;
          const float coef_ff = dh * dw;
          const float coef_fc = dh * (1 - dw);
          const float coef_cc = (1 - dh) * (1 - dw);
          const float coef_cf = (1 - dh) * dw;

          int src_h = static_cast<int>(image_h);
          int src_w = static_cast<int>(image_w);

          if(options.debug){
            const auto &pix_ff = src_image.at<cv::Vec3b>(src_h, src_w);
            const auto &pix_fc = src_image.at<cv::Vec3b>(src_h, src_w + 1);
            const auto &pix_cc = src_image.at<cv::Vec3b>(src_h + 1, src_w + 1);
            const auto &pix_cf = src_image.at<cv::Vec3b>(src_h + 1, src_w);

            auto src_sample = coef_cc * pix_ff + coef_cf * pix_fc + coef_ff * pix_cc + coef_fc * pix_cf;
            project_image.at<cv::Vec3b>(h, w) = src_sample;
          }

          const auto d_ff = src_depth.at<float>(src_h, src_w);
          const auto d_fc = src_depth.at<float>(src_h, src_w + 1);
          const auto d_cc = src_depth.at<float>(src_h + 1, src_w + 1);
          const auto d_cf = src_depth.at<float>(src_h + 1, src_w);
          const float d = coef_cc * d_ff + coef_cf * d_fc + coef_ff * d_cc + coef_fc * d_cf;
          Eigen::Vector3f src2ref_pos = d * src_xy.homogeneous();
          src2ref_pos = R_inv*src2ref_pos+T_inv;
          if(src2ref_pos.z() <= 0.f){
            continue;
          }
          const auto src2ref_xy = src2ref_pos.hnormalized();
          if((ref_xy - src2ref_xy).norm() < options.disparity_threshold){
            quality_image.at<int32_t>(h, w) += 1;
          }
        }
      }
    }
  }
  for(int h = 0; h < image_size[0]; h++) {
    for (int w = 0; w < image_size[1]; w++) {
      if(quality_image.at<int32_t>(h, w) >= options.num_view){
        Eigen::Vector3f pos = ref_depth.at<float>(h, w)* Eigen::Vector3f(w + 0.5, h + 0.5, 1.0);
        pos = pd.R_inv * pos;
        pos = pos - pd.T_inv;
        points_3d.emplace_back();
        auto &point = points_3d.back();
        point.x = pos.x();
        point.y = pos.y();
        point.z = pos.z();
        const auto &rgb = ref_image.at<cv::Vec3b>(h, w);
        point.b = rgb(0);
        point.g = rgb(1);
        point.r = rgb(2);
      }
    }
  }
}

void depth_filter(const ProjectInfo & project_info, std::vector<std::vector<PlyPoint>> &fused_points)
{
  fused_points.resize(project_info.images.size());
  if(!options.gpu){
    dump_gpu_info();
  }

#pragma omp parallel for default(none) shared(project_info, fused_points, options, boost::log::keywords::severity)
  for(auto i = 0; i < project_info.images.size(); i++){
    BOOST_LOG_TRIVIAL(info) << "process " << project_info.images[i] << " ...";
    ProblemDesc pd;
    pd.ref_image = cv::imread(project_info.images[i].string(), cv::IMREAD_UNCHANGED);
    pd.ref_depth = cv::imread(project_info.depths[i].string(), cv::IMREAD_UNCHANGED);
    pd.ref_prob = cv::imread(project_info.probs[i].string(), cv::IMREAD_UNCHANGED);
    pd.R_inv = project_info.R_invs[i];
    pd.T_inv = project_info.Ts[i];

    pd.src_depths.reserve(project_info.pair_info[i].size());
    pd.src_Rs.reserve(project_info.pair_info[i].size());
    pd.src_Ts.reserve(project_info.pair_info[i].size());
    pd.src_R_invs.reserve(project_info.pair_info[i].size());
    pd.src_T_invs.reserve(project_info.pair_info[i].size());

    for(const auto &it: project_info.pair_info[i]){
      auto src_id = it.first;
      if(options.debug){
        pd.src_images.emplace_back(cv::imread(project_info.images[src_id].string(), cv::IMREAD_UNCHANGED));
      }
      pd.src_depths.emplace_back(cv::imread(project_info.depths[src_id].string(), cv::IMREAD_UNCHANGED));

      pd.src_Rs.emplace_back(project_info.relative_Rs[i][src_id]);
      pd.src_Ts.emplace_back(project_info.relative_Ts[i][src_id]);
      pd.src_R_invs.emplace_back(project_info.relative_Rs[src_id][i]);
      pd.src_T_invs.emplace_back(project_info.relative_Ts[src_id][i]);
    }

    const auto image_size = pd.ref_depth.size;
    cv::Mat quality_image(image_size[0], image_size[1], CV_32SC1);
    std::vector<cv::Mat> project_image;
    if(options.gpu){
      depth_filter_launch_gpu(pd, quality_image, fused_points[i], project_image);
    } else {
      if(options.debug){
        project_image.resize(project_info.pair_info[i].size());
      }
      depth_filter_launch(pd, quality_image, fused_points[i], project_image);
    }

    if(options.debug){
      BOOST_LOG_TRIVIAL(info) << project_info.images[i] << " have " << cv::countNonZero(quality_image)*1.0/(image_size[0]*image_size[1]);
      cv::imshow("ref", pd.ref_image);
      cv::imshow("test", project_image[0]);
      while(cv::waitKey(1000) != 'q');
//        break;
    }
  }
}

