//
// Created by lucius on 10/10/20.
//

#include "depth_filter.hpp"
#include <boost/log/trivial.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

void depth_filter(const ProjectInfo & project_info, std::vector<std::vector<PlyPoint>> &fused_points)
{
  fused_points.resize(project_info.images.size());

#pragma omp parallel for default(none) shared(project_info, fused_points, options, boost::log::keywords::severity)
  for(auto i = 0; i < project_info.images.size(); i++){
    const auto ref_image = cv::imread(project_info.images[i].string(), cv::IMREAD_UNCHANGED);
    const auto ref_depth = cv::imread(project_info.depths[i].string(), cv::IMREAD_UNCHANGED);
    const auto ref_prob = cv::imread(project_info.probs[i].string(), cv::IMREAD_UNCHANGED);
    const auto image_size = ref_depth.size;
    cv::Mat quality_image(image_size[0], image_size[1], CV_32SC1);
    quality_image.setTo(0);

    BOOST_LOG_TRIVIAL(info) << "process " << project_info.images[i] << " ...";
    for(const auto &it: project_info.pair_info[i]){
      auto src_id = it.first;
      const auto src_depth = cv::imread(project_info.depths[src_id].string(), cv::IMREAD_UNCHANGED);
//      const auto src_prob = cv::imread(project_info.probs[src_id].string(), cv::IMREAD_UNCHANGED);

      const auto &R = project_info.relative_Rs[i][src_id];
      const auto &T = project_info.relative_Ts[i][src_id];
      const auto &R_inv = project_info.relative_Rs[src_id][i];
      const auto &T_inv = project_info.relative_Ts[src_id][i];

      cv::Mat src_image;
      cv::Mat project_image;
      if(options.debug){
        src_image = cv::imread(project_info.images[src_id].string(), cv::IMREAD_UNCHANGED);
        project_image = cv::Mat::zeros(image_size[0], image_size[1], CV_8UC3);
      }

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
            if(src2ref_pos.z() < 0.f){
              continue;
            }
            const auto src2ref_xy = src2ref_pos.hnormalized();
            if((ref_xy - src2ref_xy).norm() < options.disparity_threshold){
              quality_image.at<int32_t>(h, w) += 1;
            }
          }
        }
      }
      auto &points_3d = fused_points[i];
      for(int h = 0; h < image_size[0]; h++) {
        for (int w = 0; w < image_size[1]; w++) {
          if(quality_image.at<int32_t>(h, w) >= options.num_view){
            Eigen::Vector3f pos = ref_depth.at<float>(h, w)* Eigen::Vector3f(w + 0.5, h + 0.5, 1.0);
            pos = project_info.R_invs[i] * pos;
            pos = pos - project_info.Ts[i];
            auto &point = points_3d.emplace_back();
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
      if(options.debug){
        BOOST_LOG_TRIVIAL(info) << project_info.images[i] << " have " << cv::countNonZero(quality_image)*1.0/(image_size[0]*image_size[1]);
        cv::imshow("ref", ref_image);
        cv::imshow("test", project_image);
        while(cv::waitKey(1000) != 'q');
//        break;
      }
    }
  }
}
