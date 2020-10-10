//
// Created by lucius on 10/10/20.
//

#ifndef CUSTOM_FUSE_DEPTH_FILTER_HPP
#define CUSTOM_FUSE_DEPTH_FILTER_HPP


#include "load_inputs.hpp"
#include "ply.hpp"

void depth_filter(const ProjectInfo & project_info, std::vector<std::vector<PlyPoint>> &fused_points);


#endif //CUSTOM_FUSE_DEPTH_FILTER_HPP
