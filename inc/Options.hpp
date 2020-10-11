//
// Created by lucius on 9/26/20.
//

#ifndef CUSTOM_VIEW_SELECT_OPTIONS_HPP
#define CUSTOM_VIEW_SELECT_OPTIONS_HPP

#include <string>

struct Options {
  std::string inputs_dir;
  std::string output_dir;
  std::string pair_info;

  bool debug = false;

  bool generate_depth = false;
  bool gpu = true;
  uint64_t num_view = 3;
  float prob_threshold = 0.4;
  float disparity_threshold = 0.3;
};

extern Options options;

void parse_commandline(int argc, char *argv[]);

#endif //CUSTOM_VIEW_SELECT_OPTIONS_HPP
