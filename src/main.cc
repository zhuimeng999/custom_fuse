#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include "Options.hpp"
#include "load_inputs.hpp"
#include "ply.hpp"
#include "depth_filter.hpp"

int main(int argc, char *argv[]) {
  boost::log::core::get()->set_filter(
      boost::log::trivial::severity >= boost::log::trivial::info
  );

  parse_commandline(argc, argv);
  ProjectInfo projectInfo;
  load_project(projectInfo);

  std::vector<std::vector<PlyPoint>> fused_points;
  depth_filter(projectInfo, fused_points);

  BOOST_LOG_TRIVIAL(info) << "write ply file ...";
  WriteBinaryPlyPoints(options.output_dir + "/custom_fused.ply", fused_points, false, true);
  BOOST_LOG_TRIVIAL(info) << "done";
  return 0;
}
