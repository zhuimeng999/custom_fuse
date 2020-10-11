//
// Created by lucius on 9/26/20.
//

#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>
#include "Options.hpp"

namespace po = boost::program_options;

Options options;

void parse_commandline(int argc, char *argv[]) {
  try {
    po::options_description desc{"Options"};
    desc.add_options()("help,h", "Help screen")
        ("inputs_dir", po::value<std::string>(&options.inputs_dir), "inputs_dir")
        ("output_dir", po::value<std::string>(&options.output_dir), "output_dir")
        ("pair_info", po::value<std::string>(&options.pair_info), "pair_info")
        ("debug", po::bool_switch(&options.debug), "debug")
        ("generate_depth", po::bool_switch(&options.generate_depth), "generate_depth")
        ("gpu", po::bool_switch(&options.gpu), "gpu")
        ("num_view", po::value<uint64_t>(&options.num_view), "num_view")
        ("prob_threshold", po::value<float>(&options.prob_threshold), "prob_threshold")
        ("disparity_threshold", po::value<float>(&options.disparity_threshold), "disparity_threshold");

    po::positional_options_description pos_desc;
    pos_desc.add("inputs_dir", 1).add("output_dir", 1);

    po::command_line_parser parser{argc, argv};
//    parser.options(desc).positional(pos_desc).allow_unregistered();
    parser.options(desc).positional(pos_desc);
    po::parsed_options parsed_options = parser.run();

    po::variables_map vm;
    store(parsed_options, vm);
    notify(vm);

    if (vm.count("help")) {
      BOOST_LOG_TRIVIAL(error) << desc << '\n';
      exit(EXIT_FAILURE);
    }

    if (options.inputs_dir.empty() or options.output_dir.empty()) {
      BOOST_LOG_TRIVIAL(error) << "you must provide [inputs_dir output_dir]";
      exit(EXIT_FAILURE);
    }
  }
  catch (const po::error &ex) {
    BOOST_LOG_TRIVIAL(error) << ex.what() << '\n';
    exit(EXIT_FAILURE);
  }
}
