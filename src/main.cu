/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   main.cu
 *  @author Thomas Müller, NVIDIA
 */

#include <neural-graphics-primitives/testbed.h>

#include <tiny-cuda-nn/common.h>

#include <args/args.hxx>

#include <filesystem/path.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <time.h>

using namespace args;
using namespace ngp;
using namespace std;
using namespace tcnn;
namespace fs = ::filesystem;

/*! Try to get lock. Return its file descriptor or -1 if failed.
 *
 *  @param lockName Name of file used as lock (i.e. '/var/lock/myLock').
 *  @return File descriptor of lock file, or -1 if failed.
 */
int tryGetLock(char const *lockName) {
    mode_t m = umask(0);
    int fd = open(lockName, O_RDWR|O_CREAT, 0666);
    umask(m);
    if ( fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) < 0) {
        close(fd);
        fd = -1;
    }
    return fd;
}

/*! Release the lock obtained with tryGetLock( lockName ).
 *
 *  @param fd File descriptor of lock returned by tryGetLock( lockName ).
 *  @param lockName Name of file used as lock (i.e. '/var/lock/myLock').
 */
void releaseLock(int fd, char const *lockName) {
    if (fd < 0) return;
    close(fd);
}

float standardDevation(std::vector<float> v) {
  float sum = std::accumulate(v.begin(), v.end(), 0.0);
  float mean = sum / v.size();

  std::vector<float> diff(v.size());
  std::transform(v.begin(), v.end(), diff.begin(),
                std::bind2nd(std::minus<float>(), mean));
  float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  return std::sqrt(sq_sum / v.size());
}

int main(int argc, char** argv) {
	ArgumentParser parser{
		"neural graphics primitives\n"
		"version " NGP_VERSION,
		"",
	};

	HelpFlag help_flag{
		parser,
		"HELP",
		"Display this help menu.",
		{'h', "help"},
	};

	ValueFlag<string> mode_flag{
		parser,
		"MODE",
		"Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified.",
		{'m', "mode"},
	};

	ValueFlag<string> network_config_flag{
		parser,
		"CONFIG",
		"Path to the network config. Uses the scene's default if unspecified.",
		{'n', 'c', "network", "config"},
	};

	Flag no_gui_flag{
		parser,
		"NO_GUI",
		"Disables the GUI and instead reports training progress on the command line.",
		{"no-gui"},
	};

	Flag no_train_flag{
		parser,
		"NO_TRAIN",
		"Disables training on startup.",
		{"no-train"},
	};

	ValueFlag<string> scene_flag{
		parser,
		"SCENE",
		"The scene to load. Can be NeRF dataset, a *.obj mesh for training a SDF, an image, or a *.nvdb volume.",
		{'s', "scene"},
	};

	ValueFlag<string> lock_flag{
		parser,
		"LOCK",
		"Stream lock.",
		{"lock"},
	};

	ValueFlag<string> change_flag{
		parser,
		"CHANGE",
		"Indicate whether the dataset is changed.",
		{"change"},
	};

	ValueFlag<string> snapshot_flag{
		parser,
		"SNAPSHOT",
		"Optional snapshot to load upon startup.",
		{"snapshot"},
	};

	ValueFlag<uint32_t> width_flag{
		parser,
		"WIDTH",
		"Resolution width of the GUI.",
		{"width"},
	};

	ValueFlag<uint32_t> height_flag{
		parser,
		"HEIGHT",
		"Resolution height of the GUI.",
		{"height"},
	};

	Flag version_flag{
		parser,
		"VERSION",
		"Display the version of neural graphics primitives.",
		{'v', "version"},
	};

	// Parse command line arguments and react to parsing
	// errors using exceptions.
	try {
		parser.ParseCLI(argc, argv);
	} catch (const Help&) {
		cout << parser;
		return 0;
	} catch (const ParseError& e) {
		cerr << e.what() << endl;
		cerr << parser;
		return -1;
	} catch (const ValidationError& e) {
		cerr << e.what() << endl;
		cerr << parser;
		return -2;
	}

	if (version_flag) {
		tlog::none() << "neural graphics primitives version " NGP_VERSION;
		return 0;
	}

	try {
		ETestbedMode mode;
		if (!mode_flag) {
			if (!scene_flag) {
				tlog::error() << "Must specify either a mode or a scene";
				return 1;
			}

			fs::path scene_path = get(scene_flag);
			if (!scene_path.exists()) {
				tlog::error() << "Scene path " << scene_path << " does not exist.";
				return 1;
			}

			if (scene_path.is_directory() || equals_case_insensitive(scene_path.extension(), "json")) {
				mode = ETestbedMode::Nerf;
			} else if (equals_case_insensitive(scene_path.extension(), "obj") || equals_case_insensitive(scene_path.extension(), "stl")) {
				mode = ETestbedMode::Sdf;
			} else if (equals_case_insensitive(scene_path.extension(), "nvdb")) {
				mode = ETestbedMode::Volume;
			} else {
				mode = ETestbedMode::Image;
			}
		} else {
			auto mode_str = get(mode_flag);
			if (equals_case_insensitive(mode_str, "nerf")) {
				mode = ETestbedMode::Nerf;
			} else if (equals_case_insensitive(mode_str, "sdf")) {
				mode = ETestbedMode::Sdf;
			} else if (equals_case_insensitive(mode_str, "image")) {
				mode = ETestbedMode::Image;
			} else if (equals_case_insensitive(mode_str, "volume")) {
				mode = ETestbedMode::Volume;
			} else {
				tlog::error() << "Mode must be one of 'nerf', 'sdf', 'image', and 'volume'.";
				return 1;
			}
		}

		Testbed testbed{mode};

		if (scene_flag) {
			fs::path scene_path = get(scene_flag);
			if (!scene_path.exists()) {
				tlog::error() << "Scene path " << scene_path << " does not exist.";
				return 1;
			}
			testbed.load_training_data(scene_path.str());
		}

		std::string mode_str;
		switch (mode) {
			case ETestbedMode::Nerf:   mode_str = "nerf";   break;
			case ETestbedMode::Sdf:    mode_str = "sdf";    break;
			case ETestbedMode::Image:  mode_str = "image";  break;
			case ETestbedMode::Volume: mode_str = "volume"; break;
		}

		if (snapshot_flag) {
			// Load network from a snapshot if one is provided
			fs::path snapshot_path = get(snapshot_flag);
			if (!snapshot_path.exists()) {
				tlog::error() << "Snapshot path " << snapshot_path << " does not exist.";
				return 1;
			}

			testbed.load_snapshot(snapshot_path.str());
			testbed.m_train = false;
		} else {
			// Otherwise, load the network config and prepare for training
			fs::path network_config_path = fs::path{"configs"}/mode_str;
			if (network_config_flag) {
				auto network_config_str = get(network_config_flag);
				if ((network_config_path/network_config_str).exists()) {
					network_config_path = network_config_path/network_config_str;
				} else {
					network_config_path = network_config_str;
				}
			} else {
				network_config_path = network_config_path/"base.json";
			}

			if (!network_config_path.exists()) {
				tlog::error() << "Network config path " << network_config_path << " does not exist.";
				return 1;
			}

			testbed.reload_network_from_file(network_config_path.str());
			testbed.m_train = !no_train_flag;
		}

		bool gui = !no_gui_flag;
#ifndef NGP_GUI
		gui = false;
#endif

		if (gui) {
			testbed.init_window(width_flag ? get(width_flag) : 1920, height_flag ? get(height_flag) : 1080);
		}

		// Render/training loop
    std::cout << "Koke_Cacao: enter training loop" << std::endl;

		while (testbed.frame()) {
			if (!gui) {
				tlog::info() << "iteration=" << testbed.m_training_step << " loss=" << testbed.m_loss_scalar.val();
			}
      std::cout << "\33[2K\rKoke_Cacao: Iteration=" << testbed.m_training_step << " Loss=" << testbed.m_loss_scalar.val() << std::flush;

      // calculate if loss stops decay
      vector<float> subvector_left = {testbed.m_loss_graph.begin(), testbed.m_loss_graph.end() - 1};
      vector<float> subvector_right = {testbed.m_loss_graph.begin() + 1, testbed.m_loss_graph.end()};
      std::transform(subvector_right.begin(), subvector_right.end(), subvector_left.begin(), subvector_right.begin(), std::minus<float>());
      float stdev = standardDevation(subvector_right);
      // tlog::info() << "step: " << testbed.m_training_step << ", sample: " << testbed.m_loss_graph_samples << ", stdev: " << stdev;

      if (lock_flag && scene_flag && testbed.m_loss_graph_samples > testbed.m_loss_graph.size() && stdev < 0.10f) {
        std::cout << "\33[2K\rKoke_Cacao: Iteration=" << testbed.m_training_step << " Loss=" << testbed.m_loss_scalar.val() << " Slow Training!" << std::flush;
        fs::path lock_path = get(lock_flag);
        const std::string& str = lock_path.str();
        const char *cstr = str.c_str();
        int lock_fd = tryGetLock(cstr);
        if (lock_fd > -1) {
          // 1. if this doesn't unlock, blender could not execute (blender can read ngp lock)
          // 2. if I don't load training data here, ngp will wait for blender finish execute (npg can read blender's lock, npg does not access illegal stuff outside of lock)
          // 3. if ngp got lock, and blender change file, it will break
          fs::path scene_path = get(scene_flag);
          const std::string& scene_string = scene_path.str();
          fs::path change_path = get(change_flag);

          if (change_path.exists()) {
            clock_t start = clock();
            change_path.remove_file();
            testbed.load_training_data(scene_string);
            std::cout << std::endl;
            tlog::info() << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms. Will Release Lock";
            std::cout << std::endl;
          }

          releaseLock(lock_fd, cstr);
        } else {
          std::cout << std::endl;
				  tlog::warning() << "Cannot Aquire Lock at " << lock_path.str();
          std::cout << std::endl;
        }
      }
		}
	} catch (const exception& e) {
		tlog::error() << "Uncaught exception: " << e.what();
		return 1;
	}
}
