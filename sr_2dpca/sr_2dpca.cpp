// sr_2dpca.cpp : Defines the entry point for the console application.
//

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread/mutex.hpp>

#include <iostream>
#include <vector>
#include <stdint.h>
#include <signal.h>

#include "defines.h"
#include "logic\capture\CamCaptureLib.h"
#include "logic\capture\CamCaptureModern.h"
#include "logic\capture\VideoCapture.h"

boost::mutex io_mutex;

void parsing_parameters(int argc, char* argv[], std::vector<std::string> *parameters_value) {
	namespace po = boost::program_options;
	std::string appName = boost::filesystem::basename(argv[0]);
	po::options_description desc("Options");

	desc.add_options()
		("help,h", "Print help message")
		("c_c_l,l", po::value<int32_t>(), "Capture videos with the camera (by library)")
		("c_c_m,m", po::value<int32_t>(), "Capture videos with the camera (modern)")
		("c_i_v,i", po::value<std::string>(), "Input video from file")
		("c_f_t,t", po::value<int32_t>()->required(), "Capture frames with delay (ms)");

	po::variables_map vm;

	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);

		if (vm.count("help")) {
			std::cout << "Master research work by N.N. Kupriyanov" << std::endl;
			std::cout << "Use of super resolution method for detecting face images in the video stream" << std::endl << std::endl;
			std::cout << "USAGE: " << appName << " [-h] <-l ARG> <-m ARG> <-i STRING> <-t ARG>" << std::endl;
			std::cout << "-- Option Description --" << std::endl;
			std::cout << "\t-h[--help]" << "\t" << "Print help message" << std::endl;
			std::cout << "\t-l[--c_c_l]" << "\t" << "Capture videos with the camera (by library)" << std::endl;
			std::cout << "\t-m[--c_c_m]" << "\t" << "Capture videos with the camera (modern)" << std::endl;
			std::cout << "\t-i[--c_i_v]" << "\t" << "Input video from file" << std::endl;
			std::cout << "\t-t[--c_f_t]" << "\t" << "Delay (msec.) between capture frames" << std::endl;
			exit(0);
		}

		po::notify(vm);

	}
	catch (boost::program_options::required_option &e) {
		std::cout << "Error: " << e.what() << std::endl << std::endl;
		std::cout << "For print help message use -h[--help] key" << std::endl;
		exit(1);
	}
	catch (boost::program_options::error &e) {
		std::cout << "Error: " << e.what() << std::endl << std::endl;
		exit(2);
	}

	CAPTURE_TYPE_ENUM capture_type = CAPTURE_TYPE_ENUM::not_capture;

	try {
		if (vm.count("c_c_l")) 
			capture_type = CAPTURE_TYPE_ENUM::input_library;
		else if (vm.count("c_c_m"))
			capture_type = CAPTURE_TYPE_ENUM::modern_capture;
		else if (vm.count("c_i_v")) 
			capture_type = CAPTURE_TYPE_ENUM::video_import;

		if (capture_type == CAPTURE_TYPE_ENUM::not_capture)
			throw;
		else {
			parameters_value->clear();
			parameters_value->push_back(std::to_string(capture_type));
		}

		switch (capture_type)
		{
		case input_library:
			parameters_value->push_back(std::to_string(vm["c_c_l"].as<int32_t>()));
			break;

		case modern_capture:
			parameters_value->push_back(std::to_string(vm["c_c_m"].as<int32_t>()));
			break;

		case video_import:
			parameters_value->push_back(vm["c_i_v"].as<std::string>());
			break;

		default:
			throw;
		}

		parameters_value->push_back(std::to_string(vm["c_f_t"].as<int32_t>()));
	}
	catch (...) {
		std::cout << std::endl << "Error input parameters" << std::endl;
		std::cout << "For print help message use -h[--help] key" << std::endl;
		exit(3);
	}
}

void print_input_parameters(std::vector<std::string> *vec_parameters) {
	std::vector<std::string>::iterator it = vec_parameters->begin();
	uint8_t index = 0;
	while (it != vec_parameters->end()) {
		std::cout << "Parameters #" << std::to_string(index) << " = " << *(it) << std::endl;
		index++;
		++it;
	}
	getchar();
}

void signalHandler(int signal) {
	{
		boost::mutex::scoped_lock lock(io_mutex);
		std::cout << "Signal " << signal << " was send" << std::endl;
	}

	exit(signal);
}

int main(int argc, char* argv[])
{
	std::vector<std::string> parameters_value;
	parsing_parameters(argc, argv, &parameters_value);

	signal(SIGINT, signalHandler);
	signal(SIGTERM, signalHandler);

	// print_input_parameters(&parameters_value);

	CAPTURE_TYPE_ENUM capture_type = CAPTURE_TYPE_ENUM(boost::lexical_cast<int32_t>(parameters_value.front()));
	int32_t snapshot_delay = boost::lexical_cast<int32_t>(parameters_value.at(2));

	switch (capture_type)
	{
	case input_library: {
		CamCaptureLib *cam_capture_lib = new CamCaptureLib(boost::lexical_cast<int32_t>(parameters_value.at(1)),
														   snapshot_delay);
		if (cam_capture_lib) {
			cam_capture_lib->run_capture();
			delete cam_capture_lib;
		}
	} break;

	case modern_capture: {
		CamCaptureModern *cam_capture_modern = new CamCaptureModern(boost::lexical_cast<int32_t>(parameters_value.at(1)),
																    snapshot_delay);
		if (cam_capture_modern) {
			cam_capture_modern->run_capture();
			delete cam_capture_modern;
		}
	} break;

	case video_import: {
		VideoCapture *video_capture = new VideoCapture(parameters_value.at(1), snapshot_delay);
		if (video_capture) {
			video_capture->run_capture();
			delete video_capture;
		}
	} break;

	default:
		std::cout << std::endl << "Error input capture type" << std::endl;
		std::cout << "For print help message use -h[--help] key" << std::endl;
		exit(4);
	}

    return 0;
}

