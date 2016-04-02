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
		("c_f_t,t", po::value<int32_t>(), "Capture frames with delay (ms)")
		("t_e_f_r,e", po::value<std::string>(), "Test eigen face recognizer")
		("t_c_c_r,c", po::value<std::string>(), "Test cascade classifier face recognizer");

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
			std::cout << "\t-e[--t_e_f_r]" << "\t" << "Test eigen face recognizer" << std::endl;
			std::cout << "\t-c[--t_c_c_r]" << "\t" << "Test cascade classifier face recognizer" << std::endl;
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

	APPLICATION_OPTIONS_ENUM options_value = APPLICATION_OPTIONS_ENUM::not_choisen;

	try {
		if (vm.count("c_c_l")) 
			options_value = APPLICATION_OPTIONS_ENUM::input_library;
		else if (vm.count("c_c_m"))
			options_value = APPLICATION_OPTIONS_ENUM::modern_capture;
		else if (vm.count("c_i_v")) 
			options_value = APPLICATION_OPTIONS_ENUM::video_import;
		else if (vm.count("t_e_f_r"))
			options_value = APPLICATION_OPTIONS_ENUM::eigen_test;
		else if (vm.count("t_c_c_r"))
			options_value = APPLICATION_OPTIONS_ENUM::haar_cascade_test;
		
		if (options_value == APPLICATION_OPTIONS_ENUM::not_choisen)
			throw;
		else {

			parameters_value->clear();

			parameters_value->push_back(std::to_string(options_value));
			switch (options_value)
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

				case eigen_test:
					parameters_value->push_back(vm["t_e_f_r"].as<std::string>());
					break;

				case haar_cascade_test:
					parameters_value->push_back(vm["t_c_c_r"].as<std::string>());
					break;

				default:
					throw;
			}

			switch (options_value) {
			case input_library:
			case modern_capture:
			case video_import:
				if (vm.count("c_f_t"))
					parameters_value->push_back(std::to_string(vm["c_f_t"].as<int32_t>()));
			}
		}
		
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

	APPLICATION_OPTIONS_ENUM options_value = APPLICATION_OPTIONS_ENUM(boost::lexical_cast<int32_t>(parameters_value.front()));
	int32_t snapshot_delay = 0;
	if (parameters_value.size() > 2)
		snapshot_delay = boost::lexical_cast<int32_t>(parameters_value.at(2));

	switch (options_value)
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

	case eigen_test: {
		NS_EigenRecognize::EigenRecognize *eigen_test = new NS_EigenRecognize::EigenRecognize(parameters_value.at(1));
		if (eigen_test)	{
			eigen_test->run_test("output");
			delete eigen_test;
			eigen_test = NULL;
		}
	} break;

	case haar_cascade_test: {
		NS_CascadeClassifier::CascadeClassifier *haar_cascade = new NS_CascadeClassifier::CascadeClassifier(0);
		if (haar_cascade) {
			haar_cascade->set_initial_params(parameters_value.at(1));
			haar_cascade->run_recognize_lib();
			delete haar_cascade;
			haar_cascade = NULL;
		}
	} break;

	default:
		std::cout << std::endl << "Error input capture type" << std::endl;
		std::cout << "For print help message use -h[--help] key" << std::endl;
		exit(4);
	}

    return 0;
}

