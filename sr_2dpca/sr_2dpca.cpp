// sr_2dpca.cpp : Defines the entry point for the console application.
//

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <iostream>
#include <stdint.h>
#include <vector>

#include "defines.h"

void parsing_parameters(int argc, char* argv[], std::vector<std::string> *parameters_value) {
	namespace po = boost::program_options;
	std::string appName = boost::filesystem::basename(argv[0]);
	po::options_description desc("Options");

	desc.add_options()
		("help,h", "Print help message")
		("c_c_l,l", "Capture videos with the camera (by library)")
		("c_c_m,m", "Capture videos with the camera (modern)")
		("c_i_v,i", po::value<std::string>(), "Input video from file")
		("c_f_t,t", po::value<int>()->required(), "Capture frames with delay (ms)");

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
			parameters_value->push_back(std::to_string(vm["c_c_l"].as<int>()));
			break;

		case modern_capture:
			parameters_value->push_back(std::to_string(vm["c_c_m"].as<int>()));
			break;

		case video_import:
			parameters_value->push_back(vm["c_i_v"].as<std::string>());
			break;

		default:
			throw;
		}

		parameters_value->push_back(std::to_string(vm["c_f_t"].as<int>()));
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

int main(int argc, char* argv[])
{
	std::vector<std::string> parameters_value;
	parsing_parameters(argc, argv, &parameters_value);
	print_input_parameters(&parameters_value);

    return 0;
}

