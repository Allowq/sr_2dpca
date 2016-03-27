#ifndef _EIGENRECOGNIZE_H_
#define _EIGENRECOGNIZE_H_

#pragma once

#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>

#include <fstream>
#include <stdint.h>

namespace NS_EigenRecognize {

	class EigenRecognize
	{
	private:
		// Get the path to our CSV.
		std::string csv_path;

		// These vectors hold the images and corresponding labels.
		std::vector<cv::Mat> images;
		std::vector<int32_t> labels;

	public:
		explicit EigenRecognize(std::string _csv_path);

		void read_csv(const std::string &file_name,
					  char separator) throw (cv::Exception &);

		virtual ~EigenRecognize();
	};
}

#endif