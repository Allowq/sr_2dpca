#ifndef _PCA2D_H_
#define _PCA2D_H_

#pragma once

#include <opencv2\opencv.hpp>

#include <stdint.h>
#include <string>
#include <fstream>

namespace NS_PCA2D {

	class PCA2D
	{
	private:
		std::vector<cv::Mat> images;
		std::vector<cv::Mat> centre_images;

		uint32_t num_classes;
		uint32_t num_etalons_in_class;
		uint32_t num_images_in_class;

		void read_csv(const std::string &file_name, char separator = ';'); // throw (cv::Exception &);

	public:
		explicit PCA2D(uint32_t _num_classes = 40,
					   uint32_t _num_etalons_in_class = 1,
					   uint32_t _num_images_in_class = 10);

		bool training(const std::string &csv_path);

		virtual ~PCA2D();
	};

}

#endif