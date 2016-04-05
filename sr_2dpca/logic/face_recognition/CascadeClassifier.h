#ifndef _CASCADECLASSIFIER_H_
#define _CASCADECLASSIFIER_H_

#pragma once

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\face.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\objdetect.hpp>
#include "..\capture\videoInput.h"

#include <stdint.h>
#include <iostream>
#include <fstream>
#include <string>

namespace NS_CascadeClassifier {

	class CascadeClassifier
	{
		// порядковый индекс камеры компьютера
		int32_t device_id;
		// путь к файлу описания классов csv
		std::string csv_path;
		// путь к файлу классификатора
		std::string haar_path;
		// уникальное имя окна, в котором будет отображаться захват изображения с камеры
		std::string window_name;

		// These vectors hold the images and corresponding labels.
		std::vector<cv::Mat> images;
		std::vector<int32_t> labels;

		// объект захвата кадра из видеопотока
		IplImage *frame;

		videoInput video_input;

		void read_csv(char separator = ';') throw (cv::Exception &);
		void stop_capture();

	public:
		explicit CascadeClassifier(int32_t _camera_index,
								   int32_t frame_rate = 1,
								   int32_t height_capture = 640,
								   int32_t width_capture = 480,
								   bool show_camera_settings = false);
		
		int32_t run_recognize();
		void run_recognize_lib();
		void set_initial_params(const std::string &csv, const std::string &haar = "");

		virtual ~CascadeClassifier();
	};
}

#endif