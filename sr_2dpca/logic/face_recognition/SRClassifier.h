#ifndef _SRCLASSIFIER_H_
#define _SRCLASSIFIER_H_

#pragma once

#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\face.hpp>

#include <boost\thread\thread.hpp>

#include <stdint.h>
#include <fstream>

#include "..\super_resolution\SuperResolution.h"
#include "..\capture\videoInput.h"

class SRClassifier
{
private:
	// порядковый индекс камеры компьютера
	int32_t device_id;
	// объект связи с библиотекой захвата изображения с камеры
	videoInput video_input;

	// объект захвата кадра из видеопотока, до применения супер-разрешения
	IplImage *frame;
	// отображаемый кадр, после применения супер-разрешения
	IplImage *sr_frame;

	// путь к файлу описания классов csv
	std::string csv_path;
	// путь к файлу классификатора
	std::string haar_path;
	// уникальное имя окна, в котором будет отображаться захват изображения с камеры
	std::string window_name;
	// уникальное имя окна, в котором будет отображатсья результат применения супер-разрешения
	std::string sr_window_name;

	// фильтр супер-разрешения изображения
	NS_SuperResolution::SuperResolution *btv_sr;

	// вектора для хранения изображений лиц и соответствующих индексов классов
	std::vector<cv::Mat> images;
	std::vector<int32_t> labels;

	// чтение информации о классах изображений лиц
	void read_csv(char separator = ';'); // throw (cv::Exception &)
	// процедура остановки захвата изображения с камеры
	void stop_capture();

public:
	explicit SRClassifier(int32_t camera_index,
						  int32_t frame_rate = 15,
						  int32_t height_capture = 480,
						  int32_t width_capture = 640);

	void run_capture(int32_t scale);
	// указать путь к csv-файлу и классификатору Хоара
	void set_initial_params(const std::string &csv, const std::string &haar = "");

	virtual ~SRClassifier();
};

#endif