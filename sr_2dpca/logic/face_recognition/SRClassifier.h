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
	// ���������� ������ ������ ����������
	int32_t device_id;
	// ������ ����� � ����������� ������� ����������� � ������
	videoInput video_input;

	// ������ ������� ����� �� �����������, �� ���������� �����-����������
	IplImage *frame;
	// ������������ ����, ����� ���������� �����-����������
	IplImage *sr_frame;

	// ���� � ����� �������� ������� csv
	std::string csv_path;
	// ���� � ����� ��������������
	std::string haar_path;
	// ���������� ��� ����, � ������� ����� ������������ ������ ����������� � ������
	std::string window_name;
	// ���������� ��� ����, � ������� ����� ������������ ��������� ���������� �����-����������
	std::string sr_window_name;

	// ������ �����-���������� �����������
	NS_SuperResolution::SuperResolution *btv_sr;

	// ������� ��� �������� ����������� ��� � ��������������� �������� �������
	std::vector<cv::Mat> images;
	std::vector<int32_t> labels;

	// ������ ���������� � ������� ����������� ���
	void read_csv(char separator = ';'); // throw (cv::Exception &)
	// ��������� ��������� ������� ����������� � ������
	void stop_capture();

public:
	explicit SRClassifier(int32_t camera_index,
						  int32_t frame_rate = 15,
						  int32_t height_capture = 480,
						  int32_t width_capture = 640);

	void run_capture(int32_t scale);
	// ������� ���� � csv-����� � �������������� �����
	void set_initial_params(const std::string &csv, const std::string &haar = "");

	virtual ~SRClassifier();
};

#endif