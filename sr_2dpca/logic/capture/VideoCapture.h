#ifndef _VIDEOCAPTURE_H_
#define _VIDEOCAPTURE_H_

#pragma once

#include <opencv\cv.h>
#include <opencv\highgui.h>

#include <boost\thread\thread.hpp>

#include <stdint.h>

#include "..\super_resolution\SuperResolution.h"

class VideoCapture
{
private:
	// ������ ������� ����������� � ������
	CvCapture *capture_frame;
	// ������ ������� ����� �� �����������
	IplImage *frame;
	// ���� � �������������� ����������
	std::string path_to_video;
	// ������ ������� ���������� ����� �� �����������
	int32_t snapshot_delay;
	// ���������� ��� ����, � ������� ����� ������������ ������ ����������� � ������
	std::string window_name;

	NS_SuperResolution::SuperResolution *btv_sr;

public:
	explicit VideoCapture(std::string _path_to_video,
						  int32_t _snapshot_delay = 500);

	void run_capture();
	void stop_capture();

	virtual ~VideoCapture();
};

#endif