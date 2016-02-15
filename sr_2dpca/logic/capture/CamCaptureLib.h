#ifndef _CAMCAPTURELIB_H_
#define _CAMCAPTURELIB_H_

#pragma once

#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>

#include <boost\thread\thread.hpp>
#include <boost\timer.hpp>

#include <stdint.h>

#include "videoInput.h"
#include "..\super_resolution\SuperResolution.h"
#include "..\filters\DegradeFilter.h"

class CamCaptureLib {
private:
	// ���������� ������ ������ ����������
	int32_t camera_index;
	// ������ ������� ����� �� �����������
	IplImage *frame;
	// ������ ������� ���������� ����� �� �����������
	int32_t snapshot_delay;
	// ������ ����� � ����������� ������� ����������� � ������
	videoInput video_input;
	// ���������� ��� ����, � ������� ����� ������������ ������ ����������� � ������
	std::string window_name;

	NS_DegradeFilter::DegradeFilter *degrade_filter;
	NS_SuperResolution::SuperResolution *btv_sr;

public:
	explicit CamCaptureLib(int32_t _camera_index, 
						   int32_t _snapshot_delay = 500,
						   int32_t frame_rate = 15, 
						   int32_t height_capture = 640, 
						   int32_t width_capture = 480, 
						   bool show_camera_settings = false);

	void run_capture();
	void stop_capture();

	virtual ~CamCaptureLib();
};

#endif