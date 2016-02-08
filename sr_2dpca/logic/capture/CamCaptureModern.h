#ifndef _CAMCAPTUREMODERN_H_
#define _CAMCAPTUREMODERN_H_

#pragma once

#include <opencv\cv.h>
#include <opencv\highgui.h>

#include <boost\thread\thread.hpp>

#include <stdint.h>

class CamCaptureModern
{
private:
	// ���������� ������ ������ ����������
	int32_t camera_index;
	// ������ ������� ����������� � ������
	CvCapture *capture_frame;
	// ������ ������� ����� �� �����������
	IplImage *frame;
	// ������ ������� ���������� ����� �� �����������
	int32_t snapshot_delay;
	// ���������� ��� ����, � ������� ����� ������������ ������ ����������� � ������
	std::string window_name;

public:
	explicit CamCaptureModern(int32_t _camera_index,
							  int32_t _snapshot_delay = 500);

	void run_capture();
	void stop_capture();

	virtual ~CamCaptureModern();
};

#endif