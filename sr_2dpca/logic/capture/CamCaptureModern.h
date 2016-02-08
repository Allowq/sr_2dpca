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
	// порядковый индекс камеры компьютера
	int32_t camera_index;
	// объект захвата изображения с камеры
	CvCapture *capture_frame;
	// объект захвата кадра из видеопотока
	IplImage *frame;
	// таймер захвата очередного кадра из видеопотока
	int32_t snapshot_delay;
	// уникальное имя окна, в котором будет отображаться захват изображения с камеры
	std::string window_name;

public:
	explicit CamCaptureModern(int32_t _camera_index,
							  int32_t _snapshot_delay = 500);

	void run_capture();
	void stop_capture();

	virtual ~CamCaptureModern();
};

#endif