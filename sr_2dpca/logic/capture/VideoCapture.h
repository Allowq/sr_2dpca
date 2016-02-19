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
	// объект захвата изображения с камеры
	CvCapture *capture_frame;
	// объект захвата кадра из видеопотока
	IplImage *frame;
	// путь к проигрываемому видеофайлу
	std::string path_to_video;
	// таймер захвата очередного кадра из видеопотока
	int32_t snapshot_delay;
	// уникальное имя окна, в котором будет отображаться захват изображения с камеры
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