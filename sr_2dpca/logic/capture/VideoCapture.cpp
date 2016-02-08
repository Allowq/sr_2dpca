#include "VideoCapture.h"

VideoCapture::VideoCapture(std::string _path_to_video,
						   int32_t _snapshot_delay) 
	: capture_frame(nullptr), frame(nullptr), path_to_video(_path_to_video), snapshot_delay(_snapshot_delay), window_name("Capture")
{
	capture_frame = cvCreateFileCapture(path_to_video.c_str());
}

void VideoCapture::run_capture() {
	// уникальный номер нажатой клавиши
	char ch = 0;
	cvNamedWindow(window_name.c_str(), CV_WINDOW_AUTOSIZE);

	try {
		while (true) {
			frame = cvQueryFrame(capture_frame);
			if (!frame) {
				stop_capture();
				break;
			}
			cvShowImage(window_name.c_str(), frame);

			ch = cvWaitKey(33);
			// Если была нажата клавиша ESC, то завершаем испонление
			if (ch == 27) {
				stop_capture();
				break;
			}
		}
	}
	catch (boost::thread_interrupted) {
		stop_capture();
		throw boost::thread_interrupted();
	}
}

void VideoCapture::stop_capture() {
	// освобождаем ресурсы
	cvReleaseCapture(&capture_frame);
	cvDestroyWindow(window_name.c_str());
}


VideoCapture::~VideoCapture() {
}
