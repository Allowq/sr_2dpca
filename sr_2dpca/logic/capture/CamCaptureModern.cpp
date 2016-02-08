#include "CamCaptureModern.h"

CamCaptureModern::CamCaptureModern(int32_t _camera_index,
								   int32_t _snapshot_delay) 
	: camera_index(_camera_index), snapshot_delay(_snapshot_delay), frame(nullptr), capture_frame(nullptr), window_name("Capture")
{
	capture_frame = cvCreateCameraCapture(camera_index);

	/*
	if (capture != NULL) {
		double width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
		double height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
		printf("[i] %.0f x %.0f\n", width, height);
	}
	*/
}

void CamCaptureModern::run_capture() {
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

void CamCaptureModern::stop_capture() {
	// освобождаем ресурсы
	cvReleaseImage(&frame);
	cvReleaseCapture(&capture_frame);
	cvDestroyWindow(window_name.c_str());
}

CamCaptureModern::~CamCaptureModern() {
}
