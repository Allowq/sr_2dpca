#include "VideoCapture.h"

VideoCapture::VideoCapture(std::string _path_to_video,
						   int32_t _snapshot_delay) 
	: capture_frame(nullptr), frame(nullptr), path_to_video(_path_to_video), snapshot_delay(_snapshot_delay), window_name("Capture"), btv_sr(nullptr)
{
	capture_frame = cvCreateFileCapture(path_to_video.c_str());
	btv_sr = new NS_SuperResolution::SuperResolution();
}

void VideoCapture::run_capture() {
	// уникальный номер нажатой клавиши
	char ch = 0;

	// cvNamedWindow(window_name.c_str(), CV_WINDOW_AUTOSIZE);

	try {

		// пор€дковый номер скришота, который мы сохран€ем
		uint32_t index = 0;
		// требующеес€ нам количество кадров дл€ работы алгоритма
		uint32_t image_count = 16;
		// количество итераций алгоритма sr_btv
		uint32_t number_of_iteration = 60; // 180
		// величина шага в методе наискорейшего спуска
		float beta = 0.1f; // 1.3f
		// коэффициент регул€ризации, увеличение ведЄт к сглаживанию сотрых краЄв (прежде чем удал€етс€ шум)
		float lambda = 0.01f; // 0.03
		// скал€рный вес, примен€етс€ дл€ добавлени€ пространственно затухающего эффекта суммировани€ слагаемых регул€ризации
		float alpha = 0.7f;

		std::vector<cv::Mat> degrade_images; degrade_images.resize(16);
		std::vector<cv::SparseMat> DHF; DHF.resize(16);
		cv::Mat dest = cv::Mat(cvSize(640, 480), CV_8UC3);

		while (true) {
			frame = cvQueryFrame(capture_frame);
			if (!frame) {
				stop_capture();
				break;
			}

			if ((index > 0) && ((index % image_count) == 0))
			{
				index = 0;
				btv_sr->bilateral_total_variation_sr(degrade_images,
													 dest,
													 DHF,
													 image_count,
													 number_of_iteration,
													 beta,
													 lambda,
													 alpha,
													 cv::Size(7, 7),
													 NS_SuperResolution::SR_DATA_L1);

				lambda += 0.05f;
				if (lambda == 1.0f) {
					stop_capture();
					break;
				}
				else {
					cvSetCaptureProperty(capture_frame, CV_CAP_PROP_POS_AVI_RATIO, 0);
				}
			}
			else {
				degrade_images[index] = cv::cvarrToMat(frame);
				DHF[index] = degrade_images[index];
				index++;
			}

			// cvShowImage(window_name.c_str(), frame);
			// ch = cvWaitKey(33);
			// ≈сли была нажата клавиша ESC, то завершаем испонление
			// if (ch == 27) {
			// 	stop_capture();
			//	break;
			// }
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
