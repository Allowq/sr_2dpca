#include "VideoCapture.h"

VideoCapture::VideoCapture(std::string _path_to_video,
						   int32_t _snapshot_delay) 
	: capture_frame(nullptr), frame(nullptr), path_to_video(_path_to_video), snapshot_delay(_snapshot_delay), window_name("Capture"), btv_sr(nullptr)
{
	capture_frame = cvCreateFileCapture(path_to_video.c_str());
	btv_sr = new NS_SuperResolution::SuperResolution();
}

int32_t VideoCapture::run_capture() {
	// уникальный номер нажатой клавиши
	char ch = 0;
	// таймер сн€ти€ скриншота
	boost::timer snapshot_timer;
	snapshot_timer.restart();

	char snapshot_name[80];

	// cvNamedWindow(window_name.c_str(), CV_WINDOW_AUTOSIZE);

	try {

		// пор€дковый номер скришота, который мы сохран€ем
		uint32_t index = 0;
		// требующеес€ нам количество кадров дл€ работы алгоритма
		uint32_t image_count = 10;
		// количество итераций алгоритма sr_btv
		uint32_t number_of_iteration = 60; // 180
		// величина шага в методе наискорейшего спуска
		float beta = 1.3f; // 1.3f
		// коэффициент регул€ризации, увеличение ведЄт к сглаживанию сотрых краЄв (прежде чем удал€етс€ шум)
		float lambda = 0.03f; // 0.03f
		// скал€рный вес, примен€етс€ дл€ добавлени€ пространственно затухающего эффекта суммировани€ слагаемых регул€ризации
		float alpha = 0.7f; // 0.7f

		uint32_t test_step = 0;

		std::vector<cv::Mat> degrade_images; degrade_images.resize(image_count);
		std::vector<cv::SparseMat> DHF; DHF.resize(image_count);
		cv::Mat dest = cv::Mat(cvSize(640, 480), CV_8UC3);
		cv::Mat ideal = cv::Mat(cvSize(640, 480), CV_8UC3);

		while (true) {
			frame = cvQueryFrame(capture_frame);
			if (!frame) {
				stop_capture();
				break;
			}

			if (test_step > 5) {
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
													 NS_SuperResolution::SR_DATA_L1,
													 ideal,
													 test_step);
				test_step++;
				cvSetCaptureProperty(capture_frame, CV_CAP_PROP_POS_AVI_RATIO, 0);
				snapshot_timer.restart();
			}
			else {
				if (snapshot_timer.elapsed() > (snapshot_delay / 1000.0)) {
					snapshot_timer.restart();

					if (index == 0)
						ideal = cv::cvarrToMat(frame);

					degrade_images[index] = cv::cvarrToMat(frame);
					DHF[index] = cv::cvarrToMat(frame);

					sprintf(snapshot_name, ".//snapshots//Image%d.jpg", index);
					cvSaveImage(snapshot_name, frame);
					index++;
				}
			}

			// cvShowImage(window_name.c_str(), frame);
		    ch = cvWaitKey(33);
			// ≈сли была нажата клавиша ESC, то завершаем испонление
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

	return 0;
}

void VideoCapture::stop_capture() {
	// освобождаем ресурсы
	cvReleaseCapture(&capture_frame);
	cvDestroyWindow(window_name.c_str());
}


VideoCapture::~VideoCapture() {
}
