#include "CamCaptureLib.h"

CamCaptureLib::CamCaptureLib(int32_t _camera_index, 
							 int32_t _snapshot_delay,
							 int32_t frame_rate, 
							 int32_t height_capture, 
							 int32_t width_capture, 
							 bool show_camera_settings)
	: camera_index(0), snapshot_delay(_snapshot_delay), frame(nullptr), window_name("Capture"), degrade_filter(nullptr), btv_sr(nullptr)
{
	// получение списка доступных видеоустройств, возвращаетс§ число устройств
	int32_t numDevices = video_input.listDevices();

	if (_camera_index >= 0 || _camera_index < numDevices)
		camera_index = _camera_index;

	// частота кадров
	video_input.setIdealFramerate(camera_index, frame_rate);

	// указываем разрешение
	video_input.setupDevice(camera_index, height_capture, width_capture, VI_COMPOSITE);

	// показать окошко настроек камеры
	if (show_camera_settings)
		video_input.showSettingsWindow(camera_index); 

	degrade_filter = new NS_DegradeFilter::DegradeFilter();
	btv_sr = new NS_SuperResolution::SuperResolution();
}

void CamCaptureLib::run_capture() {
	// уникальный номер нажатой клавиши
	char ch = 0;
	// им€ сохран€емого скриншота
	char snapshot_name[512];
	// пор€дковый номер скришота, который мы сохран€ем
	int32_t index = 0;
	// таймер сн€ти€ скриншота
	boost::timer snapshot_timer;

	// создаЄм картинку нужного размера
	frame = cvCreateImage(cvSize(video_input.getWidth(camera_index), video_input.getHeight(camera_index)), IPL_DEPTH_8U, 3);
	
	//cvNamedWindow(window_name.c_str(), CV_WINDOW_AUTOSIZE);

	try {

		std::vector<cv::Mat> degrade_images; degrade_images.resize(16);
		std::vector<cv::SparseMat> DHF; DHF.resize(16);
		cv::Mat dest = cv::Mat(cvSize(video_input.getWidth(camera_index), video_input.getHeight(camera_index)), CV_8UC3);

		uint32_t image_count = 16;
		uint32_t number_of_iteration = 60; // 180
		float beta = 0.1f; // 1.3f
		// коэффициент регул€ризации, увеличение ведЄт к сглаживанию сотрых краЄв (прежде чем удал€етс€ шум)
		float lambda = 0.03f;
		// скал€рный вес, примен€етс€ дл€ добавлени€ пространственно затухающего эффекта суммировани€ слагаемых регул€ризации
		float alpha = 0.7f;

		while (true) {
			if (video_input.isFrameNew(camera_index)) {
				// первый параметр - индекс видеоустройсва
				// второй - указатель на буфер дл€ сохранени§ данных
				// третий - флаг, определ§ющий мен€ть ли местами B и R -составл€ющий
				// четвЄртый - флаг, определ€ющий поворачивать картинку или нет
				video_input.getPixels(camera_index, (unsigned char *)frame->imageData, false, true); // получение пикселей в BGR

				/*
				if (snapshot_timer.elapsed() > (snapshot_delay / 1000.0)) {
					snapshot_timer.restart();
					sprintf(snapshot_name, ".//snapshots//Image%d.jpg", index);
					cvSaveImage(snapshot_name, frame);
					index++;
				}
				*/

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

					beta += 0.1f;
					if (beta == 5.0f) {
						stop_capture();
						break;
					}
				}
				else {
					degrade_images[index] = cv::cvarrToMat(frame);
					DHF[index] = degrade_images[index];
					index++;
				}

																					   //
																					   // здесь уже можно работать с картинкой
																					   // с помощью функций OpenCV
																					   //

																					   // показываем картинку
				// cvShowImage(window_name.c_str(), frame);
			}

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
}

void CamCaptureLib::stop_capture() {
	// освобождаем ресурсы
	cvReleaseImage(&frame);
	cvDestroyWindow(window_name.c_str());
	// останавливаем видеозахват
	video_input.stopDevice(camera_index);
}

CamCaptureLib::~CamCaptureLib() {
	if (frame)
		cvReleaseImage(&frame);
	if (degrade_filter)
		delete degrade_filter;
	if (btv_sr)
		delete btv_sr;
}