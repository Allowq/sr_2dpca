#include "CamCaptureLib.h"

CamCaptureLib::CamCaptureLib(int32_t _camera_index, 
							 int32_t _snapshot_delay,
							 int32_t frame_rate, 
							 int32_t height_capture, 
							 int32_t width_capture, 
							 bool show_camera_settings)
	: camera_index(0), snapshot_delay(_snapshot_delay), frame(nullptr), sr_frame(nullptr), window_name("Capture"), sr_window_name("SuperResolution"),
	  degrade_filter(nullptr), btv_sr(nullptr)
{
	// ��������� ������ ��������� ��������������, ����������� ����� ���������
	int32_t numDevices = video_input.listDevices();

	if (_camera_index >= 0 || _camera_index < numDevices)
		camera_index = _camera_index;

	// ������� ������
	video_input.setIdealFramerate(camera_index, frame_rate);

	// ��������� ����������
	video_input.setupDevice(camera_index, width_capture, height_capture, VI_COMPOSITE);

	// �������� ������ �������� ������
	if (show_camera_settings)
		video_input.showSettingsWindow(camera_index); 

	sr_frame = cvCreateImage(cvSize(width_capture, height_capture), 8, 3);

	degrade_filter = new NS_DegradeFilter::DegradeFilter();
	btv_sr = new NS_SuperResolution::SuperResolution();
}

void CamCaptureLib::run_capture(int32_t scale) {
	// ���������� ����� ������� �������
	char ch = 0;
	// ��� ������������ ���������
	char snapshot_name[512];
	// ���������� ����� ��������, ������� �� ���������
	int32_t index = 0;
	// ������ ������ ���������
	boost::timer snapshot_timer;

	// ������ �������� ������� �������
	frame = cvCreateImage(cvSize(video_input.getWidth(camera_index), video_input.getHeight(camera_index)), IPL_DEPTH_8U, 3);
	
	//cvNamedWindow(window_name.c_str(), CV_WINDOW_AUTOSIZE);

	try {
		uint32_t image_count = 5;

		std::vector<cv::Mat> degrade_images; degrade_images.resize(image_count);
		std::vector<cv::SparseMat> DHF; DHF.resize(image_count);

		cv::Mat dest = cv::Mat(cvSize(video_input.getWidth(camera_index), video_input.getHeight(camera_index)), CV_8UC3);
		cv::Mat ideal = cv::Mat(cvSize(video_input.getWidth(camera_index), video_input.getHeight(camera_index)), CV_8UC3);
		cv::Mat temp_1 = cv::Mat(cvSize(video_input.getWidth(camera_index) / (scale / 2), video_input.getHeight(camera_index) / (scale / 2)), CV_8UC3);
		cv::Mat temp_2 = cv::Mat(cvSize(video_input.getWidth(camera_index) / scale, video_input.getHeight(camera_index) / scale), CV_8UC3);

		// ���������� ����� ������������
		uint32_t test_step = 0;
		// ���������� �������� ��������� SR ��� ������������
		uint32_t number_of_iteration = 15; // 5
		// ��������������� �������� ������ ������������� ������
		float beta = 1.3f; // 0.6 = 24.1dB default = 1.3f
		// ������� ����������� ������� ������ � �����������
		// ����������� �������������, ���������� ���� � ����������� �c���� ����
		float lambda = 0.03f; // default = 0.03f
		// �������� ����������������� ������������� � btv
		// ��������� ���, ����������� ��� ���������� ��������������� ����������� ������� ������������ ��������� �������������
		float alpha = 0.7f; // default = 0.7f

		while (true) {
			if (video_input.isFrameNew(camera_index)) {
				// ������ �������� - ������ ��������������
				// ������ - ��������� �� ����� ��� ��������� ������
				// ������ - ����, ����������� ������ �� ������� B � R -������������
				// �������� - ����, ������������ ������������ �������� ��� ���
				video_input.getPixels(camera_index, (unsigned char *)frame->imageData, false, true); // ��������� �������� � BGR

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

					sr_frame = cvCloneImage(&(IplImage)dest);
					cvShowImage("SuperResolution", sr_frame);

					test_step++;
				}
				else {
					if (snapshot_timer.elapsed() > (snapshot_delay / 1000.0)) {
						snapshot_timer.restart();

						if (index == 0) 
							ideal = cv::cvarrToMat(frame);

						cv::pyrDown(cv::cvarrToMat(frame), temp_1, cv::Size(temp_1.cols, temp_1.rows));
						cv::pyrDown(temp_1, temp_2, cv::Size(temp_2.cols, temp_2.rows));

						// ���������� ��������, ��� ������� ����� ������������ �����-����������
						cvShowImage(window_name.c_str(), cvCloneImage(&(IplImage)temp_2));

						degrade_images[index] = temp_2;
						DHF[index] = degrade_images[index];

						//sprintf(snapshot_name, ".//snapshots//Image%d.jpg", index);
						//cvSaveImage(snapshot_name, frame);
						index++;
					}
				}

															//
															// ����� ��� ����� �������� � ���������
															// � ������� ������� OpenCV
				
			}

			ch = cvWaitKey(33);
			// ���� ���� ������ ������� ESC, �� ��������� ����������
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
	// ����������� �������
	cvReleaseImage(&frame);
	cvReleaseImage(&sr_frame);
	cvDestroyWindow(window_name.c_str());
	cvDestroyWindow(sr_window_name.c_str());
	// ������������� �����������
	video_input.stopDevice(camera_index);
}

CamCaptureLib::~CamCaptureLib() {
	if (frame)
		cvReleaseImage(&frame);
	if (sr_frame)
		cvReleaseImage(&sr_frame);
	if (degrade_filter)
		delete degrade_filter;
	if (btv_sr)
		delete btv_sr;
}