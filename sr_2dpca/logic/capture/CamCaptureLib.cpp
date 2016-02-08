#include "CamCaptureLib.h"

CamCaptureLib::CamCaptureLib(int32_t _camera_index, 
							 int32_t _snapshot_delay,
							 int32_t frame_rate, 
							 int32_t height_capture, 
							 int32_t width_capture, 
							 bool show_camera_settings)
	: camera_index(0), snapshot_delay(_snapshot_delay), frame(nullptr), window_name("Capture")
{
	// ��������� ������ ��������� ��������������, ����������� ����� ���������
	int32_t numDevices = video_input.listDevices();

	if (_camera_index >= 0 || _camera_index < numDevices)
		camera_index = _camera_index;

	// ������� ������
	video_input.setIdealFramerate(camera_index, frame_rate);

	// ��������� ����������
	video_input.setupDevice(camera_index, height_capture, width_capture, VI_COMPOSITE);

	// �������� ������ �������� ������
	if (show_camera_settings)
		video_input.showSettingsWindow(camera_index); 
}

void CamCaptureLib::run_capture() {
	// ���������� ����� ������� �������
	char ch = 0;
	// ������ �������� ������� �������
	frame = cvCreateImage(cvSize(video_input.getWidth(camera_index), video_input.getHeight(camera_index)), IPL_DEPTH_8U, 3);
	cvNamedWindow(window_name.c_str(), CV_WINDOW_AUTOSIZE);

	try {
		while (true) {
			if (video_input.isFrameNew(camera_index)) {
				// ������ �������� - ������ ��������������
				// ������ - ��������� �� ����� �� ��������� ������
				// ������ - ����, ����������� ����� �� ������� B � R -�����������
				// �������� - ����, ����������� ������������ �������� ��� ���
				video_input.getPixels(camera_index, (unsigned char *)frame->imageData, false, true); // ��������� �������� � BGR

																					   //
																					   // ����� ��� ����� �������� � ���������
																					   // � ������� ������� OpenCV
																					   //

																					   // ���������� ��������
				cvShowImage(window_name.c_str(), frame);
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
	cvDestroyWindow(window_name.c_str());
	// ������������� �����������
	video_input.stopDevice(camera_index);
}

CamCaptureLib::~CamCaptureLib() {

}