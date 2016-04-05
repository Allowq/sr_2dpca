#include "SRClassifier.h"

SRClassifier::SRClassifier(int32_t camera_index,
						   int32_t frame_rate,
						   int32_t height_capture,
						   int32_t width_capture) 

	: device_id(0), frame(nullptr), sr_frame(nullptr), window_name("Capture"), sr_window_name("FilterApply"), btv_sr(nullptr),
	  csv_path(""), haar_path("")
{
	// получение списка доступных видеоустройств, возвращаетс¤ число устройств
	int32_t numDevices = video_input.listDevices();

	if (camera_index >= 0 || camera_index < numDevices)
		device_id = camera_index;

	// частота кадров
	video_input.setIdealFramerate(device_id, frame_rate);

	// указываем разрешение
	video_input.setupDevice(device_id, width_capture, height_capture, VI_COMPOSITE);

	sr_frame = cvCreateImage(cvSize(width_capture, height_capture), 8, 3);

	btv_sr = new NS_SuperResolution::SuperResolution();
}

void SRClassifier::read_csv(char separator)
{
	std::ifstream file(csv_path.c_str(), std::ifstream::in | std::ifstream::binary);
	if (!file) {
		std::string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	std::string line, path, class_label;
	std::string folder = csv_path.substr(0, csv_path.find_last_of("/\\"));

	while (getline(file, line)) {
		std::stringstream liness(line);
		path.clear();
		getline(liness, path, separator);
		getline(liness, class_label);
		path = folder + path;
		if (!path.empty() && !class_label.empty()) {
			images.push_back(cv::imread(path, 0));
			labels.push_back(atoi(class_label.c_str()));
		}
	}
}

void SRClassifier::run_capture(int32_t scale) {
	// уникальный номер нажатой клавиши
	char ch = 0;
	// порядковый номер скришота, который мы сохраняем
	int32_t index = 0;

	// создаём картинку нужного размера
	frame = cvCreateImage(cvSize(video_input.getWidth(device_id), video_input.getHeight(device_id)), IPL_DEPTH_8U, 3);


	/* Начало подготовки для захвата и супер-разрешения */
	uint32_t image_count = 5;

	std::vector<cv::Mat> degrade_images; degrade_images.resize(image_count);
	std::vector<cv::SparseMat> DHF; DHF.resize(image_count);

	// изображение после применения фильтра супер-разрешения
	cv::Mat dest = cv::Mat(cvSize(video_input.getWidth(device_id), video_input.getHeight(device_id)), CV_8UC3);
	// оригинальное изображение
	cv::Mat ideal = cv::Mat(cvSize(video_input.getWidth(device_id), video_input.getHeight(device_id)), CV_8UC3);
	// LR = HR / 2
	cv::Mat temp_1 = cv::Mat(cvSize(video_input.getWidth(device_id) / (scale / 2), video_input.getHeight(device_id) / (scale / 2)), CV_8UC3);
	// LR = HR / 4
	cv::Mat temp_2 = cv::Mat(cvSize(video_input.getWidth(device_id) / scale, video_input.getHeight(device_id) / scale), CV_8UC3);

	// количество итераций алгоритма SR над изображением
	uint32_t number_of_iteration = 5; // 5

	// асимптотическое значение метода наискорейшего спуска
	float beta = 1.3f; // 0.6 = 24.1dB default = 1.3f

	// весовой коэффициент баланса данных и сглаживания
	// коэффициент регуляризации, увеличение ведёт к сглаживанию оcтрых краёв
	float lambda = 0.03f; // default = 0.03f

	// параметр пространственного распределения в btv
	// скалярный вес, применяется для добавления пространственно затухающего эффекта суммирования слагаемых регуляризации
	float alpha = 0.7f; // default = 0.7f
	/* Конец подготовки для захвата и супер-разрешения */



	/* Начало подготовки для распознования и классификатора */
	try {
		read_csv();
	}
	catch (cv::Exception& e) {
		std::cerr << "Error opening file \"" << csv_path << "\". Reason: " << e.msg << std::endl;
		exit(1);
	}

	int im_width = images[0].cols;
	int im_height = images[0].rows;

	// Create a FaceRecognizer and train it on the given images:
	cv::Ptr<cv::face::FaceRecognizer> model = cv::face::createEigenFaceRecognizer();
	model->train(images, labels);
	// That's it for learning the Face Recognition model. You now
	// need to create the classifier for the task of Face Detection.
	// We are going to use the haar cascade you have specified in the
	// command line arguments:
	//
	cv::CascadeClassifier haar_cascade;
	haar_cascade.load(haar_path);
	/* Конец подготовки для распознования и классификатора */

	try {
		while (true) {
			if (video_input.isFrameNew(device_id)) {
				video_input.getPixels(device_id, (unsigned char *)frame->imageData, false, true); 

				if ((index > 0) && ((index % image_count) == 0))
				{
					index = 0;

					btv_sr->run_filter(degrade_images,
									   dest,
									   DHF,
									   image_count,
									   number_of_iteration,
									   beta,
									   lambda,
									   alpha,
									   cv::Size(7, 7),
									   NS_SuperResolution::SR_DATA_L1);

					// Clone the current frame:
					cv::Mat original = dest.clone();
					// Convert the current frame to grayscale:
					cv::Mat gray;
					cvtColor(original, gray, CV_BGR2GRAY);
					// Find the faces in the frame:
					std::vector< cv::Rect_<int32_t> > faces;
					haar_cascade.detectMultiScale(gray, faces);

					// At this point you have the position of the faces in
					// faces. Now we'll get the faces, make a prediction and
					// annotate it in the video. Cool or what?
					for (int32_t i = 0; i < faces.size(); i++) {
						// Process face by face:
						cv::Rect face_i = faces[i];
						// Crop the face from the image. So simple with OpenCV C++:
						cv::Mat face = gray(face_i);
						// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
						// verify this, by reading through the face recognition tutorial coming with OpenCV.
						// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
						// input data really depends on the algorithm used.
						//
						// I strongly encourage you to play around with the algorithms. See which work best
						// in your scenario, LBPH should always be a contender for robust face recognition.
						//
						// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
						// face you have just found:
						cv::Mat face_resized;
						cv::resize(face, face_resized, cv::Size(im_width, im_height), 1.0, 1.0, cv::INTER_CUBIC);
						// Now perform the prediction, see how easy that is:
						int prediction = model->predict(face_resized);
						// And finally write all we've found out to the original image!
						// First of all draw a green rectangle around the detected face:
						rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
						// Create the text we will annotate the box with:
						std::string box_text = cv::format("Prediction = %d", prediction);
						// Calculate the position for annotated text (make sure we don't
						// put illegal values in there):
						int32_t pos_x = (std::max)(face_i.tl().x - 10, 0);
						int32_t pos_y = (std::max)(face_i.tl().y - 10, 0);
						// And now put it into the image:
						putText(original, box_text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
					}

					// Show the result:
					sr_frame = cvCloneImage(&(IplImage)original);
					cvShowImage("SuperResolution", sr_frame);
				}
				else {
					if (index == 0)
						ideal = cv::cvarrToMat(frame);

					cv::pyrDown(cv::cvarrToMat(frame), temp_1, cv::Size(temp_1.cols, temp_1.rows));
					cv::pyrDown(temp_1, temp_2, cv::Size(temp_2.cols, temp_2.rows));

					// показываем картинку, над которой будет производится супер-разрешение
					cvShowImage(window_name.c_str(), cvCloneImage(&(IplImage)temp_2));

					degrade_images[index] = temp_2;
					DHF[index] = degrade_images[index];

					index++;
				}
			}

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

void SRClassifier::set_initial_params(const std::string &csv, const std::string &haar) {
	csv_path = csv;
	if (haar != "")
		haar_path = haar;
	else {
		std::string folder = csv.substr(0, csv.find_last_of("/\\"));
		haar_path = folder.substr(0, folder.find_last_of("/\\")).append("\\haarcascades\\haarcascade_frontalface_default.xml");
	}
}

void SRClassifier::stop_capture() {
	// освобождаем ресурсы
	cvReleaseImage(&frame);
	cvReleaseImage(&sr_frame);
	cvDestroyWindow(window_name.c_str());
	cvDestroyWindow(sr_window_name.c_str());
	// останавливаем видеозахват
	video_input.stopDevice(device_id);
}

SRClassifier::~SRClassifier() {
	if (frame)
		cvReleaseImage(&frame);
	if (sr_frame)
		cvReleaseImage(&sr_frame);
	if (btv_sr)
		delete btv_sr;
}
