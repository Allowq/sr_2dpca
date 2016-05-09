#include "CascadeClassifier.h"

namespace NS_CascadeClassifier {

	CascadeClassifier::CascadeClassifier(int32_t _camera_index,
										 int32_t frame_rate,
										 int32_t height_capture,
										 int32_t width_capture,
										 bool show_camera_settings)
		: csv_path(""), haar_path(""), window_name("")
	{
		cv::setNumThreads(cv::getNumberOfCPUs());

		// получение списка доступных видеоустройств, возвращаетс¤ число устройств
		int32_t numDevices = video_input.listDevices();

		if (_camera_index >= 0 || _camera_index < numDevices)
			device_id = _camera_index;

		// частота кадров
		video_input.setIdealFramerate(device_id, frame_rate);

		// указываем разрешение
		video_input.setupDevice(device_id, height_capture, width_capture, VI_COMPOSITE);

		// показать окошко настроек камеры
		if (show_camera_settings)
			video_input.showSettingsWindow(device_id);
	}

	void CascadeClassifier::read_csv(char separator)
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

	int32_t CascadeClassifier::run_recognize() {
		try {
			read_csv();
		}
		catch (cv::Exception& e) {
			std::cerr << "Error opening file \"" << csv_path << "\". Reason: " << e.msg << std::endl;
			// nothing more we can do
			exit(1);
		}
		// Get the height from the first image. We'll need this
		// later in code to reshape the images to their original
		// size AND we need to reshape incoming faces to this size:
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

		// Holds the current frame from the Video device:
		cv::Mat frame_cap;
		// Get a handle to the Video device:
		cv::VideoCapture cap(device_id);
		if (!cap.isOpened()) {
			std::cerr << "Capture Device ID " << device_id << "cannot be opened." << std::endl;
			return -6;
		}

		while (true) {
			cap >> frame_cap;
			// Clone the current frame:
			cv::Mat original = frame_cap.clone();
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
			imshow(window_name, original);
			// And display it:
			char key = (char)cv::waitKey(20);
			// Exit this loop on escape:
			if (key == 27)
				break;
		}
	}

	void CascadeClassifier::run_recognize_lib() {
		// Read in the data (fails if no valid input filename is given, but you'll get an error message):
		try {
			read_csv();
		}
		catch (cv::Exception& e) {
			std::cerr << "Error opening file \"" << csv_path << "\". Reason: " << e.msg << std::endl;
			// nothing more we can do
			exit(1);
		}
		// Get the height from the first image. We'll need this
		// later in code to reshape the images to their original
		// size AND we need to reshape incoming faces to this size:
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

		// Holds the current frame from the Video device:
		cv::Mat frame_cap;

		char ch = 0;

		frame = cvCreateImage(cvSize(video_input.getWidth(device_id), video_input.getHeight(device_id)), IPL_DEPTH_8U, 3);

		while (true) {
			if (video_input.isFrameNew(device_id)) {
				video_input.getPixels(device_id, (unsigned char *)frame->imageData, false, true);

				frame_cap = cv::cvarrToMat(frame);
				// Clone the current frame:
				cv::Mat original = frame_cap.clone();
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

					cvShowImage("Detect", &(IplImage)face_resized);

					// Now perform the prediction, see how easy that is:
					int prediction = model->predict(face_resized);

					cvShowImage("Predict", &(IplImage)images[prediction * 10]);

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
				imshow(window_name, original);
			}
			ch = cvWaitKey(33);
			// Если была нажата клавиша ESC, то завершаем испонление
			if (ch == 27) {
				stop_capture();
				break;
			}
		}
	}

	void CascadeClassifier::set_initial_params(const std::string &csv, const std::string &haar) {
		csv_path = csv;
		if (haar != "")
			haar_path = haar;
		else {
			std::string folder = csv.substr(0, csv.find_last_of("/\\"));
			haar_path = folder.substr(0, folder.find_last_of("/\\")).append("\\haarcascades\\haarcascade_frontalface_default.xml");
		}
		window_name = "face_recognizer";
	}

	void CascadeClassifier::stop_capture() {
		// освобождаем ресурсы
		cvReleaseImage(&frame);
		cvDestroyWindow(window_name.c_str());
		// останавливаем видеозахват
		video_input.stopDevice(device_id);
	}

	CascadeClassifier::~CascadeClassifier() {
	}

}