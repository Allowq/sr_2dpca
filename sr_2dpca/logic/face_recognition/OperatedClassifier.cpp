#include "OperatedClassifier.h"

OperatedClassifier::OperatedClassifier()
	: num_classes(41), num_etalons_in_class (1), num_images_in_class(10)
{
}

void OperatedClassifier::read_csv(const std::string &file_name, char separator)
{
	std::ifstream file(file_name.c_str(), std::ifstream::in | std::ifstream::binary);
	if (!file) {
		std::string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	std::string line, path, class_label;
	std::string folder = file_name.substr(0, file_name.find_last_of("/\\"));

	int32_t class_index = 0;
	while (getline(file, line)) {
		std::stringstream liness(line);
		path.clear();
		getline(liness, path, separator);
		getline(liness, class_label);
		path = folder + path;
		if (!path.empty() && !class_label.empty()) {
			if ((class_index % num_images_in_class) < num_etalons_in_class) {
				etalons[atoi(class_label.c_str())] = cv::imread(path, 0);
			}
			else {
				images.push_back(cv::imread(path, 0));
				labels.push_back(atoi(class_label.c_str()));
			}
			class_index++;
		}
	}
	file.close();

	std::cout << "Number of images = " << images.size() << " (" << labels.size() << ")" << std::endl;
	std::cout << "Number of etalons = " << etalons.size() << std::endl << std::endl;
}

bool OperatedClassifier::run(std::string csv_path) {
	try {
		read_csv(csv_path);
	}
	catch (cv::Exception& e) {
		std::string error_message = "Error opening file \"";
		error_message.append(csv_path).append("\". Reason: ").append(e.msg).append("\n");
		CV_Error(cv::Error::StsError, error_message);
		return false;
	}

	if (images.size() <= 1) {
		std::string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(cv::Error::StsError, error_message);
		return false;
	}

	NS_SuperResolution::SuperResolution *sr = new NS_SuperResolution::SuperResolution();

	std::cout << "-- Training start --" << std::endl;
	cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::createEigenFaceRecognizer(); // cv::face::createEigenFaceRecognizer();
	model->train(images, labels);
	std::cout << "-- Training complete --" << std::endl << std::endl;

	/*
	int32_t results = this->apply_filter(model, FILTER_TYPE_ENUM::none);
	std::string result_message = "None filter results:\n";
	result_message.append(cv::format("Recognized %d from %d images\nRecognition quality: %3.2f", results, num_classes, float(results)/num_classes));
	std::cout << result_message << std::endl << std::endl;
	*/

	int32_t results = this->apply_filter(model, FILTER_TYPE_ENUM::scale);
	std::string result_message = "Scale filter results:\n";
	result_message.append(cv::format("Recognized %d from %d images\nRecognition quality: %3.2f", results, num_classes, float(results) / num_classes));
	std::cout << result_message << std::endl << std::endl;

	results = this->apply_filter_sr(model, sr, FILTER_TYPE_ENUM::scale);
	result_message = "Scale filter super-resolution results:\n";
	result_message.append(cv::format("Recognized %d from %d images\nRecognition quality: %3.2f", results, num_classes, float(results) / num_classes));
	std::cout << result_message << std::endl << std::endl;

	/*
	results = this->apply_filter(model, FILTER_TYPE_ENUM::dhf);
	result_message = "DHF filter results:\n";
	result_message.append(cv::format("Recognized %d from %d images\nRecognition quality: %3.2f", results, num_classes, float(results) / num_classes));
	std::cout << result_message << std::endl << std::endl;
	*/

	cv::waitKey(0);
	getchar();

	if (sr)
	{
		delete sr;
		sr = NULL;
	}

	return true;
}

int32_t OperatedClassifier::apply_filter(cv::Ptr<cv::face::BasicFaceRecognizer> &model, FILTER_TYPE_ENUM value) const {
	cv::Mat test_sample = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_8UC1);
	int32_t predict_label = -1, // предполагаемый результат распознавания 
			magic = 0;

	std::cout << "Recognition scale test start" << std::endl;

	std::map<int32_t, cv::Mat>::const_iterator it_etalon = etalons.begin();
	while (it_etalon != etalons.end())
	{
		it_etalon->second.copyTo(test_sample);
		//imshow("Normal", test_sample);

		switch (value)
		{
		case OperatedClassifier::none:
			break;
		case OperatedClassifier::scale:
			NS_DegradeFilter::DegradeFilter::down_up_scale_image(4, test_sample);
			break;
		case OperatedClassifier::dhf:
			NS_DegradeFilter::DegradeFilter::dhf_image(1, &test_sample);
			break;
		default:
			return 0;
		}

		//imshow("Modern", test_sample);
		predict_label = model->predict(test_sample);
		//imshow("Predict", images[predict_label * (num_images_in_class - num_etalons_in_class)]);

		if (predict_label == it_etalon->first)
			magic++;

		++it_etalon;
		//cv::waitKey(0);
	}

	return magic;
}

int32_t OperatedClassifier::apply_filter_sr(cv::Ptr<cv::face::BasicFaceRecognizer> &model, 
										    NS_SuperResolution::SuperResolution *sr, 
											FILTER_TYPE_ENUM value) const 
{
	cv::Mat dest = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_8UC1), 
			lr_gray = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_8UC1),
			lr_copy = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_8UC3);
	int32_t predict_label = -1, // предполагаемый результат распознавания 
			magic = 0;
	std::vector<cv::Mat> degrade_images; degrade_images.resize(num_images_in_class);
	std::vector<cv::SparseMat> DHF; DHF.resize(num_images_in_class);

	std::cout << "Recognition scale test start" << std::endl;

	std::map<int32_t, cv::Mat>::const_iterator it_etalon = etalons.begin();
	while (it_etalon != etalons.end())
	{
		it_etalon->second.copyTo(lr_gray);
		//imshow("Normal", lr_gray);

		switch (value)
		{
		case OperatedClassifier::scale:
			cvtColor(lr_gray, lr_copy, CV_GRAY2RGB);
			NS_DegradeFilter::DegradeFilter::down_scale_image(4, lr_copy, degrade_images, DHF);
			// degrade_images[0] = lr_copy;
			// DHF[0] = lr_copy;
			sr->run_filter(degrade_images,
							dest,
							DHF,
							num_etalons_in_class,
							20,
							1.3,
							0.03,
							0.7,
							cv::Size(7, 7),
							NS_SuperResolution::SR_DATA_L2);
			cv::cvtColor(dest, lr_gray, CV_RGB2GRAY);
			break;
		case OperatedClassifier::dhf:
			NS_DegradeFilter::DegradeFilter::dhf_image(1, &lr_gray);
			break;
		default:
			return 0;
		}

		//imshow("Modern", lr_gray);
		predict_label = model->predict(lr_gray);
		//imshow("Predict", images[predict_label * (num_images_in_class - num_etalons_in_class)]);

		if (predict_label == it_etalon->first)
			magic++;

		++it_etalon;
		//cv::waitKey(0);
	}

	return magic;
}

OperatedClassifier::~OperatedClassifier() {
}
