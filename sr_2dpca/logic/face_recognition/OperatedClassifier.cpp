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
	cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::createFisherFaceRecognizer(); // cv::face::createEigenFaceRecognizer();
	model->train(images, labels);
	std::cout << "-- Training complete --" << std::endl << std::endl;

	/*
	int32_t results = this->apply_filter(model, FILTER_TYPE_ENUM::none);
	std::string result_message = "None filter results:\n";
	result_message.append(cv::format("Recognized %d from %d images\nRecognition quality: %3.2f", results, num_classes, float(results)/num_classes));
	std::cout << result_message << std::endl << std::endl;
	*/

	/*
	int32_t results = this->apply_filter(model, FILTER_TYPE_ENUM::scale);
	std::string result_message = "Scale filter results:\n";
	result_message.append(cv::format("Recognized %d from %d images\nRecognition quality: %3.2f", results, num_classes, float(results) / num_classes));
	std::cout << result_message << std::endl << std::endl;
	*/

	/*
	int32_t results = this->apply_filter(model, FILTER_TYPE_ENUM::dhf);
	std::string result_message = "DHF filter results:\n";
	result_message.append(cv::format("Recognized %d from %d images\nRecognition quality: %3.2f", results, num_classes, float(results) / num_classes));
	std::cout << result_message << std::endl << std::endl;
	*/

	/*
	std::cout << "Recognition test (base on super-resolution) start" << std::endl;
	this->apply_filter_sr(model, sr, FILTER_TYPE_ENUM::scale);
	std::cout << "Recognition test (base on super-resolution) end" << std::endl;
	*/

	std::cout << "Recognition test (base on super-resolution) start" << std::endl;
	this->apply_filter_sr(model, sr, FILTER_TYPE_ENUM::dhf);
	std::cout << "Recognition test (base on super-resolution) end" << std::endl;

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
		imshow("Normal", test_sample);

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

		imshow("Modern", test_sample);
		predict_label = model->predict(test_sample);
		imshow("Predict", images[predict_label * (num_images_in_class - num_etalons_in_class)]);

		if (predict_label == it_etalon->first)
			magic++;

		++it_etalon;
		cv::waitKey(0);
	}

	return magic;
}

void OperatedClassifier::apply_filter_sr(cv::Ptr<cv::face::BasicFaceRecognizer> &model, 
										 NS_SuperResolution::SuperResolution *sr, 
										 FILTER_TYPE_ENUM value) const 
{
	cv::Mat dest = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_8UC1), 
			lr_gray = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_8UC1),
			lr_copy = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_8UC3);

	std::vector<cv::Mat> degrade_images; degrade_images.resize(num_images_in_class);
	std::vector<cv::SparseMat> DHF; DHF.resize(num_images_in_class);

	int32_t predict_label = -1, // предполагаемый результат распознавания 
			magic = 0;			// количество распознанных образов

	uint32_t reduce_value = 1;

	std::string result_message = "Image filter:";
	switch (value)
	{
	case OperatedClassifier::none:
		result_message.append("none\n");
		break;
	case OperatedClassifier::scale:
		result_message.append("scale\n");
		break;
	case OperatedClassifier::dhf:
		result_message.append("dhf\n");
		break;
	}

	result_message.append("Reduce factor = ").append(std::to_string(reduce_value)).append("\n");
	result_message.append("Length of sequence = ").append(std::to_string(num_images_in_class)).append("\n");
	std::cout << result_message << std::endl << std::endl;

	std::map<int32_t, cv::Mat>::const_iterator it_etalon;
	NS_SuperResolution::NORM_VALUE norm_value = NS_SuperResolution::SR_DATA_L2;
	float beta = 1.4f, // 3.3f
		  lambda = 0.004f, // 0.01f
		  alpha = 0.7f; // 1.1

	int32_t iteration_count = 36,
			 kernel_zise = 4, // 3
			 image_index;

	int64_t timer;

	while (magic != 41)
	{
		image_index = 0;
		magic = 0;
		it_etalon = etalons.begin();
		timer = cv::getTickCount();

		while (it_etalon != etalons.end())
		{
			if (it_etalon == etalons.begin())
			{
				std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl
					<< "Number of iteration per image = " << iteration_count << std::endl
					<< "Norm: L" << norm_value + 1 << std::endl
					<< "Kernel size (" << kernel_zise << "x" << kernel_zise << ")" << std::endl
					<< "Alpha = " << alpha << std::endl
					<< "Beta = " << beta << std::endl
					<< "Lambda = " << lambda << std::endl << std::endl;
			}

			it_etalon->second.copyTo(lr_gray);
			// imshow("Normal", lr_gray);

			cvtColor(lr_gray, lr_copy, CV_GRAY2RGB);
			switch (value)
			{
			case OperatedClassifier::scale:
				NS_DegradeFilter::DegradeFilter::down_scale_image(reduce_value, lr_copy, degrade_images, DHF);
				break;
			case OperatedClassifier::dhf:
				NS_DegradeFilter::DegradeFilter::dhf_image_multy(reduce_value, lr_copy, degrade_images, DHF);
				break;
			default:
				return;
			}

			sr->run_filter(degrade_images,
							dest,
							DHF,
							num_etalons_in_class,
							iteration_count,
							beta,
							lambda,
							alpha,
							cv::Size(kernel_zise, kernel_zise),
							norm_value);

			cv::cvtColor(dest, lr_gray, CV_RGB2GRAY);

			std::cout << "Image index = " << image_index << std::endl;
			std::cout << "PSNR = " << sr->get_PSNR(lr_copy, dest, 10) << std::endl;

			// imshow("Modern", lr_gray);
			predict_label = model->predict(lr_gray);
			// imshow("Predict", images[predict_label * (num_images_in_class - num_etalons_in_class)]);

			if (predict_label == it_etalon->first) {
				std::cout << "Recognized" << std::endl << std::endl;
				magic++;
			}
			else
				std::cout << "Unrecognized" << std::endl << std::endl;

			image_index++;
			++it_etalon;
			// cv::waitKey(0);
		}

		std::cout << "Experiment time: " << (cv::getTickCount() - timer) * 1000.0 / cv::getTickFrequency() << "ms" << std::endl;
		result_message = cv::format("Recognized %d from %d images\nRecognition quality: %3.2f", magic, num_classes, float(magic) / num_classes);
		std::cout << result_message << std::endl << std::endl << std::endl;
	}
}

OperatedClassifier::~OperatedClassifier() {
}
