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
	file.close();
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

	std::cout << "Training start" << std::endl;
	cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::createFisherFaceRecognizer(); // cv::face::createEigenFaceRecognizer();
	model->train(images, labels);

	int32_t results = this->run_easy(model);
	
	std::string result_message = cv::format("Easy run >> Recognized = %d / Number of classes = %d.", results, num_classes);
	std::cout << result_message << std::endl << std::endl;

	results = this->run_scale(model);
	result_message = cv::format("Scale run >> Recognized = %d / Number of classes = %d.", results, num_classes);
	std::cout << result_message << std::endl << std::endl;

	results = this->run_dhf(model);
	result_message = cv::format("DHF run >> Recognized = %d / Number of classes = %d.", results, num_classes);
	std::cout << result_message << std::endl << std::endl;

	cv::waitKey(0);

	getchar();

	return true;
}

int32_t OperatedClassifier::run_dhf(cv::Ptr<cv::face::BasicFaceRecognizer> &model) const {
	cv::Mat test_sample = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_8UC1);
	int32_t test_label = -1,	// номер изображения в наборе
			predict_label = -1, // предполагаемый результат распознавания 
			max_freq = 0,		// частота встречаемости элемента
			total_images = num_classes * num_images_in_class,
			magic = 0;

	std::map<uint32_t, uint32_t> freq_occ;
	std::map<uint32_t, uint32_t>::iterator itr = freq_occ.end();

	std::cout << "Recognition dhf test start" << std::endl;
	for (size_t i_class = 0; i_class < num_classes; i_class++)
	{
		freq_occ.clear();
		max_freq = 0;
		predict_label = -1;

		for (size_t j_etalons = 0; j_etalons < num_etalons_in_class; j_etalons++)
		{
			test_label = num_images_in_class*i_class + j_etalons;
			test_sample = images[test_label];

			if (i_class == 0)
				imshow("Normal", test_sample);

			NS_DegradeFilter::DegradeFilter::dhf_image(4, &test_sample);
			if (i_class == 0)
				imshow("Degrade", test_sample);

			predict_label = model->predict(test_sample);
			freq_occ[predict_label]++;
		}

		itr = freq_occ.begin();
		while (itr != freq_occ.end())
		{
			if (itr->second > max_freq)
			{
				max_freq = itr->second;
				predict_label = itr->first;
			}
			++itr;
		}

		if (predict_label == i_class)
			magic++;
	}

	return magic;
}

int32_t OperatedClassifier::run_easy(cv::Ptr<cv::face::BasicFaceRecognizer> &model) const {
	cv::Mat test_sample = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_8UC1);
	int32_t test_label = -1,	// номер изображения в наборе
			predict_label = -1, // предполагаемый результат распознавания 
			max_freq = 0,		// частота встречаемости элемента
			total_images = num_classes * num_images_in_class,
			magic = 0;

	std::map<uint32_t, uint32_t> freq_occ;
	std::map<uint32_t, uint32_t>::iterator itr = freq_occ.end();
	
	std::cout << "Recognition easy test start" << std::endl;
	for (size_t i_class = 0; i_class < num_classes; i_class++)
	{
		freq_occ.clear();
		max_freq = 0;
		predict_label = -1;

		for (size_t j_etalons = 0; j_etalons < num_etalons_in_class; j_etalons++)
		{
			test_label = num_images_in_class*i_class + j_etalons;
			test_sample = images[test_label];

			predict_label = model->predict(test_sample);
			freq_occ[predict_label]++;
		}

		itr = freq_occ.begin();
		while (itr != freq_occ.end())
		{
			if (itr->second > max_freq)
			{
				max_freq = itr->second;
				predict_label = itr->first;
			}
			++itr;
		}

		if (predict_label == i_class)
			magic++;
	}

	return magic;
}

int32_t OperatedClassifier::run_scale(cv::Ptr<cv::face::BasicFaceRecognizer> &model) const {
	cv::Mat test_sample = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_8UC1);
	int32_t test_label = -1,	// номер изображения в наборе
			predict_label = -1, // предполагаемый результат распознавания 
			max_freq = 0,		// частота встречаемости элемента
			total_images = num_classes * num_images_in_class,
			magic = 0;

	std::map<uint32_t, uint32_t> freq_occ;
	std::map<uint32_t, uint32_t>::iterator itr = freq_occ.end();

	std::cout << "Recognition scale test start" << std::endl;
	for (size_t i_class = 0; i_class < num_classes; i_class++)
	{
		freq_occ.clear();
		max_freq = 0;
		predict_label = -1;

		for (size_t j_etalons = 0; j_etalons < num_etalons_in_class; j_etalons++)
		{
			test_label = num_images_in_class*i_class + j_etalons;
			test_sample = images[test_label];

			NS_DegradeFilter::DegradeFilter::down_up_scale_image(4, test_sample);

			predict_label = model->predict(test_sample);
			freq_occ[predict_label]++;
		}

		itr = freq_occ.begin();
		while (itr != freq_occ.end())
		{
			if (itr->second > max_freq)
			{
				max_freq = itr->second;
				predict_label = itr->first;
			}
			++itr;
		}

		if (predict_label == i_class)
			magic++;
	}

	return magic;
}

OperatedClassifier::~OperatedClassifier() {
}
