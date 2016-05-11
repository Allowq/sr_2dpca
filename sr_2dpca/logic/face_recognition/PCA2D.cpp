#include "PCA2D.h"

namespace NS_PCA2D {

	PCA2D::PCA2D(uint32_t _num_classes,
				 uint32_t _num_etalons_in_class,
				 uint32_t _num_images_in_class)
		: num_classes(_num_classes),
		  num_etalons_in_class(_num_etalons_in_class),
		  num_images_in_class(_num_images_in_class)
	{
	}

	void PCA2D::read_csv(const std::string &file_name, char separator)
	{
		std::ifstream file(file_name.c_str(), std::ifstream::in | std::ifstream::binary);
		if (!file) 
		{
			std::string error_message = "No valid input file was given, please check the given filename.";
			CV_Error(CV_StsBadArg, error_message);
		}

		std::string line, path, class_label;
		std::string folder = file_name.substr(0, file_name.find_last_of("/\\"));

		while (getline(file, line)) 
		{
			std::stringstream liness(line);
			path.clear();
			getline(liness, path, separator);
			getline(liness, class_label);
			path = folder + path;
			if (!path.empty() && !class_label.empty()) 
			{
				images.push_back(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE));
			//	labels.push_back(atoi(class_label.c_str()));
			}
		}
		file.close();
	}

	bool PCA2D::training(const std::string &csv_path) {
		system("cls");
		std::cout << ">> Training started";
		images.clear();
		try {
			read_csv(csv_path);
		}
		catch (cv::Exception& e) {
			std::cerr << "Error opening file \"" << csv_path << "\". Reason: " << e.msg << std::endl;
			return false;
		}

		cv::Mat image_mean = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_32S);
		cv::Mat image_temp = cv::Mat::zeros(cv::Size(images[0].cols, images[0].rows), CV_32S);

		for each (cv::Mat img_index in images) {
			img_index.convertTo(image_temp, CV_32S);
			image_mean += image_temp;
		}
		image_mean /= num_classes * num_images_in_class;
		image_mean.convertTo(image_mean, CV_8UC1);
		cv::imshow("Mean", image_mean);

		centre_images.clear();
		for each (cv::Mat img_index in images) 
			centre_images.push_back(img_index - image_mean);

		/*

		cv::Mat covariance_row = cv::Mat::zeros(cv::Size(images[0].cols, images[0].cols), CV_32S);
		cv::Mat covariance_column = cv::Mat::zeros(cv::Size(images[0].rows, images[0].rows), CV_32S);
		
		for each (cv::Mat img_index in centre_images) {
			covariance_row += img_index.mul(img_index.t());
			covariance_column += img_index.t().mul(img_index);
		}
		*/

		cv::waitKey(5000);

		return true;
	}

	PCA2D::~PCA2D() {
	}
}