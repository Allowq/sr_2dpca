#include "EigenRecognize.h"

namespace NS_EigenRecognize {

	EigenRecognize::EigenRecognize(std::string _csv_path) 
		: csv_path(_csv_path)
	{
	}

	void EigenRecognize::read_csv(const std::string &file_name, 
								  char separator) 
	{
		std::ifstream file(file_name.c_str(), std::ifstream::in | std::ifstream::binary);
		if (!file) {
			std::string error_message = "No valid input file was given, please check the given filename.";
			CV_Error(CV_StsBadArg, error_message);
		}
		std::string line, path, class_label;
		while (getline(file, line)) {
			std::stringstream liness(line);
			getline(liness, path, separator);
			getline(liness, class_label);
			if (!path.empty() && !class_label.empty()) {
				images.push_back(cv::imread(path, 0));
				labels.push_back(atoi(class_label.c_str()));
			}
		}
	}

	EigenRecognize::~EigenRecognize() {
	}
}
