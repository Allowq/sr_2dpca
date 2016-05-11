#ifndef _OPERATED_CLASSIFIER_H_
#define _OPERATED_CLASSIFIER_H_

#pragma once

#include <opencv2\opencv.hpp>
#include <opencv2\face\facerec.hpp>

#include "..\filters\DegradeFilter.h"

#include <stdint.h>
#include <fstream>

class OperatedClassifier
{
private:
	uint32_t num_classes;
	uint32_t num_etalons_in_class;
	uint32_t num_images_in_class;

	std::vector<cv::Mat> images;
	std::vector<int32_t> labels;

	void read_csv(const std::string &file_name, char separator = ';');
	int32_t run_dhf(cv::Ptr<cv::face::BasicFaceRecognizer> &model) const;
	int32_t run_easy(cv::Ptr<cv::face::BasicFaceRecognizer> &model) const;
	int32_t OperatedClassifier::run_scale(cv::Ptr<cv::face::BasicFaceRecognizer> &model) const;

public:
	explicit OperatedClassifier();

	bool run(std::string csv_path);

	virtual ~OperatedClassifier();
};

#endif