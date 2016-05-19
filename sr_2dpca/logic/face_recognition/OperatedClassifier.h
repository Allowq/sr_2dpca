#ifndef _OPERATED_CLASSIFIER_H_
#define _OPERATED_CLASSIFIER_H_

#pragma once

#include <opencv2\opencv.hpp>
#include <opencv2\face\facerec.hpp>

#include "..\filters\DegradeFilter.h"
#include "..\super_resolution\SuperResolution.h"

#include <stdint.h>
#include <fstream>
#include <map>

class OperatedClassifier
{
	enum FILTER_TYPE_ENUM {
		none = 0, scale, dhf
	};

private:
	uint32_t num_classes;
	uint32_t num_etalons_in_class;
	uint32_t num_images_in_class;

	std::vector<cv::Mat> images;
	std::map<int32_t, cv::Mat> etalons;
	std::vector<int32_t> labels;

	void read_csv(const std::string &file_name, char separator = ';');
	int32_t apply_filter(cv::Ptr<cv::face::BasicFaceRecognizer> &model, FILTER_TYPE_ENUM value) const;
	void apply_filter_sr(cv::Ptr<cv::face::BasicFaceRecognizer> &model, NS_SuperResolution::SuperResolution *sr, FILTER_TYPE_ENUM value) const;

public:
	explicit OperatedClassifier();

	bool run(std::string csv_path);

	virtual ~OperatedClassifier();
};

#endif