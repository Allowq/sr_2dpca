#ifndef _DEGRADERFILTER_H_
#define _DEGRADERFILTER_H_

#pragma once

#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>

#include <stdint.h>
#include <vector>
#include <iostream>

#ifdef _OPENMP
	#include <omp.h>
#endif

namespace NS_DegradeFilter {

	class DegradeFilter
	{
	private:
		// генератор случайных чисел для смещения генерируемых изображений
		cv::RNG rnd;
		// вектора смещений для генерируемых изображений
		std::vector<cv::Point2d> move;
		// матрица под временное хранение изображений
		cv::Mat *image_temp;

		// добавить гаусов-шум
		static void add_gauss_noise(cv::Mat *src, cv::Mat &dest, double sigma);
		// добавить шум в виде всплесков
		static void add_spike_noise(cv::Mat& src, cv::Mat& dest, int32_t val);

		static cv::SparseMat create_degraded_image_and_sparseMat32F(cv::Mat& src, cv::Mat *dest, cv::Point2d move, int32_t amp);
		static cv::SparseMat create_downsampled_motionand_blur_CCDSparseMat32f(cv::Mat& src, int32_t amp, cv::Point2d move);
		static void merge_channels(cv::Mat *src, cv::Mat &dest);
		static void mul_sparseMat32f(cv::SparseMat& smat, cv::Mat& src, cv::Mat& dest, bool isTranspose = false);
		cv::Mat visualize_sparse_mat(cv::SparseMat& smat, cv::Size out_imsize);

	public:
		explicit DegradeFilter();

		// сгенерировать набор изображений, подвергнувшихся применению фильтра
		bool generate_degrade_images(int8_t image_count,
									 int8_t rfactor,
									 cv::Mat src, 
									 std::vector<cv::SparseMat> *A, 
									 std::vector<cv::Mat> *degrade_images);

		static void down_up_scale_image(int8_t rfactor, cv::Mat &src);
		static void down_scale_image(int8_t rfactor, cv::Mat &src);
		static void down_scale_image(int8_t rfactor, cv::Mat &src, std::vector<cv::Mat> &degrade_images, std::vector<cv::SparseMat> &DHF);
		static void dhf_image(int8_t rfactor, cv::Mat *src);
		static void dhf_image_multy(int8_t rfactor, cv::Mat &src, std::vector<cv::Mat> &degrade_images, std::vector<cv::SparseMat> &DHF);

		virtual ~DegradeFilter();
	};
}

/*
	How to use?

	std::vector<cv::SparseMat> A; A.resize(16);
	std::vector<cv::Mat> degrade_images; degrade_images.resize(16);
	IplImage *frame;

	frame = cvCreateImage(cvSize(video_input.getWidth(camera_index), video_input.getHeight(camera_index)), IPL_DEPTH_8U, 3);

	degrade_filter->generate_degrade_images(16, 4, cv::cvarrToMat(frame), &A, &degrade_images);


*/

#endif