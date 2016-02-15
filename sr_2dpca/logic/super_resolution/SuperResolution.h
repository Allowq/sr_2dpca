#ifndef _SUPERRESOLUTION_H_
#define _SUPERRESOLUTION_H_

#pragma once

#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>

#include <stdint.h>

#ifdef _OPENMP
	#include <omp.h>
#endif

namespace NS_SuperResolution {

	enum {
		SR_DATA_L1 = 0,
		SR_DATA_L2
	};

	class SuperResolution
	{
	private:
		void btv_regularization(cv::Mat &src_vec, cv::Size kernel, float alpha, cv::Mat &dst_vec, cv::Size size);
		double get_PSNR(cv::Mat& src1, cv::Mat& src2, int32_t bb);
		void mul_sparseMat32f(cv::SparseMat &smat, cv::Mat &src, cv::Mat &dest, bool is_transpose = false);
		void subtract_sign(cv::Mat &src1, cv::Mat &src2, cv::Mat &dest);
		void sum_float_OMP(cv::Mat src[], cv::Mat& dest, int32_t numofview, float beta);

	public:
		explicit SuperResolution();

		// super_resolution_sparseMat32f
		void bilateral_total_variation_sr(std::vector<cv::Mat> &degrade_images,
										  cv::Mat& dest,
										  std::vector<cv::SparseMat> &DHF,
										  const int32_t num_of_view, 
										  int32_t iteration, 
										  float beta, 
										  float lambda, 
										  float alpha, 
										  cv::Size reg_window,
										  int32_t method);

		virtual ~SuperResolution();
	};
}

#endif