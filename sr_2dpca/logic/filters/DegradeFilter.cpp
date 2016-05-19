#include "DegradeFilter.h"

namespace NS_DegradeFilter {

	DegradeFilter::DegradeFilter() : image_temp(nullptr) {

	}

	void DegradeFilter::add_gauss_noise(cv::Mat *src, cv::Mat &dest, double sigma) 
	{
		cv::Mat noise(src->rows, src->cols, CV_32FC1);
		cv::Mat src_f;
		std::vector<cv::Mat> images;
		split(*src, images);
		for (int32_t i = 0; i < src->channels(); i++)
		{
			images[i].convertTo(src_f, CV_32FC1);
			randn(noise, cv::Scalar(0.0), cv::Scalar(sigma));
			cv::Mat temp = noise + src_f;
			temp.convertTo(images[i], CV_8UC1);
		}
		merge(images, dest);
	}

	void DegradeFilter::add_spike_noise(cv::Mat& src, cv::Mat& dest, int32_t val)
	{
		src.copyTo(dest);
		
#pragma omp parallel for
		for (int32_t j = 0; j < src.rows; j++) 
		{
			cv::RNG rnd(cv::getTickCount());
			uchar* d = dest.ptr<uchar>(j);
			for (int32_t i = 0; i < src.cols; i++)
			{
				if (rnd.uniform(0, val) < 1)
				{
					d[3 * i] = 255;
					d[3 * i + 1] = 255;
					d[3 * i + 2] = 255;
				}
			}
		}
	}

	cv::SparseMat DegradeFilter::create_degraded_image_and_sparseMat32F(cv::Mat &src, cv::Mat *dest, cv::Point2d move, int32_t amp)
	{
		cv::SparseMat DHF = create_downsampled_motionand_blur_CCDSparseMat32f(src, amp, move);

		int matsize = src.cols*src.rows;
		int matsize2 = src.cols*src.rows / (amp*amp);

		cv::Mat svec;
		src.reshape(3, matsize).convertTo(svec, CV_32FC3);
		cv::Mat dvec(matsize2, 1, CV_32FC3);

		mul_sparseMat32f(DHF, svec, dvec);

		// imshow("smat", visualize_sparse_mat(DHF, Size(512, 512 / amp / amp)));
		// waitKey(30);

		// re-convert  1D vecor structure to Mat image structure
		dvec.reshape(3, dest->rows).convertTo(*dest, CV_8UC3);

		return DHF;
	}

	cv::SparseMat DegradeFilter::create_downsampled_motionand_blur_CCDSparseMat32f(cv::Mat& src, int32_t amp, cv::Point2d move)
	{
		//(1)'
		//D down sampling matrix
		//H blur matrix, in this case, we use only ccd sampling blur.
		//F motion matrix, in this case, threr is only global shift motion.

		float div = 1.0f / ((float)(amp*amp));
		int x1 = (int)(move.x + 1);
		int x0 = (int)(move.x);
		float a1 = (float)(move.x - x0);
		float a0 = (float)(1.0 - a1);

		int y1 = (int)(move.y + 1);
		int y0 = (int)(move.y);
		float b1 = (float)(move.y - y0);
		float b0 = (float)(1.0 - b1);

		int bsx = x1;
		int bsy = y1;

		int matsize = src.cols*src.rows;
		int matsize2 = src.cols*src.rows / (amp*amp);
		int size2[2] = { matsize,matsize2 };
		cv::SparseMat DHF(2, size2, CV_32FC1);

		const int step = src.cols / amp;
		for (int j = 0;j<src.rows;j += amp)
		{
			for (int i = 0;i<src.cols;i += amp)
			{
				int y = src.cols*j + i;
				int s = step*j / amp + i / amp;

				if (i >= bsx &&i<src.cols - bsx - amp &&j >= bsy &&j<src.rows - bsy - amp)
				{
					for (int l = 0;l<amp;l++)
					{
						for (int k = 0;k<amp;k++)
						{
							DHF.ref<float>(y + src.cols*(y0 + l) + x0 + k, s) += a0*b0*div;
							DHF.ref<float>(y + src.cols*(y0 + l) + x1 + k, s) += a1*b0*div;
							DHF.ref<float>(y + src.cols*(y1 + l) + x0 + k, s) += a0*b1*div;
							DHF.ref<float>(y + src.cols*(y1 + l) + x1 + k, s) += a1*b1*div;
						}
					}
				}
			}
		}
		return DHF;
	}

	void DegradeFilter::dhf_image(int8_t rfactor, cv::Mat *src) {
		cv::Mat copy_img = cv::Mat::zeros(cv::Size(src->cols, src->rows), CV_8UC3);
		cvtColor(*src, copy_img, CV_GRAY2RGB);

		cv::RNG rnd;
		cv::Mat degrade_image = cv::Mat::zeros(copy_img.rows / rfactor, copy_img.cols / rfactor, CV_8UC3);
		cv::Mat temp = cv::Mat::zeros(copy_img.rows / rfactor, copy_img.cols / rfactor, CV_8UC3);

		cv::Point2d move(0,0);

		degrade_image.create(copy_img.rows / rfactor, copy_img.cols / rfactor, CV_8UC3);
		create_degraded_image_and_sparseMat32F(copy_img, &temp, move, rfactor);
		//add gaussian noise 
		add_gauss_noise(&temp, degrade_image, 200.0); // 10.0
		//add spike noise 
		add_spike_noise(degrade_image, degrade_image, 500); // 500
		cv::cvtColor(degrade_image, *src, CV_RGB2GRAY);
	}

	void DegradeFilter::dhf_image_multy(int8_t rfactor, cv::Mat &src, std::vector<cv::Mat> &degrade_images, std::vector<cv::SparseMat> &DHF) {
		cv::RNG rnd;
		cv::Point2d move[10];
		cv::Mat imtemp(src.rows / rfactor, src.cols / rfactor, CV_8UC3);

		for (size_t i = 0; i < degrade_images.size(); i++)
		{
			if (i == 0)
			{
				move[i].x = 0;
				move[i].y = 0;
			}
			else {
				move[i].x = rnd.uniform(0.0, 4.0);
				move[i].y = rnd.uniform(0.0, 4.0);
			}

			degrade_images[i].create(src.rows / rfactor, src.cols / rfactor, CV_8UC3);
			DHF[i] = create_degraded_image_and_sparseMat32F(src, &imtemp, move[i], rfactor);
			add_gauss_noise(&imtemp, degrade_images[i], 200.0); // 10.0
			add_spike_noise(degrade_images[i], degrade_images[i], 500); // 500
		}
	}

	void DegradeFilter::down_up_scale_image(int8_t rfactor, cv::Mat &src) {
		cv::Mat temp_img = cv::Mat::zeros(src.cols / rfactor, src.rows / rfactor, CV_8UC1);
		resize(src, temp_img, cv::Size(src.cols / rfactor, src.rows / rfactor));
		resize(temp_img, src, src.size());
	}

	void DegradeFilter::down_scale_image(int8_t rfactor, cv::Mat &src) {
		cv::Mat temp_img = cv::Mat::zeros(src.cols / rfactor, src.rows / rfactor, CV_8UC1);
		resize(src, temp_img, cv::Size(src.cols / rfactor, src.rows / rfactor));
		temp_img.copyTo(src);
	}

	void DegradeFilter::down_scale_image(int8_t rfactor, cv::Mat &src, std::vector<cv::Mat> &degrade_images, std::vector<cv::SparseMat> &DHF) {
		cv::RNG rnd;
		cv::Point2d move[10];
		cv::Mat imtemp(src.rows / rfactor, src.cols / rfactor, CV_8UC3);

		for (size_t i = 0; i < degrade_images.size(); i++)
		{
			if (i == 0)
			{
				move[i].x = 0;
				move[i].y = 0;
			}
			else {
				move[i].x = rnd.uniform(0.0, 4.0);
				move[i].y = rnd.uniform(0.0, 4.0);
			}
			
			degrade_images[i].create(src.rows / rfactor, src.cols / rfactor, CV_8UC3);

			DHF[i] = create_degraded_image_and_sparseMat32F(src, &imtemp, move[i], rfactor);
			merge_channels(&imtemp, degrade_images[i]);
		}
	}

	bool DegradeFilter::generate_degrade_images(int8_t image_count,
												int8_t rfactor, 
												cv::Mat src,
												std::vector<cv::SparseMat> *A, 
												std::vector<cv::Mat> *degrade_images) 
	{
		if (src.empty() || A == nullptr || degrade_images == nullptr || image_count == 0 || rfactor == 0 || A->size() != image_count ||
			degrade_images->size() != image_count) 
		{
			return false;
		}

		try {
			if (!move.empty())
				move.clear();
			move.resize(image_count);

			if (image_temp != nullptr)
				delete image_temp;
			image_temp = new cv::Mat(src.rows / rfactor, src.cols / rfactor, CV_8UC3);

			for (int32_t i = 0; i < image_count; i++)
			{
				if (i == 0) {
					move.front().x = 0;
					move.front().y = 0;
				}
				else {
					// рандомное смещение по оси x
					move[i].x = rnd.uniform(0.0, 4.0);
					// рандомное смещение по оси y
					move[i].y = rnd.uniform(0.0, 4.0);
				}

				degrade_images->at(i).create(src.rows / rfactor, src.cols / rfactor, CV_8UC3);
				A->at(i) = create_degraded_image_and_sparseMat32F(src, image_temp, move[i], rfactor);
				//add gaussian noise 
				add_gauss_noise(image_temp, degrade_images->at(i), 10.0);
				//add spike noise 
				add_spike_noise(degrade_images->at(i), degrade_images->at(i), 500);
				cv::imshow("degrade_image", degrade_images->at(i));
			}
		}
		catch (...) {
			return false;
		}

		return true;
	}

	void DegradeFilter::merge_channels(cv::Mat *src, cv::Mat &dest) {
		cv::Mat src_f;
		std::vector<cv::Mat> images;
		split(*src, images);
		for (int c = 0;c<src->channels();c++)
		{
			images[c].convertTo(src_f, CV_32FC1);
			src_f.convertTo(images[c], CV_8UC1);
		}
		merge(images, dest);
	}

	void DegradeFilter::mul_sparseMat32f(cv::SparseMat& smat, cv::Mat& src, cv::Mat& dest, bool isTranspose)
	{
		dest.setTo(0);
		cv::SparseMatIterator it = smat.begin(), it_end = smat.end();
		if (!isTranspose)
		{
			for (;it != it_end;++it)
			{
				int i = it.node()->idx[0];
				int j = it.node()->idx[1];
				float* d = dest.ptr<float>(j);
				float* s = src.ptr<float>(i);
				for (int i = 0;i<3;i++)
				{
					d[i] += it.value<float>() * s[i];
				}
			}
		}
		else // for transpose matrix
		{
			for (;it != it_end;++it)
			{
				int i = it.node()->idx[1];
				int j = it.node()->idx[0];
				float* d = dest.ptr<float>(j);
				float* s = src.ptr<float>(i);
				for (int i = 0;i<3;i++)
				{
					d[i] += it.value<float>() * s[i];
				}
			}
		}
	}

	cv::Mat DegradeFilter::visualize_sparse_mat(cv::SparseMat& smat, cv::Size out_imsize)
	{
		cv::Mat data = cv::Mat::zeros(out_imsize, CV_8U);
		double inv_size0 = 1.0 / smat.size(0);
		double inv_size1 = 1.0 / smat.size(1);

		cv::SparseMatIterator it = smat.begin(), it_end = smat.end();
		for (;it != it_end;++it)
		{
			int j = (int)(((double)it.node()->idx[0] * inv_size0*out_imsize.height));
			int i = (int)(((double)it.node()->idx[1] * inv_size1*out_imsize.width));
			data.at<uchar>(j, i) = 255;
		}

		cv::Mat zeromat = cv::Mat::zeros(out_imsize, CV_8U);
		std::vector<cv::Mat> image;
		image.push_back(zeromat);
		image.push_back(data);
		image.push_back(zeromat);

		cv::Mat ret;
		cv::merge(image, ret);

		std::cout << "number of non zero elements: " << smat.nzcount() << std::endl;
		return ret;
	}

	DegradeFilter::~DegradeFilter() {
		if (image_temp)
			delete image_temp;
	}
}