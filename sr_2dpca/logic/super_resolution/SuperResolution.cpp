#include "SuperResolution.h"

namespace NS_SuperResolution {

#define sign_float(a,b) (a>b)?1.0f:(a<b)?-1.0f:0.0f

	SuperResolution::SuperResolution() {
	}

	void SuperResolution::bilateral_total_variation_sr(std::vector<cv::Mat> &degrade_images,
													   cv::Mat& dest,
													   std::vector<cv::SparseMat> &DHF,
													   const int32_t num_of_view,
													   int32_t iteration,
													   float beta,
													   float lambda,
													   float alpha,
													   cv::Size reg_window,
													   int32_t method,
													   cv::Mat ideal,
													   uint32_t test_step)
	{
		//(3) create initial image by simple linear interpolation
		resize(degrade_images[0], dest, dest.size());
		// std::cout << "PSNR" << get_PSNR(dest, ideal, 10) << "dB" << std::endl;

		//(4)convert Mat image structure to 1D vecor structure
		cv::Mat dest_vec;
		dest.reshape(3, dest.cols * dest.rows).convertTo(dest_vec, CV_32FC3);

		cv::Mat *dest_vec_temp = new cv::Mat[num_of_view];
		cv::Mat *svec = new cv::Mat[num_of_view];
		cv::Mat *svec2 = new cv::Mat[num_of_view];

		for (int32_t n = 0; n < num_of_view; n++)
		{
			degrade_images[n].reshape(3, degrade_images[0].cols * degrade_images[0].rows).convertTo(svec[n], CV_32FC3);
			degrade_images[n].reshape(3, degrade_images[0].cols * degrade_images[0].rows).convertTo(svec2[n], CV_32FC3);

			dest_vec_temp[n] = dest_vec.clone();
		}

		//regularization vector
		cv::Mat reg_vec = cv::Mat::zeros(dest.rows * dest.cols, 1, CV_32FC3);

		//(5)steepest descent method for L1 norm minimization
		for (int32_t i = 0; i < iteration; i++)
		{
			std::cout << "iteration" << i << std::endl;
			int64 t = cv::getTickCount();
			cv::Mat diff = cv::Mat::zeros(dest_vec.size(), CV_32FC3);

			//(5-1)btv
			if (lambda > 0.0) 
				btv_regularization(dest_vec, reg_window, alpha, reg_vec, dest.size());

#pragma omp parallel for
			for (int32_t n = 0; n < num_of_view; n++)
			{
				//degrade current estimated image
				mul_sparseMat32f(DHF[n], dest_vec, svec2[n]);

				//compere input and degraded image
				cv::Mat temp(degrade_images[0].cols * degrade_images[0].rows, 1, CV_32FC3);
				if (method == SR_DATA_L1)
				{
					subtract_sign(svec2[n], svec[n], temp);
				}
				else
				{
					subtract(svec2[n], svec[n], temp);
					// temp = svec2[n]- svec[n]; //supported in OpenCV2.1
				}

				//blur the subtructed vector with transposed matrix
				mul_sparseMat32f(DHF[n], temp, dest_vec_temp[n], true);
			}

			//creep ideal image, beta is parameter of the creeping speed.
			//add transeposed difference vector. sum_float_OMP is parallelized function of following for loop
			/*
			for(int32_t n = 0; n < num_of_view; n++)
			{
				addWeighted(dest_vec, 1.0, dest_vec_temp[n], -beta, 0.0, dest_vec);
				//dstvec -= (beta*dstvectemp[n]);//supported in OpenCV2.1
			}
			*/

			sum_float_OMP(dest_vec_temp, dest_vec, num_of_view, beta);


			//add smoothness term
			if (lambda > 0.0)
			{
				addWeighted(dest_vec, 1.0, reg_vec, -beta*lambda, 0.0, dest_vec);

				//supported in OpenCV2.1
				//dstvec -=lambda*beta*reg_vec; 
			}

			// show SR imtermediate process information. these processes does not be required at actural implimentation.
			dest_vec.reshape(3, dest.rows).convertTo(dest, CV_8UC3);
			std::cout << "PSNR" << get_PSNR(dest, ideal, 10) << "dB" << std::endl;

			//char name[64];
			//sprintf(name, "%03d: %.1f dB", i, get_PSNR(dest, ideal, 10));
			//putText(dest, name, cv::Point(15, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, CV_RGB(255, 255, 255), 2);

			//sprintf(name, "iteration%04d.png", i);

			// imshow("SRimage", dest);
			// cv::waitKey(30);
			// imwrite(name, dest);
			std::cout << "time/iteration" << (cv::getTickCount() - t)*1000.0 / cv::getTickFrequency() << "ms" << std::endl;
		}

		//re-convert  1D vecor structure to Mat image structure
		dest_vec.reshape(3, dest.rows).convertTo(dest, CV_8UC3);

		char sr_rezult[64];
		sprintf(sr_rezult, "sr_rezult_%03d.png", test_step);

		imwrite(sr_rezult, dest);

		if (dest_vec_temp)
			delete[] dest_vec_temp;
		if (svec)
			delete[] svec;
		if (svec2)
			delete[] svec2;
	}

	void SuperResolution::btv_regularization(cv::Mat &src_vec, cv::Size kernel, float alpha, cv::Mat &dst_vec, cv::Size size)
	{
		cv::Mat src;
		src_vec.reshape(3, size.height).convertTo(src, CV_32FC3);
		cv::Mat dest(size.height, size.width, CV_32FC3);

		const int kw = (kernel.width - 1) / 2;
		const int kh = (kernel.height - 1) / 2;

		float* weight = new float[kernel.width*kernel.height];
		for (int m = 0, count = 0;m <= kh;m++)
		{
			for (int l = kw;l + m >= 0;l--)
			{
				weight[count] = pow(alpha, abs(m) + abs(l));
				count++;
			}
		}
		
		//a part of under term of Eq (22) lambda*\sum\sum ...
#pragma omp parallel for
		for (int j = kh;j<src.rows - kh;j++)
		{
			float* s = src.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			for (int i = kw;i<src.cols - kw;i++)
			{
				float r = 0.0;
				float g = 0.0;
				float b = 0.0;

				const float sr = s[3 * (i)+0];
				const float sg = s[3 * (i)+1];
				const float sb = s[3 * (i)+2];
				for (int m = 0, count = 0;m <= kh;m++)
				{
					float* s2 = src.ptr<float>(j - m);
					float* ss = src.ptr<float>(j + m);
					for (int l = kw;l + m >= 0;l--)
					{
						r += weight[count] * (sign_float(sr, ss[3 * (i + l) + 0]) - sign_float(s2[3 * (i - l) + 0], sr));
						g += weight[count] * (sign_float(sg, ss[3 * (i + l) + 1]) - sign_float(s2[3 * (i - l) + 1], sg));
						b += weight[count] * (sign_float(sb, ss[3 * (i + l) + 2]) - sign_float(s2[3 * (i - l) + 2], sb));
						count++;
					}
				}
				d[3 * i + 0] = r;
				d[3 * i + 1] = g;
				d[3 * i + 2] = b;
			}
		}
		dest.reshape(3, size.height*size.width).convertTo(dst_vec, CV_32FC3);
		delete[] weight;
	}

	double SuperResolution::get_PSNR(cv::Mat& src1, cv::Mat& src2, int32_t bb)
	{
		int i, j;
		double sse, mse, psnr;
		sse = 0.0;

		cv::Mat s1, s2;
		cvtColor(src1, s1, CV_BGR2GRAY);
		cvtColor(src2, s2, CV_BGR2GRAY);

		int count = 0;
		for (j = bb;j<s1.rows - bb;j++)
		{
			uchar* d = s1.ptr(j);
			uchar* s = s2.ptr(j);

			for (i = bb;i<s1.cols - bb;i++)
			{
				sse += ((d[i] - s[i])*(d[i] - s[i]));
				count++;
			}
		}
		if (sse == 0.0 || count == 0)
		{
			return 0;
		}
		else
		{
			mse = sse / (double)(count);
			psnr = 10.0*log10((255 * 255) / mse);
			return psnr;
		}
	}

	void SuperResolution::mul_sparseMat32f(cv::SparseMat &smat, cv::Mat &src, cv::Mat &dest, bool is_transpose)
	{
		dest.setTo(0);
		cv::SparseMatIterator it = smat.begin(), it_end = smat.end();
		if (!is_transpose)
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

	void SuperResolution::subtract_sign(cv::Mat &src1, cv::Mat &src2, cv::Mat &dest)
	{
		for (int j = 0;j<src1.rows;j++)
		{
			float* s1 = src1.ptr<float>(j);
			float* s2 = src2.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			for (int i = 0;i<src1.cols;i++)
			{
				d[3 * i] = sign_float(s1[3 * i], s2[3 * i]);
				d[3 * i + 1] = sign_float(s1[3 * i + 1], s2[3 * i + 1]);
				d[3 * i + 2] = sign_float(s1[3 * i + 2], s2[3 * i + 2]);
			}
		}
	}

	void SuperResolution::sum_float_OMP(cv::Mat src[], cv::Mat& dest, int32_t numofview, float beta)
	{
		for (int n = 0;n<numofview;n++)
		{
#pragma omp parallel for
			for (int j = 0;j<dest.rows;j++)
			{
				dest.ptr<float>(j)[0] -= beta*src[n].ptr<float>(j)[0];
				dest.ptr<float>(j)[1] -= beta*src[n].ptr<float>(j)[1];
				dest.ptr<float>(j)[2] -= beta*src[n].ptr<float>(j)[2];
			}
		}
	}

	SuperResolution::~SuperResolution() {
	}

}