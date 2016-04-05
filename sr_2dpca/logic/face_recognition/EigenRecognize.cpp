#include "EigenRecognize.h"

namespace NS_EigenRecognize {

	EigenRecognize::EigenRecognize(std::string _csv_path) 
		: csv_path(_csv_path)
	{
		cv::setNumThreads(cv::getNumberOfCPUs());
	}

	cv::Mat EigenRecognize::norm_0_255(cv::InputArray _src) {
		cv::Mat src = _src.getMat();
		// Create and return normalized image:
		cv::Mat dst;
		switch (src.channels()) {
		case 1:
			cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			break;
		case 3:
			cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
			break;
		default:
			src.copyTo(dst);
			break;
		}
		return dst;
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
	}

	int32_t EigenRecognize::run_test(std::string output_path) {
		// Read in the data. This can fail if no valid
		// input filename is given.
		try {
			read_csv(csv_path);
		}
		catch (cv::Exception& e) {
			std::cerr << "Error opening file \"" << csv_path << "\". Reason: " << e.msg << std::endl;
			// nothing more we can do
			exit(1);
		}
		// Quit if there are not enough images for this demo.
		if (images.size() <= 1) {
			std::string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
			CV_Error(cv::Error::StsError, error_message);
		}
		// Get the height from the first image. We'll need this
		// later in code to reshape the images to their original
		// size:
		int height = images[0].rows;
		// The following lines simply get the last images from
		// your dataset and remove it from the vector. This is
		// done, so that the training data (which we learn the
		// cv::BasicFaceRecognizer on) and the test data we test
		// the model with, do not overlap.
		cv::Mat testSample = images[images.size() - 1];
		int testLabel = labels[labels.size() - 1];
		//images.pop_back();
		//labels.pop_back();

		// The following lines create an Eigenfaces model for
		// face recognition and train it with the images and
		// labels read from the given CSV file.
		// This here is a full PCA, if you just want to keep
		// 10 principal components (read Eigenfaces), then call
		// the factory method like this:
		//
		//      cv::createEigenFaceRecognizer(10);
		//
		// If you want to create a FaceRecognizer with a
		// confidence threshold (e.g. 123.0), call it with:
		//
		//      cv::createEigenFaceRecognizer(10, 123.0);
		//
		// If you want to use _all_ Eigenfaces and have a threshold,
		// then call the method like this:
		//
		//      cv::createEigenFaceRecognizer(0, 123.0);
		//
		
		// cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::createEigenFaceRecognizer();
		cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::createFisherFaceRecognizer();

		model->train(images, labels);
		// The following line predicts the label of a given
		// test image:
		int predictedLabel = model->predict(testSample);
		//
		// To get the confidence of a prediction call the model with:
		//
		//      int predictedLabel = -1;
		//      double confidence = 0.0;
		//      model->predict(testSample, predictedLabel, confidence);
		//
		std::string result_message = cv::format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
		std::cout << result_message << std::endl;
		// Here is how to get the eigenvalues of this Eigenfaces model:
		cv::Mat eigenvalues = model->getEigenValues();
		// And we can do the same to display the Eigenvectors (read Eigenfaces):
		cv::Mat W = model->getEigenVectors();
		// Get the sample mean from the training data
		cv::Mat mean = model->getMean();
		// Display or save:
		if (output_path == ".") {
			imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
		}
		else {
			imwrite(cv::format("%s/mean.png", output_path.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
		}

/*
		// Display or save the Eigenfaces:
		for (int i = 0; i < cv::min(10, W.cols); i++) {
			std::string msg = cv::format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
			std::cout << msg << std::endl;
			// get eigenvector #i
			cv::Mat ev = W.col(i).clone();
			// Reshape to original size & normalize to [0...255] for imshow.
			cv::Mat grayscale = norm_0_255(ev.reshape(1, height));
			// Show the image & apply a Jet colormap for better sensing.
			cv::Mat cgrayscale;
			applyColorMap(grayscale, cgrayscale, cv::COLORMAP_JET);
			// Display or save:
			if (output_path == ".") {
				imshow(cv::format("eigenface_%d", i), cgrayscale);
			}
			else {
				imwrite(cv::format("%s/eigenface_%d.png", output_path.c_str(), i), norm_0_255(cgrayscale));
			}
		}
*/

/*
		// Display or save the image reconstruction at some predefined steps:
		for (int num_components = cv::min(W.cols, 10); num_components < cv::min(W.cols, 300); num_components += 15) {
			// slice the eigenvectors from the model
			cv::Mat evs = cv::Mat(W, cv::Range::all(), cv::Range(0, num_components));
			cv::Mat projection = cv::LDA::subspaceProject(evs, mean, images[0].reshape(1, 1));
			cv::Mat reconstruction = cv::LDA::subspaceReconstruct(evs, mean, projection);
			// Normalize the result:
			reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
			// Display or save:
			if (output_path == ".") {
				imshow(cv::format("eigenface_reconstruction_%d", num_components), reconstruction);
			}
			else {
				imwrite(cv::format("%s/eigenface_reconstruction_%d.png", output_path.c_str(), num_components), reconstruction);
			}
		}
*/

		// Display if we are not writing to an output folder:
		getchar();
		return 0;
	}

	EigenRecognize::~EigenRecognize() {
	}
}
