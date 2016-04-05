#ifndef _DEFINES_H_
#define _DEFINES_H_

#pragma once

#ifdef _OPENMP
	#include <omp.h>
#endif

enum APPLICATION_OPTIONS_ENUM {
	not_choisen = 0, 
	/* CAPTURE_TYPE_ENUM */
	input_library, 
	modern_capture, 
	video_import,
	/* FACE_RECOGNIZE_ENUM */
	eigen_test,
	haar_cascade_test,
	sr_recognize_test
};

#include "logic\capture\CamCaptureLib.h"
#include "logic\capture\CamCaptureModern.h"
#include "logic\capture\VideoCapture.h"
#include "logic\face_recognition\EigenRecognize.h"
#include "logic\face_recognition\CascadeClassifier.h"
#include "logic\face_recognition\SRClassifier.h"

#endif
