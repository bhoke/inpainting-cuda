#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cuda_inpainting.h"


using namespace std;
using namespace cv;

#define RADIUS	(16)
#define RANGE_RATIO	(2.0f)

int main(int argc, char **argv) {
	if(argc != 8) {
		cout<<"Usage: "<<argv[0]<<" input x y w h output iter_time"<<endl;
		return 0;
	}
	auto start = std::chrono::system_clock::now();
	// construct a CudaInpainting class preparing for the coming inpainting parameters
	CudaInpainting ci(argv[1]);

	// parse the arguments
	char *input = argv[1],
	     *output = argv[6];
	int maskX = atoi(argv[2]),
	    maskY = atoi(argv[3]),
	    maskW = atoi(argv[4]),
	    maskH = atoi(argv[5]),
	    iterTime = atoi(argv[7]);
	cout << "Begin to Inpainting" << endl;

	// invoke the inpainting function 
	
	ci.Inpainting(maskX, maskY, maskW, maskH, iterTime);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_seconds).count();
	cout << "Execution Time(GPU): " << millis << " milliseconds" << endl;
	cout << "Begin to write the image" << endl;

	// write the output image to the output file
	imwrite(output, ci.GetImage());
	cout << "Done" << endl;
	
	return 0;
}
