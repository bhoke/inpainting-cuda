
#include "cuda_inpainting.h"
#include <cuda.h>
#include <vector>
#include <cuda_runtime_api.h>

using namespace std;
using namespace cv;

const float CudaInpainting::RANGE_RATIO = 2.0f;
const int CudaInpainting::NODE_SIZE = 8;
const int CudaInpainting::PATCH_SIZE = NODE_SIZE * 2;

const int CudaInpainting::NODE_SIZE = CudaInpainting::PATCH_SIZE / 2;
const float CudaInpainting::FULL_MSG = CudaInpainting::PATCH_SIZE * 
			CudaInpainting::PATCH_SIZE * 255 * 255 * 3 / 2.0f;

// take one arguement as the input image file
CudaInpainting::CudaInpainting(const char *path) {
	initFlag = false;
	image = imread(path, CV_LOAD_IMAGE_COLOR);
	imageData = nullptr;
	if(!image.data) {
		cout << "Image loading failed" << endl;
		return;
	}
	image.convertTo(image, CV_32FC3);
	
	// copy the image data to float array
	imageData = new float[3 * image.cols * image.rows];
	cudaMalloc((void**)&deviceImageData, sizeof(float) * 3 * image.cols * image.rows);
	if(!imageData) {
		cout << "Memory allocation failed" << endl;
		cudaFree(deviceImageData);
		return;
	}
	for(int y = 0; y < image.rows; ++y) {
		for(int x = 0; x < image.cols; ++x) {
			Vec3f vec = image.at<Vec3f>(y, x);
			imageData[3 * image.cols * y + 3 * x] = vec[0];
			imageData[3 * image.cols * y + 3 * x + 1] = vec[1];
			imageData[3 * image.cols * y + 3 * x + 2] = vec[2];
		}
	}

	// copy the raw data to the GPU
	cudaMemcpy(deviceImageData, imageData, sizeof(float) * 3 * image.cols * image.rows, cudaMemcpyHostToDevice);
	imgWidth = image.cols;
	imgHeight = image.rows;

	// initialize all the ointers
	choiceList = nullptr;
	nodeTable = nullptr;
	patchList = nullptr;
	
	devicePatchList = nullptr;
	deviceSSDTable = nullptr;
	deviceNodeTable = nullptr;
	deviceMsgTable = nullptr;
	deviceFillMsgTable = nullptr;
	deviceEdgeCostTable = nullptr;
	deviceChoiceList = nullptr;
}

// destructor for the CudaInpainting
CudaInpainting::~CudaInpainting() {
	if(imageData) {
		delete imageData;
		cudaFree(deviceImageData);
	}
	if(choiceList)
		delete choiceList;
	if(patchList)
		delete patchList;
	if(nodeTable)
		delete nodeTable;

	if(devicePatchList)
		cudaFree(devicePatchList);
	if(deviceNodeTable)
		cudaFree(deviceNodeTable);
	if(deviceMsgTable)
		cudaFree(deviceMsgTable);
	if(deviceFillMsgTable)
		cudaFree(deviceFillMsgTable);
	if(deviceEdgeCostTable)
		cudaFree(deviceEdgeCostTable);
	if(deviceChoiceList)
		cudaFree(deviceChoiceList);
}

// GPU function to copy memory
__global__ void deviceCopyMem(float *src, float *dst, int elem) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x,
	    totalSize = blockDim.x * gridDim.x;
	for(int i = idx; i < elem; i += totalSize) {
		dst[i] = src[i];
		//printf("%d .  %f <= %f\n", i, dst[i], src[i]);
	}
}

// the main function for the image inpainting procedure
bool CudaInpainting::Inpainting(int x,int y, int width, int height, int iterTime) {
	Patch patch(x, y, width, height);
	// first generate the rounded up patch
	maskPatch = RoundUpArea(patch);

	// generate the candidate patches list
	GenPatches();
	cudaThreadSynchronize();
	if(patchListSize == 0)
		return true;

	// begin to calculate toe SSD table
	CalculateSSDTable();
	cudaThreadSynchronize();

	// before the iterations, we need to initialize the node table and the message
	InitNodeTable();
	deviceCopyMem<<<dim3(32,1), dim3(1024,1)>>>(deviceFillMsgTable, deviceMsgTable, nodeWidth * nodeHeight * DIR_COUNT * patchListSize);
	cudaThreadSynchronize();

	for(int i = 0; i < iterTime; i++) {
		RunIteration(i);
		deviceCopyMem<<<dim3(32,1), dim3(1024,1)>>>(deviceFillMsgTable, deviceMsgTable, nodeWidth * nodeHeight * DIR_COUNT * patchListSize);
		cudaThreadSynchronize();
		cout<<"ITERATION "<<i<<endl;
	}

	// calculate the best patch for each node
	SelectPatch();

	// fill the patch into the original image 
	FillPatch();

	// use a median filter to make the edge between two patches more smooth
	Rect rect(maskPatch.x - NODE_SIZE, maskPatch.y - NODE_SIZE, 
			maskPatch.width + 2 * NODE_SIZE, maskPatch.height + 2 * NODE_SIZE);
	Mat subMat = image(rect);
	Mat matArr[3];
	// split into multiple color channel
	split(subMat, matArr);
	
	for(int i = 0; i < 3; i++) {
		matArr[i].convertTo(matArr[i], CV_8U);
		medianBlur(matArr[i], matArr[i], 3);
		//GaussianBlur(matArr[i], matArr[i], Size(9,9), 0, 0);
		matArr[i].convertTo(matArr[i], CV_32F);
	}
	// merge the color changes into a color image
	merge(matArr, 3, subMat);

	return true;
}

Mat CudaInpainting::GetImage() {
	return image;
}

// private functions
CudaInpainting::Patch CudaInpainting::RoundUpArea(Patch p) {
	Patch res;
	res.x = (p.x / NODE_SIZE) * NODE_SIZE;
	res.y = (p.y / NODE_SIZE) * NODE_SIZE;
	res.width = (p.x + p.width +NODE_SIZE - 1) / NODE_SIZE * NODE_SIZE - res.x;
	res.height = (p.y + p.height + NODE_SIZE - 1) / NODE_SIZE * NODE_SIZE - res.y;
	return res;
}


// to judge if two given patches have overlap region
bool CudaInpainting::hasOverlap(Patch& p1, Patch& p2) {
	int mLX = p1.x < p2.x ? p2.x : p1.x,
	    mRX = (p1.x+p1.width) < (p2.x+p2.width) ? (p1.x+p1.width) : (p2.x+p2.width),
	    mTY = p1.y < p2.y ? p2.y : p1.y,
	    mBY = (p1.y+p1.height) < (p2.y+p2.height) ? (p1.y+p1.height) : (p2.y+p2.height);
	return mRX > mLX && mBY > mTY;
}

// generate the patches list
void CudaInpainting::GenPatches() {
	vector<Patch> tmpPatchList;
	Patch p = maskPatch;
	cout << "x=" << p.x << " y=" << p.y << " width=" << p.width << " height=" << p.height << endl;
	int hh = image.rows / NODE_SIZE,
	    ww = image.cols / NODE_SIZE;
	float midX = p.x + p.width / 2,
	      midY = p.y + p.height / 2;
	for(int i = 1; i <= hh; i++) {
		for(int j = 1; j <= ww; j++) {
			int cX, cY;
			float fcx = j * NODE_SIZE, fcy = i * NODE_SIZE;
			cY = i * NODE_SIZE - NODE_SIZE;
			cX = j * NODE_SIZE - NODE_SIZE;
			if(!(fabsf(fcx - midX) * 2 / p.width < RANGE_RATIO && fabsf(fcy - midY) * 2 / p.height < RANGE_RATIO))
				continue;
			if(image.rows - cY < PATCH_SIZE || image.cols - cX < PATCH_SIZE)
				continue;
			Patch cur(cX, cY, PATCH_SIZE, PATCH_SIZE);
			if(!hasOverlap(cur, p))
				tmpPatchList.push_back(cur);
		}
	}
	patchListSize = tmpPatchList.size();
	if(tmpPatchList.size() == 0)
		return;
	cudaMalloc((void**)&devicePatchList, sizeof(Patch) * tmpPatchList.size());
	patchList = new Patch[tmpPatchList.size()];
	if(!patchList) {
		cout << "NULL patchList! exit"<< endl;
		exit(-1);
	}
	for(int i = 0; i < tmpPatchList.size(); i++) {
		patchList[i] = tmpPatchList[i];
	}
	// copy the generated patches list to the GPU global memory
	cudaMemcpy(devicePatchList, patchList, sizeof(Patch) * tmpPatchList.size(), cudaMemcpyHostToDevice);
	cout << "GenPatch done, " << patchListSize << " patches generated" << endl;
	int idx = 23;
	cout << "Patch => " << idx << " : x=" << patchList[idx].x << " y=" << patchList[idx].y << endl;
	cout << "devicePatchList=" << devicePatchList << endl;
}

// a helper to get the message position in the message table
__device__ inline int getMsgIdx(int x, int y, CudaInpainting::EDIR dir, int l, int ww, int hh, int len) {
	return y * ww * CudaInpainting::EDIR::DIR_COUNT * len + x * CudaInpainting::EDIR::DIR_COUNT * len +
		dir * len + l;
}

// a helper to get the edge cost position in the message table
__device__ inline int getEdgeCostIdx(int x, int y, int l, int ww, int hh, int len) {
	return y * ww * len  + x * len + l;
}

// calculate the SSD table on GPU
__global__ void deviceCalculateSSDTable(float *dImg, int ww, int hh, CudaInpainting::Patch *pl, CudaInpainting::SSDEntry *dSSDTable) {
	int len = gridDim.x;
	const int patchSize = CudaInpainting::PATCH_SIZE * CudaInpainting::PATCH_SIZE;	
	__shared__ float pixels[CudaInpainting::PATCH_SIZE][CudaInpainting::PATCH_SIZE][3];
	for(int i = threadIdx.x; i < patchSize; i += blockDim.x) {
		int yy = i / CudaInpainting::PATCH_SIZE, xx = i % CudaInpainting::PATCH_SIZE;
		int iyy = pl[blockIdx.x].y + yy, ixx = pl[blockIdx.x].x + xx;
		pixels[yy][xx][0] = dImg[iyy * ww * 3 + ixx * 3];
		pixels[yy][xx][1] = dImg[iyy * ww * 3 + ixx * 3 + 1];
		pixels[yy][xx][2] = dImg[iyy * ww * 3 + ixx * 3 + 2];
	}

	__syncthreads();

	for(int i = threadIdx.x; i < len; i += blockDim.x) {
		int px = pl[i].x, py = pl[i].y;
		for(int j = 0; j < CudaInpainting::EPOS_COUNT; j++) {
			float res = 0;
			int WW, HH;
			int pxx, pyy;
			switch(j) {
				case CudaInpainting::UP_DOWN:
					WW = CudaInpainting::PATCH_SIZE;
					HH = CudaInpainting::NODE_SIZE;
					for(int dy = 0; dy < HH; ++dy) {
						for(int dx = 0; dx < WW; ++dx) {
							pxx = px + dx;
							pyy = py + dy;
							float rr = pixels[dy + CudaInpainting::NODE_SIZE][dx][0] - dImg[pyy * ww * 3 + pxx * 3],
							      gg = pixels[dy + CudaInpainting::NODE_SIZE][dx][1] - dImg[pyy * ww * 3 + pxx * 3 + 1],
							      bb = pixels[dy + CudaInpainting::NODE_SIZE][dx][2] - dImg[pyy * ww * 3 + pxx * 3 + 2];
							rr *= rr;
							gg *= gg;
							bb *= bb;
							res += rr + gg + bb;
						}
					}
					break;
				case CudaInpainting::DOWN_UP:
					WW = CudaInpainting::PATCH_SIZE;
					HH = CudaInpainting::NODE_SIZE;
					for(int dy = 0; dy < HH; ++dy) {
						for(int dx = 0; dx < WW; ++dx) {
							pxx = px + dx;
							pyy = py + dy + CudaInpainting::NODE_SIZE;
							float rr = pixels[dy][dx][0] - dImg[pyy * ww * 3 + pxx * 3],
							      gg = pixels[dy][dx][1] - dImg[pyy * ww * 3 + pxx * 3 + 1],
							      bb = pixels[dy][dx][2] - dImg[pyy * ww * 3 + pxx * 3 + 2];
							rr *= rr;
							gg *= gg;
							bb *= bb;
							res += rr + gg + bb;
						}
					}
					break;
				case CudaInpainting::RIGHT_LEFT:
					WW = CudaInpainting::NODE_SIZE;
					HH = CudaInpainting::PATCH_SIZE;
					for(int dy = 0; dy < HH; ++dy) {
						for(int dx = 0; dx < WW; ++dx) {
							pxx = px + dx + CudaInpainting::NODE_SIZE;
							pyy = py + dy;
							float rr = pixels[dy][dx][0] - dImg[pyy * ww * 3 + pxx * 3],
							      gg = pixels[dy][dx][1] - dImg[pyy * ww * 3 + pxx * 3 + 1],
							      bb = pixels[dy][dx][2] - dImg[pyy * ww * 3 + pxx * 3 + 2];
							rr *= rr;
							gg *= gg;
							bb *= bb;
							res += rr + gg + bb;
						}
					}
					break;
				case CudaInpainting::LEFT_RIGHT:
					WW = CudaInpainting::NODE_SIZE;
					HH = CudaInpainting::PATCH_SIZE;
					for(int dy = 0; dy < HH; ++dy) {
						for(int dx = 0; dx < WW; ++dx) {
							pxx = px + dx;
							pyy = py + dy;
							float rr = pixels[dy][dx + CudaInpainting::NODE_SIZE][0] - dImg[pyy * ww * 3 + pxx * 3],
							      gg = pixels[dy][dx + CudaInpainting::NODE_SIZE][1] - dImg[pyy * ww * 3 + pxx * 3 + 1],
							      bb = pixels[dy][dx + CudaInpainting::NODE_SIZE][2] - dImg[pyy * ww * 3 + pxx * 3 + 2];
							rr *= rr;
							gg *= gg;
							bb *= bb;
							res += rr + gg + bb;
						}
					}
					break;
			}
			dSSDTable[blockIdx.x * len + i].data[j] = res;
		}
	}
}

void CudaInpainting::CalculateSSDTable() {
	cudaMalloc((void**)&deviceSSDTable, sizeof(SSDEntry) * patchListSize * patchListSize);
	if(devicePatchList && deviceSSDTable) {
		cout << "Calculate SSDTable" << endl;
		int len = PATCH_SIZE * PATCH_SIZE;
		if(len > 1024)
			len = 1024;
		cout << "CUDA PARAM: " << patchListSize << "=>" << len << endl;
		deviceCalculateSSDTable<<<dim3(patchListSize, 1), dim3(len, 1)>>>(deviceImageData, imgWidth, imgHeight, devicePatchList, deviceSSDTable);
	}
}

__device__ float deviceCalculateSSD(float *dImg, int w, int h, CudaInpainting::Patch p1, CudaInpainting::Patch p2, CudaInpainting::EPOS pos) {
	float res = 0;
	int ww, hh;
	int p1x, p1y, p2x, p2y;
	switch(pos) {
		case CudaInpainting::UP_DOWN:
		case CudaInpainting::DOWN_UP:
			if(pos == CudaInpainting::UP_DOWN) {
				p1x = p1.x;
				p1y = p1.y;
				p2x = p2.x;
				p2y = p2.y;
			} else {
				p1x = p2.x;
				p1y = p2.y;
				p2x = p1.x;
				p2y = p1.y;
			}
			ww = CudaInpainting::PATCH_SIZE;
			hh = CudaInpainting::NODE_SIZE;
			for(int i = 0; i < hh; ++i) {
				for(int j = 0; j < ww; ++j) {
					float rr = dImg[(p1y + CudaInpainting::NODE_SIZE + i) * w * 3 + (p1x + j) * 3] - dImg[(p2y + i) * w * 3 + (p2x + j) * 3],
					      gg = dImg[(p1y + CudaInpainting::NODE_SIZE + i) * w * 3 + (p1x + j
) * 3 + 1] - dImg[(p2y + i) * w * 3 + (p2x + j) * 3 + 1],
					      bb = dImg[(p1y + CudaInpainting::NODE_SIZE + i) * w * 3 + (p1x + j
) * 3 + 2] - dImg[(p2y + i) * w * 3 + (p2x + j) * 3 + 2];
					rr *= rr;
					gg *= gg;
					bb *= bb;
					res += rr + gg + bb;
				}
			}
			break;
		case CudaInpainting::LEFT_RIGHT:
		case CudaInpainting::RIGHT_LEFT:
			if(pos == CudaInpainting::LEFT_RIGHT) {
				p1x = p1.x;
				p1y = p1.y;
				p2x = p2.x;
				p2y = p2.y;
			} else {
				p1x = p2.x;
				p1y = p2.y;
				p2x = p1.x;
				p2y = p1.y;
			}
			ww = CudaInpainting::NODE_SIZE;
			hh = CudaInpainting::PATCH_SIZE;
			for(int i = 0; i < hh; ++i) {
				for(int j = 0; j < ww; ++j) {
					float rr = dImg[(p1y + i) * w * 3 + (p1x + CudaInpainting::NODE_SIZE + j) * 3] - dImg[(p2y + i) * w * 3 + (p2x + j) * 3],
					      gg = dImg[(p1y + i) * w * 3 + (p1x + CudaInpainting::NODE_SIZE + j) * 3 + 1] - dImg[(p2y + i) * w * 3 + (p2x + j) * 3 + 1],
					      bb = dImg[(p1y + i) * w * 3 + (p1x + CudaInpainting::NODE_SIZE + j) * 3 + 2] - dImg[(p2y + i) * w * 3 + (p2x + j) * 3 + 2];
					rr *= rr;
					gg *= gg;
					bb *= bb;
					res += rr + gg + bb;
				}
			}
			break;
	}
	return res;
}

// initialize the coordinates of node in table
__global__ void deviceInitFirst(CudaInpainting::Node* dNodeTable, CudaInpainting::Patch p) {
	int ww = gridDim.x;
	dNodeTable[ww * threadIdx.x + blockIdx.x].x = p.x + blockIdx.x * CudaInpainting::NODE_SIZE;
	dNodeTable[ww * threadIdx.x + blockIdx.x].y = p.y + threadIdx.x * CudaInpainting::NODE_SIZE;
}

// the constructor of patch on GPU
__device__ CudaInpainting::Patch::Patch(int ww, int hh) {
	width = ww;
	height = hh;
}

// the initialize node table on GPU
__global__ void deviceInitNodeTable(float *dImg, int w, int h, CudaInpainting::Patch p, CudaInpainting::Node* dNodeTable, float *dMsgTable, float *dEdgeCostTable, CudaInpainting::Patch *dPatchList, int len) {
	int hh = gridDim.y, ww = gridDim.x;

	for(int i = threadIdx.x; i < len; i += blockDim.x * blockDim.y) {
		// initialize the message with the very large values
		dMsgTable[getMsgIdx(blockIdx.x, blockIdx.y, CudaInpainting::DIR_UP, i, ww, hh, len)] = CudaInpainting::FULL_MSG;
		dMsgTable[getMsgIdx(blockIdx.x, blockIdx.y, CudaInpainting::DIR_DOWN, i, ww, hh, len)] = CudaInpainting::FULL_MSG;
		dMsgTable[getMsgIdx(blockIdx.x, blockIdx.y, CudaInpainting::DIR_LEFT, i, ww, hh, len)] = CudaInpainting::FULL_MSG;
		dMsgTable[getMsgIdx(blockIdx.x, blockIdx.y, CudaInpainting::DIR_RIGHT, i, ww, hh, len)] = CudaInpainting::FULL_MSG;

		// initialize the edge cost 
		float val = 0;
		CudaInpainting::Patch curPatch(CudaInpainting::PATCH_SIZE, CudaInpainting::PATCH_SIZE);
		
		// to judge if the current node is on the edge of the node table
		if(((blockIdx.y == 0 || blockIdx.y == hh - 1) && (/*blockIdx.x >= 0 && */blockIdx.x <= ww - 1 )) ||
					((blockIdx.x == 0 || blockIdx.x == ww - 1) && (/*blockIdx.y >= 0 && */blockIdx.y <= hh - 1))) {
			int nodeIdx = ww * blockIdx.y + blockIdx.x;
			int valCount = 0;
			if(blockIdx.x == 0) {
				curPatch.x = dNodeTable[nodeIdx].x - CudaInpainting::PATCH_SIZE;
				curPatch.y = dNodeTable[nodeIdx].y - CudaInpainting::NODE_SIZE;
				val += deviceCalculateSSD(dImg, w, h, curPatch, dPatchList[i], CudaInpainting::LEFT_RIGHT);
				++valCount;
			} else {
				curPatch.x = dNodeTable[nodeIdx].x;
				curPatch.y = dNodeTable[nodeIdx].y - CudaInpainting::NODE_SIZE;
				val += deviceCalculateSSD(dImg, w, h, dPatchList[i], curPatch, CudaInpainting::LEFT_RIGHT);
				++valCount;
			}
			if(blockIdx.y == 0) {
				curPatch.x = dNodeTable[nodeIdx].x - CudaInpainting::NODE_SIZE;
				curPatch.y = dNodeTable[nodeIdx].y - CudaInpainting::PATCH_SIZE;
				val += deviceCalculateSSD(dImg, w, h, curPatch, dPatchList[i], CudaInpainting::UP_DOWN);
				++valCount;
			} else {
				curPatch.x = dNodeTable[nodeIdx].x - CudaInpainting::NODE_SIZE;
				curPatch.y = dNodeTable[nodeIdx].y;
				val += deviceCalculateSSD(dImg, w, h, dPatchList[i], curPatch, CudaInpainting::UP_DOWN);
				++valCount;
			}
			val /= valCount;
		}
		if(val < 0.5f)
			val = CudaInpainting::FULL_MSG;
		dEdgeCostTable[getEdgeCostIdx(blockIdx.x, blockIdx.y, i, ww, hh, len)] = val;
	}
}


// wrap for the initialization of the node table
void CudaInpainting::InitNodeTable() {
	nodeHeight = maskPatch.height / NODE_SIZE + 1;
	nodeWidth = maskPatch.width / NODE_SIZE + 1;
	cout << "NodeTable => width=" << nodeWidth << " height=" << nodeHeight << endl;
	int totalElement = nodeWidth * nodeHeight * DIR_COUNT * patchListSize;
	cout << cudaGetErrorString(cudaMalloc((void**)&deviceNodeTable, sizeof(Node) * nodeWidth * nodeHeight)) << endl;
	cout << cudaGetErrorString(cudaMalloc((void**)&deviceMsgTable, sizeof(float) * totalElement)) << endl;
	cout << cudaGetErrorString(cudaMalloc((void**)&deviceFillMsgTable, sizeof(float) * totalElement)) << endl;
	cout << cudaGetErrorString(cudaMalloc((void**)&deviceEdgeCostTable, sizeof(float) * nodeWidth * nodeHeight * patchListSize)) << endl;
	if(deviceNodeTable && deviceMsgTable && deviceFillMsgTable && deviceEdgeCostTable) {
		cout << "Initialize the Node Table and Message Table" << endl;
		// initialize node table
		deviceInitFirst<<<dim3(nodeWidth, 1), dim3(nodeHeight,1)>>>(deviceNodeTable, maskPatch);

		// initialize the messages in the node table
		deviceInitNodeTable<<<dim3(nodeWidth, nodeHeight), dim3(512,1)>>>(deviceImageData, imgWidth, imgHeight, maskPatch, deviceNodeTable, deviceFillMsgTable, deviceEdgeCostTable, devicePatchList, patchListSize);
	} else {
		cout << " Failed to cudaMalloc" << endl;
	}

	// initialize the node table on CPU
	nodeTable = new Node[nodeWidth * nodeHeight];
	if(nodeTable) {
		for(int i = 0; i < nodeHeight; ++i) {
			for(int j = 0; j < nodeWidth; ++j) {
				nodeTable[i * nodeWidth + j].x = maskPatch.x + j * NODE_SIZE;
				nodeTable[i * nodeWidth + j].y = maskPatch.y + i * NODE_SIZE;
			}
		}
	}
}

// the iteration function which will be run on GPU
__global__ void deviceIteration(CudaInpainting::SSDEntry *dSSDTable, float *dEdgeCostTable, CudaInpainting::Patch *dPatchList, int len, float *dMsgTable, float *dFillMsgTable, int times) {
	int hh = gridDim.y, ww = gridDim.x, i = blockIdx.y, j = blockIdx.x;
	float aroundMsg, msgCount, matchFactor;
	float msgFactor = 0.8f;
	matchFactor = 1.2f;
	msgCount = msgFactor * 3 + matchFactor + 1;
	// each thread handle one patch in all directions
	for(int ll = threadIdx.x; ll < len; ll += blockDim.x) {
		// use register to optimize the running time
		float up_val, down_val, left_val, right_val;
		// up
		if(i != 0) {
			int targetIdx = getMsgIdx(j, i, CudaInpainting::DIR_UP, ll, ww, hh, len);
			up_val = dFillMsgTable[targetIdx];
		}
		// down
		if(i != hh - 1) {
			int targetIdx = getMsgIdx(j, i, CudaInpainting::DIR_DOWN, ll, ww, hh, len);
			down_val = dFillMsgTable[targetIdx];
		}
		// left
		if(j != 0) {
			int targetIdx = getMsgIdx(j, i, CudaInpainting::DIR_LEFT, ll, ww, hh, len);
			left_val = dFillMsgTable[targetIdx];
		}
		// right
		if(j != ww - 1) {
			int targetIdx = getMsgIdx(j, i, CudaInpainting::DIR_RIGHT, ll, ww, hh, len);
			right_val = dFillMsgTable[targetIdx];
		}

		for(int k = 0; k < len; ++k) {
			float distDiff = 1;
			aroundMsg = 0;
			if(i != 0)
				aroundMsg += dMsgTable[getMsgIdx(j, i - 1, CudaInpainting::DIR_DOWN, k, ww, hh, len)];
			else if(i != hh - 1)
				aroundMsg += dMsgTable[getMsgIdx(j, i + 1, CudaInpainting::DIR_UP, k, ww, hh, len)];
			else if(j != 0)
				aroundMsg += dMsgTable[getMsgIdx(j - 1, i, CudaInpainting::DIR_RIGHT, k, ww, hh, len)];
			else if(j != ww - 1)
				aroundMsg += dMsgTable[getMsgIdx(j + 1, i, CudaInpainting::DIR_LEFT, k, ww, hh, len)];
			else
				aroundMsg += CudaInpainting::FULL_MSG;
			aroundMsg *= msgFactor;
			float edgeVal = dEdgeCostTable[getEdgeCostIdx(j, i, k, ww, hh, len)];
			aroundMsg += edgeVal;
			float val, oldVal;
			// up
			if(i != 0) {
				val = aroundMsg + dSSDTable[k * len + ll].data[CudaInpainting::DOWN_UP] * matchFactor * distDiff;
				val -= dMsgTable[getMsgIdx(j, i - 1, CudaInpainting::DIR_DOWN, k, ww, hh, len)] * msgFactor;
				int targetIdx = getMsgIdx(j, i, CudaInpainting::DIR_UP, ll, ww, hh, len);
				val /= msgCount;
				oldVal = up_val;
				//printf("(%d,%d,-%d) => val=%f\n", j, i, ll, val);
				if(val < oldVal) {
					up_val = val;
				}
			}
			// down
			if(i != hh - 1) {
				val = aroundMsg + dSSDTable[k * len + ll].data[CudaInpainting::UP_DOWN] * matchFactor * distDiff;
				val -= dMsgTable[getMsgIdx(j, i + 1, CudaInpainting::DIR_UP, k, ww, hh, len)] * msgFactor;
				val /= msgCount;
				int targetIdx = getMsgIdx(j, i, CudaInpainting::DIR_DOWN, ll, ww, hh, len);
				oldVal = down_val;
				if(val < oldVal) {
					down_val = val;
				}
			}
			// left
			if(j != 0) {
				val = aroundMsg + dSSDTable[k * len + ll].data[CudaInpainting::RIGHT_LEFT] * matchFactor * distDiff;
				val -= dMsgTable[getMsgIdx(j - 1, i, CudaInpainting::DIR_RIGHT, k, ww, hh, len)] * msgFactor;
				val /= msgCount;
				int targetIdx = getMsgIdx(j, i, CudaInpainting::DIR_LEFT, ll, ww, hh, len);
				oldVal = left_val;
				if(val < oldVal) {
					left_val = val;
				}
			}
			// right
			if(j != ww - 1) {
				val = aroundMsg + dSSDTable[k * len + ll].data[CudaInpainting::LEFT_RIGHT] * matchFactor;
				val -= dMsgTable[getMsgIdx(j + 1, i, CudaInpainting::DIR_LEFT, k, ww, hh, len)] * msgFactor;
				//printf("(%d,%d,-%d) => val=%f oldVal=%f\n", j, i, ll, val, oldVal);
				val /= msgCount;
				int targetIdx = getMsgIdx(j, i, CudaInpainting::DIR_RIGHT, ll, ww, hh, len);
				oldVal = right_val;
				if(val < oldVal) {
					right_val = val;
				}
			}
		}
		// up
		if(i != 0) {
			int targetIdx = getMsgIdx(j, i, CudaInpainting::DIR_UP, ll, ww, hh, len);
			dFillMsgTable[targetIdx] = up_val;
		}
		// down
		if(i != hh - 1) {
			int targetIdx = getMsgIdx(j, i, CudaInpainting::DIR_DOWN, ll, ww, hh, len);
			dFillMsgTable[targetIdx] = down_val;
		}
		// left
		if(j != 0) {
			int targetIdx = getMsgIdx(j, i, CudaInpainting::DIR_LEFT, ll, ww, hh, len);
			dFillMsgTable[targetIdx] = left_val;
		}
		// right
		if(j != ww - 1) {
			int targetIdx = getMsgIdx(j, i, CudaInpainting::DIR_RIGHT, ll, ww, hh, len);
			dFillMsgTable[targetIdx] = right_val;
		}
	}
}

// the wrap function for iteration in Belief Propagation
void CudaInpainting::RunIteration(int times) {
	if(deviceMsgTable && deviceFillMsgTable && deviceSSDTable && deviceEdgeCostTable) {
		cout << "Run Iteration" << endl;
		int lim = 1024;
		if(patchListSize < lim) {
			lim = patchListSize;
		}
		deviceIteration<<<dim3(nodeWidth, nodeHeight),dim3(lim, 1)>>>(deviceSSDTable, deviceEdgeCostTable,devicePatchList, patchListSize, deviceMsgTable, deviceFillMsgTable, times);
	}
}


// select the best patch for each node on GPU
__global__ void deviceSelectPatch(float *dMsgTable, float *dEdgeCostTable, int *dChoiceList, 
		int ww, int hh,int len) {	
	int xx = blockDim.x * blockIdx.x + threadIdx.x, yy = blockDim.y * blockIdx. y + threadIdx.y;
	if(xx < ww && yy < hh) {
		float maxB = 0;
		int maxIdx = -1;
		for(int k = 0; k < len; ++k) {
			float bl = -dEdgeCostTable[getEdgeCostIdx(xx, yy, k, ww, hh, len)];
			float val;
			if(yy - 1 >= 0) {
				val = dMsgTable[getMsgIdx(xx, yy - 1, CudaInpainting::DIR_DOWN, k, ww, hh, len)];
				if(val > 0)
					bl -= val;
			}
			if(yy + 1 < hh) {
				val = dMsgTable[getMsgIdx(xx, yy + 1, CudaInpainting::DIR_UP, k, ww, hh, len)];
				if(val > 0)
					bl -= val;
			}
			if(xx - 1 >= 0) {
				val = dMsgTable[getMsgIdx(xx - 1, yy, CudaInpainting::DIR_RIGHT, k, ww, hh, len)];
				if(val > 0)
					bl -= val;
			}
			if(xx + 1 < ww) {
				val = dMsgTable[getMsgIdx(xx + 1, yy, CudaInpainting::DIR_LEFT, k, ww, hh, len)];
				if(val > 0)
					bl -= val;
			}
			if(bl > maxB || maxIdx < 0) {
				maxB = bl;
				maxIdx = k;
			}
		}
		//printf("(%d,%d) (%d,%d) => max %f %d\n", ww, hh, xx, yy, maxB, maxIdx);
		dChoiceList[yy * ww + xx] = maxIdx;
	}
}

// the wraper for selecting best patch on GPU
void CudaInpainting::SelectPatch() {
	choiceList = new int[nodeWidth * nodeHeight];
	cudaMalloc((void**)&deviceChoiceList, sizeof(float) * nodeWidth * nodeHeight);
	if(choiceList && deviceChoiceList && deviceEdgeCostTable && deviceMsgTable) {
		cout << "Select the Best Patch" << endl;
		deviceSelectPatch<<<dim3((nodeWidth+15)/16, (nodeHeight+15)/16), dim3(16,16)>>>(deviceMsgTable, deviceEdgeCostTable, deviceChoiceList, nodeWidth, nodeHeight, patchListSize);
		cudaMemcpy(choiceList, deviceChoiceList, sizeof(int) * nodeWidth * nodeHeight, cudaMemcpyDeviceToHost);
	}
}

// the helper to paste the best patch to the specified node
void CudaInpainting::PastePatch(Mat& img, Node& n, Patch& p) {
	int xx = n.x - NODE_SIZE / 2,
	    yy = n.y - NODE_SIZE / 2;
	for(int i = 0; i < p.height / 2; ++i) {
		for(int j = 0; j < p.width / 2; ++j) {
			img.at<Vec3f>(yy + i, xx + j) = img.at<Vec3f>(p.y + NODE_SIZE/2 + i, p.x + NODE_SIZE/2 + j);
		}
	}
}

// paste best patch for all node
void CudaInpainting::FillPatch() {
	int hh = nodeHeight,
	    ww = nodeWidth;
	// just print the result
	for(int i = 0; i < hh; ++i) {
		for(int j = 0; j < ww; ++j) {
			cout<<choiceList[j + i * ww]<<" ";
		}
		cout<<endl;
	}
	for(int i = 0; i < hh; ++i) {
		for(int j = 0; j < ww; ++j) {
			int label = choiceList[j + i * ww];
			if(label >= 0) {
				PastePatch(image, nodeTable[j + i * ww], patchList[label]);
			}
		}
	}
}



