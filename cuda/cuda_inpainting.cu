
#include "cuda_inpainting.h"
#include <cuda.h>
#include <vector>
#include <cuda_runtime_api.h>

using namespace std;
using namespace cv;


const int CudaInpainting::RADIUS = 32;
const float CudaInpainting::RANGE_RATIO = 2.0f;

const int CudaInpainting::PATCH_WIDTH = CudaInpainting::RADIUS;
const int CudaInpainting::PATCH_HEIGHT = CudaInpainting::RADIUS;
const int CudaInpainting::NODE_WIDTH = CudaInpainting::PATCH_WIDTH / 2;
const int CudaInpainting::NODE_HEIGHT = CudaInpainting::PATCH_HEIGHT / 2;
const float CudaInpainting::CONST_FULL_MSG = CudaInpainting::PATCH_WIDTH * 
			CudaInpainting::PATCH_HEIGHT * 255 * 255 * 3 / 2.0f;

static void CopyToDevice(void *src, void *dst, uint32_t size) {
	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

static void CopyFromDevice(void *src, void *dst, uint32_t size) {
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

// public functions
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
	CopyToDevice(imageData, deviceImageData, sizeof(float) * 3 * image.cols * image.rows);

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

__global__ void deviceCopyMem(float *src, float *dst, int elem) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x,
	    totalSize = blockDim.x * gridDim.x;
	for(int i = idx; i < elem; i += totalSize) {
		dst[i] = src[i];
	}
}

bool CudaInpainting::Inpainting(int x,int y, int width, int height, int iterTime) {
	Patch patch(x, y, width, height);
	maskPatch = RoundUpArea(patch);

	GenPatches();

	CalculateSSDTable();

	InitNodeTable();
	deviceCopyMem<<<dim3(32,1), dim3(1024,1)>>>(deviceFillMsgTable, deviceMsgTable, nodeWidth * nodeHeight * DIR_COUNT * patchListSize);

	for(int i = 0; i < iterTime; i++) {
		RunIteration();
		deviceCopyMem<<<dim3(32,1), dim3(1024,1)>>>(deviceFillMsgTable, deviceMsgTable, nodeWidth * nodeHeight * DIR_COUNT * patchListSize);
		cout<<"ITERATION "<<i<<endl;
	}

	SelectPatch();

	FillPatch();

	return true;
}

// private functions
CudaInpainting::Patch CudaInpainting::RoundUpArea(Patch p) {
	Patch res;
	res.x = (p.x / NODE_WIDTH) * NODE_WIDTH;
	res.y = (p.y / NODE_HEIGHT) * NODE_HEIGHT;
	res.width = (p.x + p.width +NODE_WIDTH - 1) / NODE_WIDTH * NODE_WIDTH - res.x;
	res.height = (p.y + p.height + NODE_WIDTH - 1) / NODE_HEIGHT * NODE_HEIGHT - res.y;
	return res;
}


bool CudaInpainting::OverlapPatch(Patch& p1, Patch& p2) {
	int mLX = p1.x < p2.x ? p2.x : p1.x,
	    mRX = (p1.x+p1.width) < (p2.x+p2.width) ? (p1.x+p1.width) : (p2.x+p2.width),
	    mTY = p1.y < p2.y ? p2.y : p1.y,
	    mBY = (p1.y+p1.height) < (p2.y+p2.height) ? (p1.y+p1.height) : (p2.y+p2.height);
	return mRX > mLX && mBY > mTY;
}

void CudaInpainting::GenPatches() {
	vector<Patch> tmpPatchList;
	Patch p = maskPatch;
	int hh = image.rows / NODE_HEIGHT,
	    ww = image.cols / NODE_WIDTH;
	float midX = p.x + p.width / 2,
	      midY = p.y + p.height / 2;
	for(int i = 1; i <= hh; i++) {
		for(int j = 1; j <= ww; j++) {
			int cX, cY;
			float fcx = j * NODE_WIDTH, fcy = i * NODE_HEIGHT;
			cY = i * NODE_HEIGHT - NODE_HEIGHT;
			cX = j * NODE_WIDTH - NODE_WIDTH;
			if(!(fabsf(fcx - midX) * 2 / p.width < RANGE_RATIO && fabsf(fcy - midY) * 2 / p.height < RANGE_RATIO))
				continue;
			if(image.rows - cY < PATCH_HEIGHT || image.cols - cX < PATCH_WIDTH)
				continue;
			Patch cur(cX, cY, PATCH_WIDTH, PATCH_HEIGHT);
			if(!OverlapPatch(cur, p))
				tmpPatchList.push_back(cur);
		}
	}
	cudaMalloc((void**)&devicePatchList, sizeof(Patch) * tmpPatchList.size());
	patchList = new Patch[tmpPatchList.size()];
	if(!patchList) {
		cout << "NULL patchList! exit"<< endl;
		exit(-1);
	}
	for(int i = 0; i < tmpPatchList.size(); i++) {
		//CopyToDevice(&patchList[i], devicePatchList + i, sizeof(Patch));
		patchList[i] = tmpPatchList[i];
	}
	patchListSize = tmpPatchList.size();
	CopyToDevice(patchList, devicePatchList, sizeof(Patch) * tmpPatchList.size());
}

__device__ inline int getMsgIdx(int x, int y, CudaInpainting::EDIR dir, int l, int ww, int hh, int len) {
	return y * ww * CudaInpainting::EDIR::DIR_COUNT * len + x * CudaInpainting::EDIR::DIR_COUNT * len +
		dir * len + l;
}

__device__ inline int getEdgeCostIdx(int x, int y, int l, int ww, int hh, int len) {
	return y * ww * len  + x * len + l;
}

__global__ void deviceCalculateSSDTable(float *dImg, int ww, int hh, CudaInpainting::Patch *pl, CudaInpainting::SSDEntry *dSSDTable) {
	int len = gridDim.x;
	const int patchSize = CudaInpainting::PATCH_HEIGHT * CudaInpainting::PATCH_WIDTH;	
	__shared__ float pixels[CudaInpainting::PATCH_HEIGHT][CudaInpainting::PATCH_WIDTH][3];
	for(int i = threadIdx.x; i < patchSize; i += blockDim.x) {
		int yy = i / CudaInpainting::PATCH_WIDTH, xx = i % CudaInpainting::PATCH_WIDTH;
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
					WW = CudaInpainting::PATCH_WIDTH;
					HH = CudaInpainting::NODE_HEIGHT;
					for(int dy = 0; dy < HH; ++dy) {
						for(int dx = 0; dx < WW; ++dx) {
							pxx = px + dx;
							pyy = py + dy;
							float rr = pixels[dy + CudaInpainting::NODE_HEIGHT][dx][0] - dImg[pyy * ww * 3 + pxx * 3],
							      gg = pixels[dy + CudaInpainting::NODE_HEIGHT][dx][1] - dImg[pyy * ww * 3 + pxx * 3 + 1],
							      bb = pixels[dy + CudaInpainting::NODE_HEIGHT][dx][2] - dImg[pyy * ww * 3 + pxx * 3 + 2];
							rr *= rr;
							gg *= gg;
							bb *= bb;
							res = rr + gg + bb;
						}
					}
					break;
				case CudaInpainting::DOWN_UP:
					WW = CudaInpainting::PATCH_WIDTH;
					HH = CudaInpainting::NODE_HEIGHT;
					for(int dy = 0; dy < HH; ++dy) {
						for(int dx = 0; dx < WW; ++dx) {
							pxx = px + dx;
							pyy = py + dy + CudaInpainting::NODE_HEIGHT;
							float rr = pixels[dy][dx][0] - dImg[pyy * ww * 3 + pxx * 3],
							      gg = pixels[dy][dx][1] - dImg[pyy * ww * 3 + pxx * 3 + 1],
							      bb = pixels[dy][dx][2] - dImg[pyy * ww * 3 + pxx * 3 + 2];
							rr *= rr;
							gg *= gg;
							bb *= bb;
							res = rr + gg + bb;
						}
					}
					break;
				case CudaInpainting::RIGHT_LEFT:
					WW = CudaInpainting::NODE_WIDTH;
					HH = CudaInpainting::PATCH_HEIGHT;
					for(int dy = 0; dy < HH; ++dy) {
						for(int dx = 0; dx < WW; ++dx) {
							pxx = px + dx + CudaInpainting::NODE_WIDTH;
							pyy = py + dy;
							float rr = pixels[dy][dx][0] - dImg[pyy * ww * 3 + pxx * 3],
							      gg = pixels[dy][dx][1] - dImg[pyy * ww * 3 + pxx * 3 + 1],
							      bb = pixels[dy][dx][2] - dImg[pyy * ww * 3 + pxx * 3 + 2];
							rr *= rr;
							gg *= gg;
							bb *= bb;
							res = rr + gg + bb;
						}
					}
					break;
				case CudaInpainting::LEFT_RIGHT:
					WW = CudaInpainting::NODE_WIDTH;
					HH = CudaInpainting::PATCH_HEIGHT;
					for(int dy = 0; dy < HH; ++dy) {
						for(int dx = 0; dx < WW; ++dx) {
							pxx = px + dx;
							pyy = py + dy;
							float rr = pixels[dy][dx + CudaInpainting::NODE_WIDTH][0] - dImg[pyy * ww * 3 + pxx * 3],
							      gg = pixels[dy][dx + CudaInpainting::NODE_WIDTH][1] - dImg[pyy * ww * 3 + pxx * 3 + 1],
							      bb = pixels[dy][dx + CudaInpainting::NODE_WIDTH][2] - dImg[pyy * ww * 3 + pxx * 3 + 2];
							rr *= rr;
							gg *= gg;
							bb *= bb;
							res = rr + gg + bb;
						}
					}
					break;
			}
			dSSDTable[blockIdx.x * len + i].data[j] = res;
		}
	}
}

void CudaInpainting::CalculateSSDTable() {
	cudaMalloc((void**)&deviceSSDTable, sizeof(SSDEntry) * patchListSize * patchListSize * EPOS_COUNT);
	if(devicePatchList && deviceSSDTable) {
		cout << "Calculate SSDTable" << endl;
		int len = PATCH_HEIGHT * PATCH_WIDTH;
		if(len > 1024)
			len = 1024;
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
			ww = CudaInpainting::PATCH_WIDTH;
			hh = CudaInpainting::NODE_HEIGHT;
			for(int i = 0; i < hh; ++i) {
				for(int j = 0; j < ww; ++j) {
					float rr = dImg[(p1y + CudaInpainting::NODE_HEIGHT + i) * w * 3 + (p1x + j) * 3] - dImg[(p2y + i) * w * 3 + (p2x + j) * 3],
					      gg = dImg[(p1y + CudaInpainting::NODE_HEIGHT + i) * w * 3 + (p1x + j
) * 3 + 1] - dImg[(p2y + i) * w * 3 + (p2x + j) * 3 + 1],
					      bb = dImg[(p1y + CudaInpainting::NODE_HEIGHT + i) * w * 3 + (p1x + j
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
			ww = CudaInpainting::NODE_WIDTH;
			hh = CudaInpainting::PATCH_HEIGHT;
			for(int i = 0; i < hh; ++i) {
				for(int j = 0; j < ww; ++j) {
					float rr = dImg[(p1y + i) * w * 3 + (p1x + CudaInpainting::NODE_WIDTH + j) * 3] - dImg[(p2y + i) * w * 3 + (p2x + j) * 3],
					      gg = dImg[(p1y + i) * w * 3 + (p1x + CudaInpainting::NODE_WIDTH + j) * 3 + 1] - dImg[(p2y + i) * w * 3 + (p2x + j) * 3 + 1],
					      bb = dImg[(p1y + i) * w * 3 + (p1x + CudaInpainting::NODE_WIDTH + j) * 3 + 2] - dImg[(p2y + i) * w * 3 + (p2x + j) * 3 + 2];
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

__global__ void deviceInitFirst(CudaInpainting::Node* dNodeTable, CudaInpainting::Patch p) {
	int ww = gridDim.x;
	dNodeTable[ww * threadIdx.x + blockIdx.x].x = p.x + blockIdx.x * CudaInpainting::NODE_WIDTH;
	dNodeTable[ww * threadIdx.x + blockIdx.x].y = p.y + threadIdx.x * CudaInpainting::NODE_HEIGHT;
}

__device__ CudaInpainting::Patch::Patch(int ww, int hh) {
	width = ww;
	height = hh;
}

__global__ void deviceInitNodeTable(float *dImg, int w, int h, CudaInpainting::Patch p, CudaInpainting::Node* dNodeTable, float *dMsgTable, float *dEdgeCostTable, CudaInpainting::Patch *dPatchList, int len) {
	int hh = gridDim.y, ww = gridDim.x;
	/*
	dNodeTable[ww * blockIdx.y + blockIdx.x].x = p.x + blockIdx.x * CudaInpainting::NODE_WIDTH;
	dNodeTable[ww * blockIdx.y + blockIdx.x].y = p.y + blockIdx.y * CudaInpainting::NODE_HEIGHT;
	*/

	for(int i = threadIdx.x; i < len; i += blockDim.x * blockDim.y) {
		dMsgTable[getMsgIdx(blockIdx.x, blockIdx.y, CudaInpainting::DIR_UP, i, ww, hh, len)] = CudaInpainting::CONST_FULL_MSG;
		dMsgTable[getMsgIdx(blockIdx.x, blockIdx.y, CudaInpainting::DIR_DOWN, i, ww, hh, len)] = CudaInpainting::CONST_FULL_MSG;
		dMsgTable[getMsgIdx(blockIdx.x, blockIdx.y, CudaInpainting::DIR_LEFT, i, ww, hh, len)] = CudaInpainting::CONST_FULL_MSG;
		dMsgTable[getMsgIdx(blockIdx.x, blockIdx.y, CudaInpainting::DIR_RIGHT, i, ww, hh, len)] = CudaInpainting::CONST_FULL_MSG;

		float val = 0;
		CudaInpainting::Patch curPatch(CudaInpainting::PATCH_WIDTH, CudaInpainting::PATCH_HEIGHT);
		if(((blockIdx.y == 0 || blockIdx.y == hh - 1) && (/*blockIdx.x >= 0 && */blockIdx.x <= ww - 1 )) ||
					((blockIdx.x == 0 || blockIdx.y == ww - 1) && (/*blockIdx.y >= 0 && */blockIdx.y <= hh - 1))) {
			int nodeIdx = ww * blockIdx.y + blockIdx.x;
			if(blockIdx.x == 0) {
				curPatch.x = dNodeTable[nodeIdx].x - CudaInpainting::PATCH_WIDTH;
				curPatch.y = dNodeTable[nodeIdx].y - CudaInpainting::NODE_HEIGHT;
				val += deviceCalculateSSD(dImg, w, h, curPatch, dPatchList[i], CudaInpainting::LEFT_RIGHT);
			} else {
				curPatch.x = dNodeTable[nodeIdx].x;
				curPatch.y = dNodeTable[nodeIdx].y - CudaInpainting::NODE_HEIGHT;
				val += deviceCalculateSSD(dImg, w, h, dPatchList[i], curPatch, CudaInpainting::LEFT_RIGHT);
			}
			if(blockIdx.y == 0) {
				curPatch.x = dNodeTable[nodeIdx].x - CudaInpainting::NODE_WIDTH;
				curPatch.y = dNodeTable[nodeIdx].y - CudaInpainting::PATCH_HEIGHT;
				val += deviceCalculateSSD(dImg, w, h, curPatch, dPatchList[i], CudaInpainting::UP_DOWN);
			} else {
				curPatch.x = dNodeTable[nodeIdx].x - CudaInpainting::NODE_WIDTH;
				curPatch.y = dNodeTable[nodeIdx].y;
				val += deviceCalculateSSD(dImg, w, h, dPatchList[i], curPatch, CudaInpainting::UP_DOWN);
			}
		}
		if(val < 1) {
			val = CudaInpainting::CONST_FULL_MSG;
		}
		dEdgeCostTable[getEdgeCostIdx(blockIdx.x, blockIdx.y, i, ww, hh, len)] = val;
	}
}

void CudaInpainting::InitNodeTable() {
	nodeHeight = maskPatch.height / NODE_HEIGHT + 1;
	nodeWidth = maskPatch.width / NODE_WIDTH + 1;
	int totalElement = nodeWidth * nodeHeight * DIR_COUNT * patchListSize;
	cudaMalloc((void**)&deviceNodeTable, sizeof(Node) * nodeWidth * nodeHeight);
	cudaMalloc((void**)&deviceMsgTable, sizeof(float) * totalElement);
	cudaMalloc((void**)&deviceFillMsgTable, sizeof(float) * totalElement);
	cudaMalloc((void**)deviceEdgeCostTable, sizeof(float) * nodeWidth * nodeHeight * patchListSize);
	if(deviceNodeTable && deviceMsgTable && deviceFillMsgTable && deviceEdgeCostTable) {
		cout << "Initialize the Node Table and Message Table" << endl;
		deviceInitFirst<<<dim3(nodeWidth, 1), dim3(nodeHeight,1)>>>(deviceNodeTable, maskPatch);
		deviceInitNodeTable<<<dim3(nodeWidth, nodeHeight), dim3(512,1)>>>(deviceImageData, imgWidth, imgHeight, maskPatch, deviceNodeTable, deviceMsgTable, deviceEdgeCostTable, devicePatchList, patchListSize);
	}
	nodeTable = new Node[nodeWidth * nodeHeight];
	if(nodeTable) {
		for(int i = 0; i < nodeHeight; ++i) {
			for(int j = 0; j < nodeWidth; ++j) {
				nodeTable[i * nodeWidth + j].x = maskPatch.x + j * NODE_WIDTH;
				nodeTable[i * nodeWidth + j].y = maskPatch.y + i * NODE_HEIGHT;
			}
		}
	}
}

void CudaInpainting::RunIteration() {
}


__global__ void deviceSelectPatch(float *dMsgTable, float *dEdgeCostTable, int *dChoiceList, 
		int ww, int hh,int len) {	
	int xx = blockDim.x * blockIdx.x + threadIdx.x, yy = blockDim.y * blockIdx. y + threadIdx.y;
	if(xx < ww && yy << hh) {
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
		dChoiceList[yy * ww + xx] = maxIdx;
	}
}
void CudaInpainting::SelectPatch() {
	choiceList = new int[nodeWidth * nodeHeight];
	cudaMalloc((void**)&deviceChoiceList, sizeof(float) * nodeWidth * nodeHeight);
	if(choiceList && deviceChoiceList && deviceEdgeCostTable && deviceMsgTable) {
		cout << "Select the Best Patch" << endl;
		deviceSelectPatch<<<dim3((nodeWidth+15)/16, (nodeHeight+15)/16), dim3(16,16)>>>(deviceMsgTable, deviceEdgeCostTable, deviceChoiceList, nodeWidth, nodeHeight, patchListSize);
		CopyFromDevice(deviceChoiceList, choiceList, sizeof(int) * nodeWidth * nodeHeight);
	}
}

void CudaInpainting::PastePatch(Mat& img, Node& n, Patch& p) {
	int xx = n.x - NODE_WIDTH,
	    yy = n.y - NODE_HEIGHT;
	for(int i = 0; i < p.height; ++i) {
		for(int j = 0; j < p.width; ++j) {
			img.at<Vec3f>(yy + i, xx + j) = img.at<Vec3f>(p.y + i, p.x + j);
		}
	}
}

void CudaInpainting::FillPatch() {
	int hh = nodeHeight,
	    ww = nodeWidth;
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


