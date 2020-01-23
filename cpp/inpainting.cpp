#include <iostream>
#include <string>
#include <vector>
#include <cmath>
//#include <unistd.h>
#include "Patch.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace std;
using namespace cv;

#define RADIUS	(16)

// how large neighborhood region should we extract the patches
#define RANGE_RATIO	(2.0f)

const int PATCH_WIDTH = RADIUS;
const int PATCH_HEIGHT = RADIUS;
const int NODE_WIDTH = PATCH_WIDTH / 2;
const int NODE_HEIGHT = PATCH_HEIGHT / 2;

// used to initialize the message which will be passed in the iteration
const float FULL_MSG = PATCH_HEIGHT * PATCH_WIDTH * 255 * 255 * 3 / 2;

// the patch class to store the information of each patch
class patch {
public:
	int x; // width
	int y; // height
	int width;
	int height;
	patch() : x(0), y(0), width(0), height(0) {}
	patch(int xx, int yy, int ww, int hh) : x(xx), y(yy), width(ww), height(hh) {}
};

// the enumeration number indicate the relative position between two patches
enum EPOS {
	UP_DOWN = 0,
	DOWN_UP,
	LEFT_RIGHT,
	RIGHT_LEFT,
	EPOS_COUNT,
};

/*
 * roud up the patch so that we have large edge which could be used
 * to calculate the edge pontential
 */
patch roundUpArea(patch p) {
	patch res;
	res.x = (p.x / NODE_WIDTH) * NODE_WIDTH;
	res.y = (p.y / NODE_HEIGHT) * NODE_HEIGHT;
	res.width = (p.x + p.width + NODE_WIDTH - 1) / NODE_WIDTH * NODE_WIDTH - res.x;
	res.height = (p.y + p.height + NODE_WIDTH - 1) / NODE_HEIGHT * NODE_HEIGHT - res.y;
	return res;
}


void drawMask(int event, int x, int y, int flags, void* param) {
		if (flags == EVENT_FLAG_LBUTTON)
		{
			Mat &img = *((Mat*)(param)); // 1st cast it back, then deref
			circle(img, Point(x, y), 7, Scalar::all(0), FILLED);
		}
}

/*
 * to judge if two patches have overlap region
 */
bool overlapPatch(patch &p1, patch &p2) {
	int mLX = p1.x < p2.x ? p2.x : p1.x,
		mRX = (p1.x + p1.width) < (p2.x + p2.width) ? (p1.x + p1.width) : (p2.x + p2.width),
		mTY = p1.y < p2.y ? p2.y : p1.y,
		mBY = (p1.y + p1.height) < (p2.y + p2.height) ? (p1.y + p1.height) : (p2.y + p2.height);
	return mRX > mLX && mBY > mTY;
}

/*
 * generate the patch list
 * note that we don't use the patches which is too far away from
 * the target region
 */
vector<patch> genPatches(Mat &img, patch p) {
	vector<patch> res;
	int hh = img.rows / NODE_HEIGHT,
		ww = img.cols / NODE_WIDTH;
	// the middle point of the target region
	float midX = p.x + p.width / 2,
		midY = p.y + p.height / 2;
	for (int i = 1; i <= hh; i++) {
		for (int j = 1; j <= ww; j++) {
			int cX, cY;
			float fcx = j * NODE_WIDTH, fcy = i * NODE_HEIGHT;
			cY = i * NODE_HEIGHT - NODE_HEIGHT;
			cX = j * NODE_WIDTH - NODE_WIDTH;
			// skip the too far away patches
			if (!(fabsf(fcx - midX) * 2 / p.width < RANGE_RATIO && fabsf(fcy - midY) * 2 / p.height < RANGE_RATIO))
				continue;
			if (img.rows - cY < PATCH_HEIGHT || img.cols - cX < PATCH_WIDTH)
				continue;
			patch cur(cX, cY, PATCH_WIDTH, PATCH_HEIGHT);
			// add patch into patch list if has no overlap region with target region
			if (!overlapPatch(cur, p))
				res.push_back(cur);
		}
	}
	return res;
}

/*
 * for simplicity, we only implement two kinds of relative position,
 * since DOWN_UP could be transformed to UP_DOWN
 * and RIGHT_LEFT could be transformed to LEFT_RIGHT
 */
float calculateSSD(Mat &img, patch &p1, patch &p2, EPOS pos) {
	float res = 0;
	int ww, hh;
	switch (pos) {
	case UP_DOWN:
		ww = PATCH_WIDTH;
		hh = NODE_HEIGHT;
		for (int i = 0; i < hh; i++) {
			for (int j = 0; j < ww; j++) {
				Vec3f pv1 = img.at<Vec3f>(p1.y + NODE_HEIGHT + i, p1.x + j),
					pv2 = img.at<Vec3f>(p2.y + i, p2.x + j);
				for (int k = 0; k < 3; k++) {
					float m = pv1[k] - pv2[k];
					res += m * m;
				}
			}
		}
		break;
	case LEFT_RIGHT:
		ww = NODE_WIDTH;
		hh = PATCH_HEIGHT;
		for (int i = 0; i < hh; i++) {
			for (int j = 0; j < ww; j++) {
				Vec3f pv1 = img.at<Vec3f>(p1.y + i, p1.x + NODE_WIDTH + j),
					pv2 = img.at<Vec3f>(p2.y + i, p2.x + j);
				for (int k = 0; k < 3; k++) {
					float m = pv1[k] - pv2[k];
					res += m * m;
				}
			}
		}
		break;
	default:
		cout << "FATAL ERROR" << endl;
		exit(-1);
	}
	return res;
}

/*
 * to calculate the whole SSD table
 * SSD means the Sum of Squared Difference
 */
vector<vector<vector<float> > > calculateSSDTable(Mat &img, vector<patch> &patchList) {
	vector<vector<vector<float> > > res;
	int len = patchList.size();
	res.resize(len);
	for (int i = 0; i < len; i++) {
		res[i].resize(len);
	}
	for (int i = 0; i < len; i++) {
		for (int j = i; j < len; j++) {
				res[i][j].resize(4); // Number of directions
				res[i][j][UP_DOWN] = calculateSSD(img, patchList[i], patchList[j], UP_DOWN);
				res[i][j][DOWN_UP] = calculateSSD(img, patchList[j], patchList[i], UP_DOWN);
				res[i][j][LEFT_RIGHT] = calculateSSD(img, patchList[i], patchList[j], LEFT_RIGHT);
				res[i][j][RIGHT_LEFT] = calculateSSD(img, patchList[j], patchList[i], LEFT_RIGHT);
		}
	}
	return res;
}

/*
 * a helper to convenient access the data in the SSD table
 */
float getSSD(vector<vector<vector<float> > > &ssdTable, int p1, int p2, EPOS pos) {
	if (p1 > p2) {
		switch (pos) {
		case UP_DOWN:
			pos = DOWN_UP;
			break;
		case DOWN_UP:
			pos = UP_DOWN;
			break;
		case LEFT_RIGHT:
			pos = RIGHT_LEFT;
			break;
		case RIGHT_LEFT:
			pos = LEFT_RIGHT;
			break;
		default:
			cout << "ERROR" << endl;
			exit(-1);
		}
		return ssdTable[p2][p1][pos];
	}
	return ssdTable[p1][p2][pos];
}

enum EDIR {
	DIR_UP = 0,
	DIR_DOWN,
	DIR_LEFT,
	DIR_RIGHT,
	DIR_COUNT,
};

// node class represent one node in the MRF model
class node {
public:
	vector<vector<float> > msg;		// message vector for all directions and all patches
	vector<vector<float> > newMsg;		// use to iteration
	vector<float> edge_cost;		// the cost on the edge, internal node will have zero in this field
	int label;				// the best patch to be selected for this node
	int x;					// Node's X coordinate
	int y;					// Node's Y coordinate
};


/*
 * to initialize the node table
 * include message and edge cost
 */
void initNodeTable(Mat &img, vector<vector<node> > &nodeTable, patch &p, vector<patch> &patchList) {
	int hh = p.height / NODE_HEIGHT + 1,
		ww = p.width / NODE_WIDTH + 1,
		len = patchList.size();
	nodeTable.resize(hh);
	cout << "hh=" << hh << " ww=" << ww << endl;
	for (int i = 0; i < hh; i++) {
		nodeTable[i].resize(ww);
		for (int j = 0; j < ww; j++) {
			nodeTable[i][j].msg.resize(DIR_COUNT);
			for (int k = 0; k < DIR_COUNT; k++) {
				nodeTable[i][j].msg[k].resize(len);
				for (int l = 0; l < len; l++) {
					//nodeTable[i][j].msg[k][l] = -1;
					nodeTable[i][j].msg[k][l] = FULL_MSG;
				}
			}
			nodeTable[i][j].label = -1;
			nodeTable[i][j].x = p.x + j * NODE_WIDTH;
			nodeTable[i][j].y = p.y + i * NODE_HEIGHT;
			nodeTable[i][j].edge_cost.resize(len);
			for (int k = 0; k < len; k++) {
				float val = 0;
				patch curPatch(0, 0, PATCH_WIDTH, PATCH_HEIGHT);
				/*
				 * only the node on the edge need to calculate the SSD
				 */
				if (((i == 0 || i == hh - 1) && (j >= 0 && j <= ww - 1)) ||
					((j == 0 || j == ww - 1) && (i >= 0 && i <= hh - 1))) {
					if (j == 0) {
						curPatch.x = nodeTable[i][j].x - PATCH_WIDTH;
						curPatch.y = nodeTable[i][j].y - NODE_HEIGHT;
						val += calculateSSD(img, curPatch, patchList[k], LEFT_RIGHT);
					}
					else {
						curPatch.x = nodeTable[i][j].x;
						curPatch.y = nodeTable[i][j].y - NODE_HEIGHT;
						val += calculateSSD(img, patchList[k], curPatch, LEFT_RIGHT);
					}
					if (i == 0) {
						curPatch.x = nodeTable[i][j].x - NODE_WIDTH;
						curPatch.y = nodeTable[i][j].y - PATCH_HEIGHT;
						val += calculateSSD(img, curPatch, patchList[k], UP_DOWN);
					}
					else {
						curPatch.x = nodeTable[i][j].x - NODE_WIDTH;
						curPatch.y = nodeTable[i][j].y;
						val += calculateSSD(img, patchList[k], curPatch, UP_DOWN);
					}
				}
				nodeTable[i][j].edge_cost[k] = val;
				if (val < 1)
					nodeTable[i][j].edge_cost[k] = FULL_MSG;
			}
			// copy the initialized message to iteration used message vector
			nodeTable[i][j].newMsg = nodeTable[i][j].msg;
		}
	}
}

/*
 * this function is iteration for the belief propagation
 */
void propagateMsg(vector<vector<node> > &nodeTable, vector<vector<vector<float> > > &ssdTable) {
	int hh = nodeTable.size(),
		ww = nodeTable[0].size(),
		len = ssdTable.size();
	for (int i = 0; i < hh; i++) {
		for (int j = 0; j < ww; j++) {
			for (int k = 0; k < len; k++) {
				float aroundMsg = 0, msgCount, matchFactor;
				float msgFactor = 0.6;				// how important is messages from the adjacent node
				matchFactor = 1.2;				// how important is the SSD between the adjacent patches
				msgCount = msgFactor * 3 + matchFactor;
				if (i != 0)
					aroundMsg += nodeTable[i - 1][j].msg[DIR_DOWN][k];
				else if (i != hh - 1)
					aroundMsg += nodeTable[i + 1][j].msg[DIR_UP][k];
				else
					aroundMsg += FULL_MSG;
				if (j != 0)
					aroundMsg += nodeTable[i][j - 1].msg[DIR_RIGHT][k];
				else if (j != ww - 1)
					aroundMsg += nodeTable[i][j + 1].msg[DIR_LEFT][k];
				else
					aroundMsg += FULL_MSG;

				aroundMsg *= msgFactor;
				aroundMsg += nodeTable[i][j].edge_cost[k];
				for (int ll = 0; ll < len; ll++) {
					float val, oldVal;
					/*
					 * in this loop, go over all the patches to update the message
					 * for each patch in all directions
					 */
					 // up
					if (i != 0) {
						val = aroundMsg + getSSD(ssdTable, k, ll, DOWN_UP) * matchFactor;
						val -= nodeTable[i - 1][j].msg[DIR_DOWN][k] * msgFactor;
						val /= msgCount;
						oldVal = nodeTable[i][j].newMsg[DIR_UP][ll];
						if (val < oldVal) {
							nodeTable[i][j].newMsg[DIR_UP][ll] = val;
						}
						else {
							nodeTable[i][j].newMsg[DIR_UP][ll] = oldVal;
						}
					}
					// down
					if (i != hh - 1) {
						val = aroundMsg + getSSD(ssdTable, k, ll, UP_DOWN) * matchFactor;
						val -= nodeTable[i + 1][j].msg[DIR_UP][k] * msgFactor;
						val /= msgCount;
						oldVal = nodeTable[i][j].newMsg[DIR_DOWN][ll];
						if (val < oldVal) {
							nodeTable[i][j].newMsg[DIR_DOWN][ll] = val;
						}
						else {
							nodeTable[i][j].newMsg[DIR_DOWN][ll] = oldVal;
						}
					}
					// left
					if (j != 0) {
						val = aroundMsg + getSSD(ssdTable, k, ll, RIGHT_LEFT) * matchFactor;
						val -= nodeTable[i][j - 1].msg[DIR_RIGHT][k] * msgFactor;
						val /= msgCount;
						oldVal = nodeTable[i][j].newMsg[DIR_LEFT][ll];
						if (val < oldVal) {
							nodeTable[i][j].newMsg[DIR_LEFT][ll] = val;
						}
						else {
							nodeTable[i][j].newMsg[DIR_LEFT][ll] = oldVal;
						}
					}
					// right
					if (j != ww - 1) {
						val = aroundMsg + getSSD(ssdTable, k, ll, LEFT_RIGHT) * matchFactor;
						val -= nodeTable[i][j + 1].msg[DIR_LEFT][k] * msgFactor;
						val /= msgCount;
						oldVal = nodeTable[i][j].newMsg[DIR_RIGHT][ll];
						if (val < oldVal) {
							nodeTable[i][j].newMsg[DIR_RIGHT][ll] = val;
						}
						else {
							nodeTable[i][j].newMsg[DIR_RIGHT][ll] = oldVal;
						}
					}
				}
			}
		}
	}
	// copy the message data for iteration use
	for (int i = 0; i < hh; i++) {
		for (int j = 0; j < ww; j++) {
			nodeTable[i][j].msg = nodeTable[i][j].newMsg;
		}
	}
}

/*
 *find the best match patch for each node
 */
void selectPatch(vector<vector<node> > &nodeTable) {
	int hh = nodeTable.size(),
		ww = nodeTable[0].size();
	for (int i = 0; i < hh; i++) {
		for (int j = 0; j < ww; j++) {
			int len = nodeTable[i][j].edge_cost.size();
			float maxB = 0;
			int maxIdx = -1;
			for (int k = 0; k < len; k++) {
				float bl = -nodeTable[i][j].edge_cost[k];
				if (i - 1 >= 0 && nodeTable[i - 1][j].msg[DIR_DOWN][k] > 0)
					bl -= nodeTable[i - 1][j].msg[DIR_DOWN][k];
				if (i + 1 < hh && nodeTable[i + 1][j].msg[DIR_UP][k] > 0)
					bl -= nodeTable[i + 1][j].msg[DIR_UP][k];
				if (j - 1 >= 0 && nodeTable[i][j - 1].msg[DIR_RIGHT][k] > 0)
					bl -= nodeTable[i][j - 1].msg[DIR_RIGHT][k];
				if (j + 1 < ww && nodeTable[i][j + 1].msg[DIR_LEFT][k] > 0)
					bl -= nodeTable[i][j + 1].msg[DIR_LEFT][k];
				if (i == 1 && j == 1) {
					//cout<<k<<" =>"<<bl<<","<<FULL_MSG<<endl;
				}
				if (bl > maxB || maxIdx < 0) {
					maxB = bl;
					maxIdx = k;
				}
			}
			//cout<<i<<","<<j<<" => "<<maxB<<" "<<maxIdx<<endl;
			nodeTable[i][j].label = maxIdx;
		}
	}
}

/*
 * paste the patch to the corresponding node
 */
void pastePatch(Mat &img, node &n, patch &p) {
	int xx = n.x - NODE_WIDTH,
		yy = n.y - NODE_HEIGHT;
	for (int i = 0; i < p.height; i++) {
		for (int j = 0; j < p.width; j++) {
			img.at<Vec3f>(yy + i, xx + j) = img.at<Vec3f>(p.y + i, p.x + j);
		}
	}
}

/*
 * loop to fill all the target region
 */
void fillPatch(Mat &img, vector<vector<node> > &nodeTable, vector<patch> &patchList) {
	int hh = nodeTable.size(),
		ww = nodeTable[0].size();
	for (int i = 0; i < hh; i++) {
		for (int j = 0; j < ww; j++) {
			cout << nodeTable[i][j].label << " ";
		}
		cout << endl;
	}
	for (int i = 0; i < hh; i++) {
		for (int j = 0; j < ww; j++) {
			int label = nodeTable[i][j].label;
			if (label >= 0) {
				pastePatch(img, nodeTable[i][j], patchList[label]);
			}
		}
	}
}

/*
 * usage: ./inpainting sourceImage x y w h destinationImage iterationTime
 *
 *	sourceImage -> the input image
 *	x  -> the left top of the target region
 *	y  -> the right bottom of the target region
 *	w  -> the width of the target region
 *	h  -> the height of the target reion
 *	destinationImage -> the output image
 *	iterationTime -> the times for iteration
 */
int main(int argc, char **argv) {
	if (argc != 4) {
		cout << "Usage: " << argv[0] << " input output iter_time" << endl;
		return 0;
	}
	char *input = argv[1],
		*output = argv[2];
	int iterTime = atoi(argv[3]);
	Mat img;
	img = imread(input, IMREAD_COLOR);
	img.convertTo(img, CV_32FC3);

	if (!img.data) {
		cout << "Error reading image" << endl;
		return 0;
	}
	cout << "Image Size is: W= " << img.cols << " H = " << img.rows << endl;
	namedWindow("Draw Mask", WINDOW_FREERATIO);
	Mat mask = 255 * Mat::ones(img.size(), CV_8UC1);
	setMouseCallback("Draw Mask", drawMask,(void*) &mask);
	//waitKey(0);
	int key;
	while (1)
	{
		imshow("Draw Mask", mask);
		key = waitKey(10);
		if (key == 'q') {
			printf("Pressed q \n");
			break;
		}
	}

	//patch missing = roundUpArea(patch(maskX, maskY, maskW, maskH));
	//cout << "x=" << missing.x << " y=" << missing.y << " width=" << missing.width << " height=" << missing.height << endl;
	//vector<patch> patchList = genPatches(img, missing);
	//cout << "Patch Size: " << patchList.size() << endl;
	//vector<vector<vector<float> > > ssdTable = calculateSSDTable(img, patchList);

	//vector<vector<node> > nodeTable;
	//initNodeTable(img, nodeTable, missing, patchList);

	////fillPatch(img, nodeTable);
	//for (int i = 0; i < iterTime; i++) {
	//	propagateMsg(nodeTable, ssdTable);
	//	cout << "ITERATION " << i << endl;
	//}
	//selectPatch(nodeTable);
	//fillPatch(img, nodeTable, patchList);

	//// write the filled image to the destination image
	//imwrite(output, img);
	return 0;
}
