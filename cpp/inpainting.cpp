#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

#define RADIUS	(32)

const int PATCH_WIDTH = RADIUS;
const int PATCH_HEIGHT = RADIUS;
const int NODE_WIDTH = PATCH_WIDTH / 2;
const int NODE_HEIGHT = PATCH_HEIGHT / 2;

const float FULL_MSG = PATCH_HEIGHT * PATCH_WIDTH * 255 * 255;

class patch {
	public:
		int x; // width
		int y; // height
		int width;
		int height;
		patch() : x(0), y(0), width(0), height(0) {}
		patch(int xx, int yy, int ww, int hh) : x(xx), y(yy), width(ww), height(hh) {}
};

enum EPOS {
	UP_DOWN = 0,
	DOWN_UP,
	LEFT_RIGHT,
	RIGHT_LEFT,
	EPOS_COUNT,
};

patch roundDownArea(patch p) {
	patch res;
	/*
	res.x = (p.x / NODE_WIDTH) * NODE_WIDTH + NODE_WIDTH;
	res.y = (p.y / NODE_HEIGHT) * NODE_HEIGHT + NODE_HEIGHT;
	res.width = (p.x + p.width) / NODE_WIDTH * NODE_WIDTH - NODE_WIDTH - res.x;
	res.height = (p.y + p.height) / NODE_HEIGHT * NODE_HEIGHT - NODE_HEIGHT - res.y;
	*/
	res.x = (p.x / NODE_WIDTH) * NODE_WIDTH - NODE_WIDTH;
	res.y = (p.y / NODE_HEIGHT) * NODE_HEIGHT - NODE_HEIGHT;
	res.width = (p.x + p.width) / NODE_WIDTH * NODE_WIDTH + NODE_WIDTH - res.x;
	res.height = (p.y + p.height) / NODE_HEIGHT * NODE_HEIGHT + NODE_HEIGHT - res.y;

	return res;
}

bool overlapPatch(patch &p1, patch &p2) {
	int mLX = p1.x < p2.x ? p2.x : p1.x,
	    mRX = (p1.x+p1.width) < (p2.x+p2.width) ? (p1.x+p1.width) : (p2.x+p2.width),
	    mTY = p1.y < p2.y ? p2.y : p1.y,
	    mBY = (p1.y+p1.height) < (p2.y+p2.height) ? (p1.y+p1.height) : (p2.y+p2.height);
	return mRX > mLX && mBY > mTY;
}

vector<patch> genPatches(Mat &img, patch p) {
	vector<patch> res;
	int hh = img.rows / NODE_HEIGHT,
	    ww = img.cols / NODE_WIDTH;
	for(int i = 1; i <= hh; i++) {
		for(int j = 1; j <= ww; j++) {
			int cX, cY;
			cY = i * NODE_HEIGHT - NODE_HEIGHT;
			cX = j * NODE_WIDTH - NODE_WIDTH;
			if(img.rows - cY < PATCH_HEIGHT || img.cols - cX < PATCH_WIDTH)
				continue;
			patch cur(cX, cY, PATCH_WIDTH, PATCH_HEIGHT);
			if(!overlapPatch(cur, p))
				res.push_back(cur);
		}
	}
	return res;
}

float calculateSSD(Mat &img, patch &p1, patch &p2, EPOS pos) {
	float res = 0;
	int ww, hh;
	switch(pos) {
		case UP_DOWN:
			ww = p1.width < p2.width ? p1.width : p2.width;
			hh = (p1.height - NODE_HEIGHT) < p2.height ? (p1.height - NODE_HEIGHT): p2.height;
			for(int i = 0; i < hh; i++) {
				for(int j = 0; j < ww; j++) {
					Vec3f pv1 = img.at<Vec3f>(p1.y + NODE_HEIGHT + i, p1.x + j),
					      pv2 = img.at<Vec3f>(p2.y + i, p2.x + j);
					for(int k = 0; k < 3; k++) {
						float m = pv1[k] - pv2[k];
						res += m * m;
					}
				}
			}
			break;
		case LEFT_RIGHT:
			hh = p1.height < p2.height ? p1.height : p2.height;
			ww = (p1.width - NODE_WIDTH) < p2.width ? (p1.width - NODE_WIDTH) : p2.width;
			for(int i = 0; i < hh; i++) {
				for(int j = 0; j < ww; j++) {
					Vec3f pv1 = img.at<Vec3f>(p1.y + i, p1.x + NODE_WIDTH + j),
					      pv2 = img.at<Vec3f>(p2.y + i, p2.x + j);
					for(int k = 0; k < 3; k++) {
						float m = pv1[k] - pv2[k];
						res += m * m;
					}
				}
			}
			break;
		default:
			cout<<"FATAL ERROR"<<endl;
			exit(-1);
	}
	return res;
}

vector<vector<vector<float> > > calculateSSDTable(Mat &img, vector<patch> &patchList) {
	vector<vector<vector<float> > > res;
	int len = patchList.size();
	res.resize(len);
	for(int i = 0; i < len; i++) {
		res[i].resize(len);
	}
	for(int i = 0; i < len; i++) {
		for(int j = i; j < len; j++) {
			if(0/*i == j*/) {
				/*
				 * do nothing since the result must be zero
				 */
				/*
				for(int k = 0; k < EPOS_COUNT; k++) 
					res[i][j].push_back(0);
				*/
			} else {
				res[i][j].resize(EPOS_COUNT);
				res[i][j][UP_DOWN] = calculateSSD(img, patchList[i], patchList[j], UP_DOWN);
				res[i][j][DOWN_UP] = calculateSSD(img, patchList[j], patchList[i], UP_DOWN);
				res[i][j][LEFT_RIGHT] = calculateSSD(img, patchList[i], patchList[j], LEFT_RIGHT);
				res[i][j][RIGHT_LEFT] = calculateSSD(img, patchList[j], patchList[i], LEFT_RIGHT);
			}
		}
	}
	return res;
}

float getSSD(vector<vector<vector<float> > > &ssdTable, int p1, int p2, EPOS pos) {
	if(p1 > p2) {
		switch(pos) {
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
				cout<<"ERROR"<<endl;
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

class node {
	public:
		vector<vector<float> > msg;
		vector<float> edge_cost;
		int label;
		int x;
		int y;
};


void initNodeTable(Mat &img, vector<vector<node> > &nodeTable, patch &p, vector<patch> &patchList) {
	int hh = p.height / NODE_HEIGHT + 1,
	    ww = p.width / NODE_WIDTH + 1,
	    len = patchList.size();
	nodeTable.resize(hh);
	for(int i = 0; i < hh; i++) {
		nodeTable[i].resize(ww);
		for(int j = 0; j < ww; j++) {
			nodeTable[i][j].msg.resize(DIR_COUNT);
			for(int k = 0; k < DIR_COUNT; k++) {
				nodeTable[i][j].msg[k].resize(len);
				for(int l = 0; l < len; l++) {
					nodeTable[i][j].msg[k][l] = FULL_MSG;
				}
			}
			nodeTable[i][j].label = -1;
			nodeTable[i][j].x = p.x + j * NODE_WIDTH;
			nodeTable[i][j].y = p.y + i * NODE_HEIGHT;
			nodeTable[i][j].edge_cost.resize(len);
			for(int k = 0; k < len; k++) {
				float val = 0;
				patch curPatch(0, 0, PATCH_WIDTH, PATCH_HEIGHT);
				if((i == 0 || i == hh - 1) && (j == 0 || j == ww - 1)) {
					if(j == 0) {
						curPatch.x = nodeTable[i][j].x - PATCH_WIDTH;
						curPatch.y = nodeTable[i][j].y - NODE_HEIGHT;
						val += calculateSSD(img, curPatch, patchList[k], LEFT_RIGHT);
					} else {
						curPatch.x = nodeTable[i][j].x;
						curPatch.y = nodeTable[i][j].y - NODE_HEIGHT;
						val += calculateSSD(img, patchList[k], curPatch, LEFT_RIGHT);
					}
					if(i == 0) {
						curPatch.x = nodeTable[i][j].x - NODE_WIDTH;
						curPatch.y = nodeTable[i][j].y - PATCH_HEIGHT;
						val += calculateSSD(img, curPatch, patchList[k], UP_DOWN);
					} else {
						curPatch.x = nodeTable[i][j].x - NODE_WIDTH;
						curPatch.y = nodeTable[i][j].y;
						val += calculateSSD(img, patchList[k], curPatch, UP_DOWN);
					}
				}
				nodeTable[i][j].edge_cost[k] = val;
			}
		}
	}
}

void propagateMsg(vector<vector<node> > &nodeTable, vector<vector<vector<float> > > &ssdTable) {
	int hh = nodeTable.size(),
	    ww = nodeTable[0].size(),
	    len = ssdTable.size();
	for(int i = 0; i < hh; i++) {
		for(int j = 0; j < ww; j++) {
			for(int k = 0; k < len; k++) {
				float aroundMsg = 0, msgCount = 0;
				if(i != 0) {
					aroundMsg += nodeTable[i-1][j].msg[DIR_DOWN][k];
					msgCount++;
				}
				if(i != hh - 1) {
					aroundMsg += nodeTable[i+1][j].msg[DIR_UP][k];
					msgCount++;
				}
				if(j != 0) {
					aroundMsg += nodeTable[i][j-1].msg[DIR_RIGHT][k];
					msgCount++;
				}
				if(j != ww - 1) {
					aroundMsg += nodeTable[i][j+1].msg[DIR_LEFT][k];
					msgCount++;
				}
				if(msgCount > 0.5) {
					aroundMsg /= msgCount;
				}
				aroundMsg += nodeTable[i][j].edge_cost[k];
				for(int ll = 0; ll < len; ll++) {
					float val, oldVal;
					// up
					if(i != 0) {
						val = aroundMsg + getSSD(ssdTable, k, ll, DOWN_UP);
						oldVal = nodeTable[i][j].msg[DIR_UP][ll];
						if(val < oldVal) {
							nodeTable[i][j].msg[DIR_UP][ll] = val;
						}
					}
					// down
					if(i != hh - 1) {
						val = aroundMsg + getSSD(ssdTable, k, ll, UP_DOWN);
						oldVal = nodeTable[i][j].msg[DIR_DOWN][ll];
						if(val < oldVal) {
							nodeTable[i][j].msg[DIR_DOWN][ll] = val;
						}
					}
					// left
					if(j != 0) {
						val = aroundMsg + getSSD(ssdTable, k, ll, RIGHT_LEFT);
						oldVal = nodeTable[i][j].msg[DIR_LEFT][ll];
						if(val < oldVal) {
							nodeTable[i][j].msg[DIR_LEFT][ll] = val;
						}
					}
					// right
					if(j != ww - 1) {
						val = aroundMsg + getSSD(ssdTable, k, ll, LEFT_RIGHT);
						oldVal = nodeTable[i][j].msg[DIR_RIGHT][ll];
						if(val < oldVal) {
							nodeTable[i][j].msg[DIR_RIGHT][ll];
						}
					}
				}
			}
		}
	}
}

void selectPatch(vector<vector<node> > &nodeTable) {
	int hh = nodeTable.size(),
	    ww = nodeTable[0].size();
	for(int i = 0; i < hh; i++) {
		for(int j = 0; j < ww; j++) {
			int len = nodeTable[i][j].edge_cost.size();
			float maxB = 0;
			int maxIdx = -1;
			for(int k = 0; k < len; k++) {
				float bl = -nodeTable[i][j].edge_cost[k];
				if(i - 1 >= 0 && nodeTable[i-1][j].msg[DIR_DOWN][k] > 0)
					bl -= nodeTable[i-1][j].msg[DIR_DOWN][k];
				if(i + 1 < hh && nodeTable[i+1][j].msg[DIR_UP][k] > 0)
					bl -= nodeTable[i+1][j].msg[DIR_UP][k];
				if(j - 1 >= 0 && nodeTable[i][j-1].msg[DIR_RIGHT][k] > 0) 
					bl -= nodeTable[i][j-1].msg[DIR_RIGHT][k];
				if(j + 1 < ww && nodeTable[i][j+1].msg[DIR_LEFT][k] > 0) 
					bl -= nodeTable[i][j+1].msg[DIR_LEFT][k];
				if(bl > maxB || maxIdx < 0) {
					maxB = bl;
					maxIdx = k;
				}
			}
			nodeTable[i][j].label = maxIdx;
		}
	}
}

void pastePatch(Mat &img, node &n, patch &p) {
	int xx = n.x - NODE_WIDTH,
	    yy = n.y - NODE_HEIGHT;
	for(int i = 0; i < p.height; i++) {
		for(int j = 0; j < p.width; j++) {
			img.at<Vec3f>(yy + i, xx + j) = img.at<Vec3f>(p.y + i, p.x + j);
		}
	}
}

void fillPatch(Mat &img, vector<vector<node> > &nodeTable, vector<patch> &patchList) {
	int hh = nodeTable.size(),
	    ww = nodeTable[0].size();
	for(int i = 0; i < hh; i++) {
		for(int j = 0; j < ww; j++) {
			cout<<nodeTable[i][j].label<<" ";
		}
		cout<<endl;
	}
	for(int i = 0; i < hh; i++) {
		for(int j = 0; j < ww; j++) {
			int label = nodeTable[i][j].label;
			if(label >= 0) {
				pastePatch(img, nodeTable[i][j], patchList[label]);
			}
		}
	}
}

int main(int argc, char **argv) {
	if(argc != 8) {
		cout<<"Usage: "<<argv[0]<<" input x y w h output iter_time"<<endl;
		return 0;
	}
	char *input = argv[1],
	     *output = argv[6];
	int maskX = atoi(argv[2]),
	    maskY = atoi(argv[3]),
	    maskW = atoi(argv[4]),
	    maskH = atoi(argv[5]),
	    iterTime = atoi(argv[7]);
	Mat img;
	img = imread(input, CV_LOAD_IMAGE_COLOR);
	img.convertTo(img, CV_32FC3);

	if(!img.data) {
		cout<<"Load Image failed"<<endl;
		return 0;
	}
	cout<<"W="<<img.cols<<endl;
	cout<<"H="<<img.rows<<endl;

	patch missing = roundDownArea(patch(maskX, maskY, maskW, maskH));
	cout<<"x="<<missing.x<<" y="<<missing.y<<" width="<<missing.width<<" height="<<missing.height<<endl;
	vector<patch> patchList = genPatches(img, missing);
	cout<<"Patch Size: "<<patchList.size()<<endl;
	vector<vector<vector<float> > > ssdTable = calculateSSDTable(img, patchList);

	vector<vector<node> > nodeTable;
	initNodeTable(img, nodeTable, missing, patchList);
	
	//fillPatch(img, nodeTable);
	for(int i = 0; i < iterTime; i++) {
		propagateMsg(nodeTable, ssdTable);
		cout<<"ITERATION "<<i<<endl;
	}
	selectPatch(nodeTable);
	fillPatch(img, nodeTable, patchList);

	imwrite(output, img);
	return 0;
}