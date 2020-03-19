#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include "Patch.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

// how large neighborhood region should we extract the patches
const float RANGE_THRESH = 2.0f;

// the enumeration number indicate the relative position between two patches
enum EPOS
{
	UP_DOWN = 0,
	DOWN_UP,
	LEFT_RIGHT,
	RIGHT_LEFT,
	EPOS_COUNT,
};

void drawMask(int event, int x, int y, int flag, void *param)
{
	Mat &img = *((Mat *)(param));
	if (flag == (EVENT_FLAG_LBUTTON + EVENT_FLAG_ALTKEY))
	{
		circle(img, Point(x, y), 7, Scalar::all(255), FILLED);
	}
}

/*
 * Create patch list from source region
 */
vector<Patch> createPatches(Mat mask)
{
	vector<Point> targetPixels;
	Mat dilatedMask, dilationKernel;
	dilationKernel = Mat::ones(PATCH_HEIGHT * 2, PATCH_WIDTH * 2, CV_8U);
	dilate(mask, dilatedMask, dilationKernel);
	Mat patchRegion = dilatedMask - mask;
	namedWindow("patch", WINDOW_FREERATIO);
	imshow("patch", patchRegion);
	waitKey(0);

	findNonZero(patchRegion, targetPixels);
	vector<Patch> patchList;
	int initX = -INT32_MAX,
		initY = -INT32_MAX;
	for (auto it = targetPixels.begin(); it < targetPixels.end(); it++)
	{
		int patchStartX = it->x / PATCH_WIDTH * PATCH_WIDTH;
		int patchStartY = it->y / PATCH_WIDTH * PATCH_WIDTH;
		if ((patchStartX > (initX + PATCH_WIDTH - 1)) || (patchStartY > (initY + PATCH_HEIGHT - 1)))
		{
			initX = patchStartX;
			initY = patchStartY;
			Patch cur(patchStartX, patchStartY, PATCH_WIDTH, PATCH_HEIGHT);
			patchList.push_back(cur);
		}
	}
	cout << "Number of patches: " << patchList.size() << endl;
	return patchList;
}

/*
 * for simplicity, we only implement two kinds of relative position,
 * since DOWN_UP could be transformed to UP_DOWN
 * and RIGHT_LEFT could be transformed to LEFT_RIGHT
 */
int calculateSSD(Mat &img, Patch &p1, Patch &p2, EPOS pos)
{
	int res = 0;
	switch (pos)
	{
	case UP_DOWN:
		for (int i = 0; i < NODE_HEIGHT; i++)
		{
			for (int j = 0; j < PATCH_WIDTH; j++)
			{
				Vec3b pv1 = img.at<Vec3b>(p1.y + NODE_HEIGHT + i, p1.x + j),
					  pv2 = img.at<Vec3b>(p2.y + i, p2.x + j);
				for (int k = 0; k < 3; k++)
				{
					int m = pv1[k] - pv2[k];
					res += m * m;
				}
			}
		}
		break;
	case LEFT_RIGHT:
		for (int i = 0; i < PATCH_HEIGHT; i++)
		{
			for (int j = 0; j < NODE_WIDTH; j++)
			{
				Vec3b pv1 = img.at<Vec3b>(p1.y + i, p1.x + NODE_WIDTH + j),
					  pv2 = img.at<Vec3b>(p2.y + i, p2.x + j);
				for (int k = 0; k < 3; k++)
				{
					int m = pv1[k] - pv2[k];
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
vector<vector<vector<int>>> calculateSSDTable(Mat &img, vector<Patch> &patchList)
{
	vector<vector<vector<int>>> res;
	size_t len = patchList.size();
	res.resize(len);
	for (size_t i = 0; i < len; i++)
	{
		res[i].resize(len);
	}
	for (size_t i = 0; i < len; i++)
	{
		for (size_t j = i; j < len; j++)
		{
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
int getSSD(vector<vector<vector<int>>> &ssdTable, int p1, int p2, EPOS pos)
{
	if (p1 > p2)
	{
		switch (pos)
		{
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

enum EDIR
{
	DIR_UP = 0,
	DIR_DOWN,
	DIR_LEFT,
	DIR_RIGHT,
	DIR_COUNT,
};

// node class represent one node in the MRF model
class node
{
public:
	vector<vector<float>> msg;	// message vector for all directions and all patches
	vector<vector<float>> newMsg; // use to iteration
	vector<float> edge_cost;	  // the cost on the edge, internal node will have zero in this field
	int label;					  // the best Patch to be selected for this node
	int x;						  // Node's X coordinate
	int y;						  // Node's Y coordinate
								  // node(int xx, int yy, int lbl) : label(lbl), x(xx), y(yy){}
};

/*
 * to initialize the node table
 * include message and edge cost
 */
void initNodeTable(Mat &img, Mat mask, vector<vector<node>> &nodeTable, vector<Patch> &patchList)
{
	vector<Point> targetPixels;
	findNonZero(mask, targetPixels);
	int initX = -INT32_MAX,
		initY = -INT32_MAX,
		ind = 0,
		ind2 = 0;

	Point first = targetPixels.front() / NODE_WIDTH * NODE_WIDTH;
	Point last = targetPixels.back() / NODE_HEIGHT * NODE_HEIGHT;
	Point frame = last - first;
	int ww = frame.x / NODE_WIDTH + 1;
	int hh = frame.y / NODE_HEIGHT + 1;

	size_t len = patchList.size();
	cout << "patch" << len << endl;
	nodeTable.resize(hh);
	//cout << "hh=" << hh << " ww=" << ww << endl;
	for (int i = 0; i < hh; i++)
	{
		nodeTable[i].resize(ww);
		for (size_t j = 0; j < ww; j++)
		{
			nodeTable[i][j].msg.resize(DIR_COUNT);
			for (size_t k = 0; k < DIR_COUNT; k++)
			{
				nodeTable[i][j].msg[k].resize(len);
				for (size_t l = 0; l < len; l++)
					nodeTable[i][j].msg[k][l] = FULL_MSG;
			}
			nodeTable[i][j].label = -1;
			nodeTable[i][j].x = first.x + j * NODE_WIDTH;
			nodeTable[i][j].y = first.y + i * NODE_HEIGHT;
			nodeTable[i][j].edge_cost.resize(len);
			for (size_t k = 0; k < len; k++)
			{
				float val = 0;
				Patch curPatch(0, 0, PATCH_WIDTH, PATCH_HEIGHT);
				// only the node on the edge need to calculate the SSD
				if ((i == 0 || i == hh - 1) || (j == 0 || j == ww - 1))
				{
					if (j == 0)
					{
						curPatch.x = nodeTable[i][j].x - PATCH_WIDTH;
						curPatch.y = nodeTable[i][j].y - NODE_HEIGHT;
						val += calculateSSD(img, curPatch, patchList[k], LEFT_RIGHT);
					}
					else
					{
						curPatch.x = nodeTable[i][j].x;
						curPatch.y = nodeTable[i][j].y - NODE_HEIGHT;
						val += calculateSSD(img, patchList[k], curPatch, LEFT_RIGHT);
					}
					if (i == 0)
					{
						curPatch.x = nodeTable[i][j].x - NODE_WIDTH;
						curPatch.y = nodeTable[i][j].y - PATCH_HEIGHT;
						val += calculateSSD(img, curPatch, patchList[k], UP_DOWN);
					}
					else
					{
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
	cout << "nodeTable Size: " << nodeTable.size() << endl;
}

/*
 * this function is iteration for the belief propagation
 */
void propagateMsg(vector<vector<node>> &nodeTable, vector<vector<vector<int>>> &ssdTable)
{
	size_t hh = nodeTable.size(),
		   len = ssdTable.size();
	for (size_t i = 0; i < hh; i++)
	{
		size_t ww = nodeTable[i].size();
		for (size_t j = 0; j < ww; j++)
		{
			for (size_t k = 0; k < len; k++)
			{
				float aroundMsg = 0, msgCount, matchFactor = 1.2f;
				float msgFactor = 0.6f; // how important is messages from the adjacent node
				msgCount = msgFactor * 3 + matchFactor;
				if (i != 0)
					aroundMsg += nodeTable[i - 1][j].msg[DIR_DOWN][k];
				else
					aroundMsg += FULL_MSG;
				if (i != hh - 1)
					aroundMsg += nodeTable[i + 1][j].msg[DIR_UP][k];
				else
					aroundMsg += FULL_MSG;
				if (j != 0)
					aroundMsg += nodeTable[i][j - 1].msg[DIR_RIGHT][k];
				else
					aroundMsg += FULL_MSG;
				if (j != ww - 1)
					aroundMsg += nodeTable[i][j + 1].msg[DIR_LEFT][k];
				else
					aroundMsg += FULL_MSG;

				aroundMsg *= msgFactor;
				aroundMsg += nodeTable[i][j].edge_cost[k];
				// std::cout << j <<  << std::endl;
				for (size_t ll = 0; ll < len; ll++)
				{
					float val, oldVal;
					/*
					 * in this loop, go over all the patches to update the message
					 * for each Patch in all directions
					 */
					// up
					if (i != 0)
					{
						val = aroundMsg + getSSD(ssdTable, k, ll, DOWN_UP) * matchFactor;
						val -= nodeTable[i - 1][j].msg[DIR_DOWN][k] * msgFactor;
						val /= msgCount;
						// cout << "i: " << i << " j: " << j << endl;
						oldVal = nodeTable[i][j].newMsg[DIR_UP][ll];
						if (val < oldVal)
						{
							// 	// cout << "NTSize: " << nodeTable.size() << endl;
							// 	// cout << "NT[i]Size: " << nodeTable[j].size() << endl;
							nodeTable[i][j].newMsg[DIR_UP][ll] = val;
						}
						else
						{
							nodeTable[i][j].newMsg[DIR_UP][ll] = oldVal;
						}
					}
					// down
					if (i != hh - 1)
					{
						val = aroundMsg + getSSD(ssdTable, k, ll, UP_DOWN) * matchFactor;
						val -= nodeTable[i + 1][j].msg[DIR_UP][k] * msgFactor;
						val /= msgCount;
						oldVal = nodeTable[i][j].newMsg[DIR_DOWN][ll];
						if (val < oldVal)
						{
							nodeTable[i][j].newMsg[DIR_DOWN][ll] = val;
						}
						else
						{
							nodeTable[i][j].newMsg[DIR_DOWN][ll] = oldVal;
						}
					}
					// left
					if (j != 0)
					{
						val = aroundMsg + getSSD(ssdTable, k, ll, RIGHT_LEFT) * matchFactor;
						val -= nodeTable[i][j - 1].msg[DIR_RIGHT][k] * msgFactor;
						val /= msgCount;
						oldVal = nodeTable[i][j].newMsg[DIR_LEFT][ll];
						if (val < oldVal)
						{
							nodeTable[i][j].newMsg[DIR_LEFT][ll] = val;
						}
						else
						{
							nodeTable[i][j].newMsg[DIR_LEFT][ll] = oldVal;
						}
					}
					// right
					if (j != ww - 1)
					{
						val = aroundMsg + getSSD(ssdTable, k, ll, LEFT_RIGHT) * matchFactor;
						val -= nodeTable[i][j + 1].msg[DIR_LEFT][k] * msgFactor;
						val /= msgCount;
						oldVal = nodeTable[i][j].newMsg[DIR_RIGHT][ll];
						if (val < oldVal)
						{
							nodeTable[i][j].newMsg[DIR_RIGHT][ll] = val;
						}
						else
						{
							nodeTable[i][j].newMsg[DIR_RIGHT][ll] = oldVal;
						}
					}
				}
			}
		}
	}
	// copy the message data for iteration use
	for (size_t i = 0; i < hh; i++)
	{
		size_t ww = nodeTable[i].size();
		for (size_t j = 0; j < ww; j++)
		{
			nodeTable[i][j].msg = nodeTable[i][j].newMsg;
		}
	}
}

/*
 *find the best match Patch for each node
 */
void selectPatch(vector<vector<node>> &nodeTable)
{
	size_t hh = nodeTable.size(),
		   ww = nodeTable[0].size();
	cout << "hh: " << hh << " ww: " << ww << endl;
	for (size_t i = 0; i < hh; i++)
	{
		for (size_t j = 0; j < ww; j++)
		{
			size_t len = nodeTable[i][j].edge_cost.size();
			float maxB = 0;
			int maxIdx = -1;
			for (size_t k = 0; k < len; k++)
			{
				// cout << " asd: " <<nodeTable[i][j].edge_cost[k] << endl;
				// cout << "i: " << i << " j: " << j << " k: " << k << endl;
				float bl = -nodeTable[i][j].edge_cost[k];
				if (i > 0)
					if (nodeTable[i - 1][j].msg[DIR_DOWN][k] > 0)
						bl -= nodeTable[i - 1][j].msg[DIR_DOWN][k];

				if (i + 1 < hh)
					if (nodeTable[i + 1][j].msg[DIR_UP][k] > 0)
						bl -= nodeTable[i + 1][j].msg[DIR_UP][k];

				if (j > 0)
					if (nodeTable[i][j - 1].msg[DIR_RIGHT][k] > 0)
						bl -= nodeTable[i][j - 1].msg[DIR_RIGHT][k];
				if (j + 1 < ww)
					if (nodeTable[i][j + 1].msg[DIR_LEFT][k] > 0)
						bl -= nodeTable[i][j + 1].msg[DIR_LEFT][k];
				
				if (bl > maxB || maxIdx < 0)
				{
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
 * paste the Patch to the corresponding node
 */
void pastePatch(Mat &img, node &n, Patch &p)
{
	int xx = n.x - NODE_WIDTH,
		yy = n.y - NODE_HEIGHT;
	for (int i = 0; i < p.height; i++)
	{
		for (int j = 0; j < p.width; j++)
		{
			img.at<Vec3b>(yy + i, xx + j) = img.at<Vec3b>(p.y + i, p.x + j);
		}
	}
}

/*
 * loop to fill all the target region
 */
void fillPatch(Mat &img, vector<vector<node>> &nodeTable, vector<Patch> &patchList)
{
	size_t hh = nodeTable.size(),
		   ww = nodeTable[0].size();
	for (size_t i = 0; i < hh; i++)
	{
		for (size_t j = 0; j < ww; j++)
		{
			cout << nodeTable[i][j].label << " ";
		}
		cout << endl;
	}
	for (size_t i = 0; i < hh; i++)
	{
		for (size_t j = 0; j < ww; j++)
		{
			int label = nodeTable[i][j].label;
			if (label >= 0)
			{
				pastePatch(img, nodeTable[i][j], patchList[label]);
			}
		}
	}
}

int main(int argc, const char *argv[])
{
	// char *input, *output;
	int iterTime;

	const char *default_args[] = {argv[0], "../topkapi.jpg", "../topkapi_inp.png", "100"};

	if (argc == 1)
	{
		argv = default_args;
	}
	else if (argc == 4)
	{
	}
	else
	{
		cout << "Usage: " << argv[0] << "input_file output_file iter_time" << endl;
		return 0;
	}

	char const *input = argv[1];
	char const *output = argv[2];
	iterTime = atoi(argv[3]);

	Mat img;
	img = imread(input, IMREAD_COLOR);

	if (!img.data)
	{
		cout << "Error reading image" << endl;
		return 0;
	}

	Mat imgCopy;
	img.copyTo(imgCopy);

	cout << "Image Size is: W= " << img.cols << " H = " << img.rows << endl;
	namedWindow("Draw Mask", WINDOW_FREERATIO);
	setMouseCallback("Draw Mask", drawMask, (void *)&imgCopy);
	int key;
	while (1)
	{
		imshow("Draw Mask", imgCopy);
		key = waitKey(10);
		if (key == 'y')
		{
			printf("Pressed y \n");
			break;
		}
		if (key == 'n')
		{
			printf("Pressed n. Process cancelled \nTerminating...");
			return 1;
		}
	}
	destroyAllWindows();
	Mat mask = imgCopy - img;
	cvtColor(mask, mask, COLOR_BGR2GRAY);
	vector<Patch> patchList = createPatches(mask);
	vector<vector<vector<int>>> ssdTable = calculateSSDTable(imgCopy, patchList);
	vector<vector<node>> nodeTable;
	initNodeTable(img, mask, nodeTable, patchList);
	// return 1;
	for (int i = 0; i < iterTime; i++)
	{
		propagateMsg(nodeTable, ssdTable);
		cout << "ITERATION " << i << endl;
	}
	selectPatch(nodeTable);
	fillPatch(imgCopy, nodeTable, patchList);

	namedWindow("Output", WINDOW_FREERATIO);
	imshow("Output", imgCopy);
	imwrite(output, imgCopy);
	waitKey(0);
	return 0;
}
