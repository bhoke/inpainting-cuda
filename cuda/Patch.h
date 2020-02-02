const int NODE_WIDTH = 8,
NODE_HEIGHT = 8,
PATCH_WIDTH = 2 * NODE_WIDTH,
PATCH_HEIGHT = 2 * NODE_HEIGHT;
const float FULL_MSG = PATCH_WIDTH * PATCH_HEIGHT * 255 * 255 * 3 / 2;

class Patch {
public:
	int x; // Top left corner of the 
	int y; 
	int width;
	int height;
	Patch();
	Patch(int xx, int yy, int ww, int hh);
    bool hasOverlap(Patch &p);
    void round();
};