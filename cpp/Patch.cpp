#include "Patch.h"

Patch::Patch(){
    x = 0;
    y = 0;
    width = 0;
    height = 0;
}

Patch::Patch(int xx, int yy, int ww, int hh){
    x = xx;
    y = yy;
    width = ww;
    height = hh;
}


// Rounds patch boundaries to obtain equal sized patches
void Patch::round() {
	Patch res;
	this->x = (this->x / NODE_WIDTH) * NODE_WIDTH;
	this->y = (this->y / NODE_HEIGHT) * NODE_HEIGHT;
	this->width = (this->x + this->width + NODE_WIDTH - 1) / NODE_WIDTH * NODE_WIDTH - this->x;
	this->height = (this->y + this->height + NODE_WIDTH - 1) / NODE_HEIGHT * NODE_HEIGHT - this->y;
}