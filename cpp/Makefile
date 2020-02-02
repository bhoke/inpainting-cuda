all: inpainting

inpainting: inpainting.cpp
	g++ -std=c++11 -g -O3 -o $@ $< Patch.cpp `pkg-config --libs --cflags opencv4`

clean:
	rm -rf inpainting
