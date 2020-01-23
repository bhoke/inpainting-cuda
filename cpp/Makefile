all: inpainting

inpainting: inpainting.cpp
	g++ -std=c++11 -g -O2 -o $@ $< `pkg-config --libs --cflags opencv4`

clean:
	rm -rf inpainting
