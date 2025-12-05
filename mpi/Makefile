MPICXX = mpicxx
CXXFLAGS = -O3 -std=c++17 `pkg-config --cflags --libs opencv4`
TARGET = hough_mpi.out
TARGET_STAR = hough_star_mpi.out

all: $(TARGET) $(TARGET_STAR)

$(TARGET): main.cpp
	$(MPICXX) main.cpp $(CXXFLAGS) -o $(TARGET) 

$(TARGET_STAR): main_star.cpp
	$(MPICXX) main_star.cpp $(CXXFLAGS) -o $(TARGET_STAR)

clean:
	rm -f $(TARGET) $(TARGET_STAR)

# mpicxx hough_mpi.cpp -O3 -std=c++17 `pkg-config --cflags --libs opencv4` -o hough_mpi


# mpirun -np 4 ./hough_mpi -i ../lena.png -o mpi_out.png --mode line
