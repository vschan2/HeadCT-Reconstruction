// Derived from VTK/Examples/Cxx/Medical2.cxx
// The original example reads a volume dataset, extracts two isosurfaces that
// represent the skin and bone, and then displays them.
//
// ======================================================================================
//
// Edited by: Chan Vei Siang
// This example shows the 3D head CT scan reconstruction using various hole-filling
// methods of pixel nearest-neighbour (PNN).
// Firstly, the program reads a volume dataset, and removes a whole slice at position n.
// User can also choose to remove 2 or 3 slices. The slice removal can be removed at a
// specific sparsity s. Then, the hole-filling method is employed to reconstruct the 
// missing region.
// 
// ======================================================================================
//
// How to run this example?
// 1) Using CMake to create this project. The procedure is the same as any VTK projects.
// 2) Build the project using Visual Studio. A new "Debug" folder will be created.
// 3) Put the volume dataset file (headsq) in the "Debug" folder.
// 4) Open command prompt and navigate to the "Debug" folder.
// 5) Run "HeadCTReconstruction *outputDatasetName(string) *sliceNo(int) *method(string) 
//    *parameter(int, float)"
//    e.g.: "HeadCTReconstruction headsq 7 mean 3"
//    e.g.: "HeadCTReconstruction headsq 7 butterfly-my 0 3 0"
// 6) Then, input the number of continuous slice to remove.
// 7) Input the sparsity value.
// 8) Lastly, input the increment limit of sparsity value. If you want to remove 2 slices
//    in every 7 slices spacing for 10 times: 2 -> 7 -> 10.
// 9) The result is displayed. The output axial slices are stored in the "figures" folder.
// 
// * NOTE: 
// 1) The dataset used in this example consists of 12-bits grey-scale pixel with 
//    little-endian arrangement.
// 2) The input dataset is preset to "fullHeadRaw" folder. See main().
//
// ======================================================================================
// 
// For more information, please visit https://doi.org/10.119/GAME47560.2019.8980511 and 
// its ResearchGate counterpart.

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkMarchingCubes.h>
#include <vtkVolume16Reader.h>
#include <vtkMetaImageReader.h>
#include <vtkNamedColors.h>
#include <vtkOutlineFilter.h>
#include <vtkPlaneSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkStripper.h>
#include <vtkImageData.h>
#include <vtkImageIterator.h>
#include <vtkPointData.h>
#include <vtkLookupTable.h>
#include <vtkImageMapToColors.h>
#include <vtkImageActor.h>
#include <vtkImageMapper3D.h>
#include <vtkExtractVOI.h>
#include <vtkPNGWriter.h>
#include <vtkNamedColors.h>

#include <array>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

const int twelveBitInt = 256;
const int totalNeighbour = 26;
const double pi = 3.1415926535897;


// Remove an image slice from the input volume i based on z in axial direction.
void cleanImageDataOnAxialZ(vtkSmartPointer<vtkImageData> i, int z)
{
	int* dims = i->GetDimensions();
	unsigned char r, g, b, a;
	vtkSmartPointer<vtkNamedColors> colors =
		vtkSmartPointer<vtkNamedColors>::New();
	colors->GetColor("Black", r, g, b, a);

	for (int y = 0; y < dims[1]; y++)
	{
		for (int x = 0; x < dims[0]; x++)
		{
			unsigned char* pixel = static_cast<unsigned char*>(i->GetScalarPointer(x, y, z));

			*pixel++ = (unsigned char)0;
			*pixel++ = (unsigned char)0;
		}
	}
}

// Perform average operation on the input vector.
float averageOperation(std::vector<int> nPs)
{
	int length = nPs.size();
	float total = .0;

	for (int i = 0; i < length; i++) total += nPs[i];

	return (total / length);
}

int findRangeWidth2(std::vector<int> nPs)
{
	int min = 0, max = 0;

	std::sort(nPs.begin(), nPs.end());

	min = nPs[0];
	max = nPs[nPs.size() - 1];

	return ((int)(max - min));
}

// Return a list of neighbouring voxel with grey-scale value around a voxel (x, y, z) 
// in input volume i within a specified kernel size k.
std::vector<int> getNeighbourVoxels(vtkSmartPointer<vtkImageData> i, int x, int y, int z, int k)
{
	int kernelSize = k;
	int temp = 0;
	int* dims = i->GetDimensions();
	unsigned char* pixel;
	std::vector<int> neighbourVoxels;

	for (int c = -kernelSize; c <= kernelSize; c++)
	{
		for (int b = -kernelSize; b <= kernelSize; b++)
		{
			for (int a = -kernelSize; a <= kernelSize; a++)
			{
				if ((x + a) == x && (y + b) == y && (z + c) == z)
					continue;

				if ((x + a) < 0 || (x + a) > dims[0] - 1 || (y + b) < 0 || (y + b) > dims[1] - 1 || 
					(z + c) < 0 || (z + c) > dims[2] - 1)
					continue;

				pixel = static_cast<unsigned char*>(i->GetScalarPointer(x + a, y + b, z + c));
				temp = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
				if (temp > 0) neighbourVoxels.push_back(temp);
			}
		}
	}

	return neighbourVoxels;
}


// Perform PNN mean operation on input volume i on axial slice z with maximum kernel size p1.
// Input volume o is the original volume (with holes). 
// Input volume i is the hole-filled volume.
void pnnMeanOperationOnAxialZ(vtkSmartPointer<vtkImageData> i, vtkSmartPointer<vtkImageData> o, int z, int p1)
{
	const int kernelSize = p1;
	int* dims = i->GetDimensions();

	cout << "Running mean operation ... " << endl
		 << "Kernel size = " << kernelSize << endl;

	for (int y = 0; y < (dims[1] - 1); y++)
	{
		for (int x = 0; x < (dims[0] - 1); x++)
		{
			int kernel = 1;
			float mean = .0;
			std::vector<int> neighbourVoxels;
			
			do {
				neighbourVoxels = getNeighbourVoxels(o, x, y, z, kernel);

				if (neighbourVoxels.size() <= 0) kernel++;
				if (kernel >= kernelSize) break;

			} while (neighbourVoxels.size() <= 0);

			mean = averageOperation(neighbourVoxels);

			unsigned char* finalPixel = static_cast<unsigned char*>(i->GetScalarPointer(x, y, z));
			*finalPixel++ = (unsigned char)((int)mean % twelveBitInt);
			*finalPixel++ = (unsigned char)((int)mean / twelveBitInt);
		}
	}
}


// Perform PNN median operation on input volume i on axial slice z with maximum kernel size p1.
// Input volume o is the original volume (with holes). 
// Input volume i is the hole-filled volume.
void pnnMedianOperationOnAxialZ(vtkSmartPointer<vtkImageData> i, vtkSmartPointer<vtkImageData> o, int z, int p1)
{
	const int kernelSize = p1;
	int* dims = i->GetDimensions();

	cout << "Running median operation ... " << endl
		 << "Kernel size = " << kernelSize << endl;

	for (int y = 0; y < (dims[1] - 1); y++)
	{
		for (int x = 0; x < (dims[0] - 1); x++)
		{
			int kernel = 1;
			std::vector<int> neighbourVoxels;

			do {
				neighbourVoxels = getNeighbourVoxels(o, x, y, z, kernel);

				if (neighbourVoxels.size() <= 0) kernel++;
				if (kernel >= kernelSize) break;

			} while (neighbourVoxels.size() <= 0);

			// Sorting the 26 neighbourhood pixels in ascending order.
			std::sort(neighbourVoxels.begin(), neighbourVoxels.end());

			float median = 0;
			int length = neighbourVoxels.size();

			if (length > 0)
			{
				if (length % 2 == 0)
				{
					int mid = length / 2;

					median = (neighbourVoxels[mid] + neighbourVoxels[mid - 1]) / 2;
				}
				else
				{
					median = neighbourVoxels[(int)length / 2];
				}
			}
			
			unsigned char* finalPixel = static_cast<unsigned char*>(i->GetScalarPointer(x, y, z));
			*finalPixel++ = (unsigned char)((int)median % twelveBitInt);
			*finalPixel++ = (unsigned char)((int)median / twelveBitInt);
		}
	}
}

// Perform PNN Olympic operation on input volume i on axial slice z with total removal percentage p1 
// (upper + lower removal percentage) and maximum kernel size p2.
// Input volume o is the original volume (with holes). 
// Input volume i is the hole-filled volume.
void pnnConventionalOlympicOperationOnAxialZ(vtkSmartPointer<vtkImageData> i, vtkSmartPointer<vtkImageData> o, int z, float p1, float p2)
{
	const int kernelSize = (int)p2;
	const float removalPercentage = ((float)p1 / 2);
	int* dims = i->GetDimensions();
	std::vector<int> allRangeWidth;

	cout << "Running conventional olympic operation ... " << endl
		 << "Removal Percentage: upper = lower = " << removalPercentage << endl
		 << "Kernel size = " << kernelSize << endl;

	for (int y = 1; y < (dims[1] - 1); y++)
	{
		for (int x = 1; x < (dims[0] - 1); x++)
		{
			int temp = 0, kernel = 1;
			float estimatedVoxel = .0;
			std::vector<int> neighbourPixels;
			std::vector<int> sortedPixels;
			std::vector<int> neighbourVoxels;
			std::vector<int> sortedVoxels;

			do {
				neighbourVoxels = getNeighbourVoxels(o, x, y, z, kernel);

				if (neighbourVoxels.size() <= 0) kernel++;
				if (kernel >= kernelSize) break;

			} while (neighbourVoxels.size() <= 0);

			if (neighbourVoxels.size() > 0)
			{
				int lower = round(neighbourVoxels.size() * removalPercentage / 100);
				int upper = neighbourVoxels.size() - lower - 1;

				std::sort(neighbourVoxels.begin(), neighbourVoxels.end());
				std::copy(neighbourVoxels.begin().operator+(lower), neighbourVoxels.begin().operator+(upper + 1), back_inserter(sortedVoxels));

				estimatedVoxel = averageOperation(sortedVoxels);
			}

			unsigned char* finalPixel = static_cast<unsigned char*>(i->GetScalarPointer(x, y, z));
			*finalPixel++ = (unsigned char)((int)estimatedVoxel % twelveBitInt);
			*finalPixel++ = (unsigned char)((int)estimatedVoxel / twelveBitInt);
		}
	}
}

// Get the list of stick with minimum length.
std::vector<int> getMinimumStickLengthIndex(std::vector<float> vectSL)
{
	float min = .0;
	int length = vectSL.size();
	std::vector<float> vectSL2(vectSL);
	std::vector<int> vectMinSLIndex;

	// Get the minimum stick length from the empty voxel.
	std::sort(vectSL2.begin(), vectSL2.end());
	min = ((float)vectSL2[0]);

	// Get the index(s) of minimum stick length.
	for (int i = 0; i < length; i++)
	{
		if (vectSL[i] == min)
		{
			vectMinSLIndex.push_back(i);
		}
	}

	return vectMinSLIndex;
}

// Perform PNN oriented sticks method on input volume i on axial slice z with maximum kernel size p1.
// Input volume o is the original volume (with holes). 
// Input volume i is the hole-filled volume.
void pnnSticksHoleFillingOnAxialZ(vtkSmartPointer<vtkImageData> i, vtkSmartPointer<vtkImageData> o, int z, float p1)
{
	const int maxKernelSize = (int)p1;
	int* dims = i->GetDimensions();
	float r = 0.5;						// Linear interpolation parameter.

	cout << "Running sticks hole-filling method ... " << endl
		 << "Maximum kernel size = " << maxKernelSize << endl;

	// 1. Get empty voxels.
	for (int y = 1; y < (dims[1] - 1); y++)
	{
		for (int x = 1; x < (dims[0] - 1); x++)
		{
			// 2. Initialize the 9 stick intensities.
			std::vector<float> stickIntensities;
			std::vector<float> stickLengths;
			unsigned char* pixel;
			int d0 = 0, d1 = 0;
			float stickIntensity = .0, stickLength = .0, estimatedVoxel = .0;

			// 3. Get each stick and perform linear interpolation to get the stick intensity.
			bool breakLoop = false;
			for (int c = -1; c <= 1; c++)
			{
				for (int b = -1; b <= 1; b++)
				{
					for (int a = -1; a <= 1; a++)
					{
						if ((x + a) == x && (y + b) == y && (z + c) == z)
						{
							breakLoop = true;
							break;
						}

						int k = 1;
						int k1 = 1, k2 = 1;

						while (k1 <= maxKernelSize && k2 <= maxKernelSize)
						{
							int pt1[3] = { (x + (a * k1)), (y + (b * k1)), (z + (c * k1)) };
							int pt2[3] = { (x - (a * k2)), (y - (b * k2)), (z - (c * k2)) };

							if (pt1[0] < 0 || pt1[0] > dims[0] - 1 || pt1[1] < 0 || pt1[1] > dims[1] - 1 ||
								pt1[2] < 0 || pt1[2] > dims[2] - 1 || pt2[0] < 0 || pt2[0] > dims[0] - 1 ||
								pt2[1] < 0 || pt2[1] > dims[1] - 1 || pt2[2] < 0 || pt2[2] > dims[2] - 1)
								break;

							pixel = static_cast<unsigned char*>(o->GetScalarPointer(pt1[0], pt1[1], pt1[2]));
							d0 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
							pixel = static_cast<unsigned char*>(o->GetScalarPointer(pt2[0], pt2[1], pt2[2]));
							d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

							if (d0 == 0) k1++;
							if (d1 == 0) k2++;
							if (d0 != 0 && d1 != 0)
							{
								r = (float)k1 / (k1 + k2);
								stickIntensity = (d0 * (1 - r)) + (d1 * r);
								stickLength = sqrt(pow((pt2[0] - pt1[0]), 2) + pow((pt2[1] - pt1[1]), 2) + pow((pt2[2] - pt1[2]), 2));
								stickIntensities.push_back(stickIntensity);
								stickLengths.push_back(stickLength);
								break;
							}
						}
					}

					if (breakLoop) break;
				}

				if (breakLoop) break;
			}

			// 4. With the stick intensities, perform the stick hole-filling method and put the final result in the empty voxel.
			float totalTop = .0, totalBottom = .0;
			int length = stickIntensities.size();

			if (length > 0)
			{
				std::vector<int> minimumStickLengthIndexes = getMinimumStickLengthIndex(stickLengths);

				for (int i = 0; i < length; i++)
				{
					// Get only the stick(s) that has/have minimum length.
					if (std::find(minimumStickLengthIndexes.begin(), minimumStickLengthIndexes.end(), i) != minimumStickLengthIndexes.end())
					{
						totalTop += (stickIntensities[i] / stickLengths[i]);
						totalBottom += (1 / stickLengths[i]);
					}
				}
			}

			estimatedVoxel = totalTop / totalBottom;

			unsigned char* finalPixel = static_cast<unsigned char*>(i->GetScalarPointer(x, y, z));
			*finalPixel++ = (unsigned char)((int)estimatedVoxel % twelveBitInt);
			*finalPixel++ = (unsigned char)((int)estimatedVoxel / twelveBitInt);
		}
	}
}

// Vertices as in the Zorin et al.'s butterfly stencil map.
struct ButterflyVertices 
{
	int a1, a2, b1, b2, c1, c2, c3, c4, d1, d2, k1, k2, x1, y1, z1;
};

void printButterflyVertices(ButterflyVertices bv)
{
	cout << bv.a1 << " " << bv.a2 << " " << bv.b1 << " " << bv.b2 << " "
		 << bv.c1 << " " << bv.c2 << " " << bv.c3 << " " << bv.c4 << " "
		 << bv.d1 << " " << bv.d2 << " " << endl;
}

// Calculate the butterfly intensity as in Zorin et al.'s method.
float calculateButterflyIntensity(std::vector<int> nPs, int a)
{
	int valence = nPs.size();
	float butterflyIntensity = .0;

	if (valence == 3)
	{
		butterflyIntensity = (a * 0.75) + (nPs[0] * 0.4166666666667) - (nPs[1] * 0.0833333333333) - (nPs[2] * 0.0833333333333);
	}
	else if (valence == 4)
	{
		butterflyIntensity = (a * 0.75) + (nPs[0] * 0.375) + (nPs[1] * 0) - (nPs[2] * 0.125) + (nPs[3] * 0);
	}
	else if (valence == 5)
	{
		float v = (a * 0.75);
		float b = .0;

		for (int j = 0; j < valence; j++)
		{
			float sj = (0.25 + cos(2 * pi * j / valence) + (0.5)*(cos(4 * pi * j / valence))) / valence;
			b += (float)nPs[j] * sj;
		}

		butterflyIntensity = v + b;
	}

	return butterflyIntensity;
}

int getButterflyCaseVertexSize(ButterflyVertices b)
{
	int count = 0;

	if (b.a1 > 0) count++;
	if (b.a2 > 0) count++;
	if (b.b1 > 0) count++;
	if (b.b2 > 0) count++;
	if (b.c1 > 0) count++;
	if (b.c2 > 0) count++;
	if (b.c3 > 0) count++;
	if (b.c4 > 0) count++;
	if (b.d1 > 0) count++;
	if (b.d2 > 0) count++;

	return count;
}

std::vector<int> getButterflyCaseVertices(ButterflyVertices b)
{
	std::vector<int> vertices;

	if (b.a1 > 0) vertices.push_back(b.a1);
	if (b.a2 > 0) vertices.push_back(b.a2);
	if (b.b1 > 0) vertices.push_back(b.b1);
	if (b.b2 > 0) vertices.push_back(b.b2);
	if (b.c1 > 0) vertices.push_back(b.c1);
	if (b.c2 > 0) vertices.push_back(b.c2);
	if (b.c3 > 0) vertices.push_back(b.c3);
	if (b.c4 > 0) vertices.push_back(b.c4);
	if (b.d1 > 0) vertices.push_back(b.d1);
	if (b.d2 > 0) vertices.push_back(b.d2);

	return vertices;
}

int getButterflyConfigsMaxVertexSize(std::vector<ButterflyVertices> bVs)
{
	int max = 0;
	int length = bVs.size();
	
	for (int i = 0; i < length; i++)
	{
		int count = getButterflyCaseVertexSize(bVs[i]);

		if (count > max) max = count;
	}

	return max;
}

int getButterflyConfigsMinRangeWidth(std::vector<ButterflyVertices> bVs)
{
	int min = 0;
	int length = bVs.size();

	for (int i = 0; i < length; i++)
	{
		int rangeWidth = findRangeWidth2(getButterflyCaseVertices(bVs[i]));

		if (i == 0) min = rangeWidth;
		else if (rangeWidth < min) min = rangeWidth;
	}

	return min;
}

// My modified butterfly interpolation scheme for PNN hole-filling method of 3D regular volume grid.
// Perform PNN modified butterfly interpolation scheme on input volume i on axial slice z with 
// maximum kernel size p1.
// Input volume o is the original volume (with holes). 
// Input volume i is the hole-filled volume.
// p1 is the butterfy weight.
// p2 is the maximum length of butterfly edge.
// p3 is the maximum neighbour interpolation distance.
void pnnMyModifiedButterflyInterpolationOnAxialZ(vtkSmartPointer<vtkImageData> i, vtkSmartPointer<vtkImageData> o, int z, float p1, float p2, float p3)
{
	const int maxKernelSize = (int)p2;
	int* dims = i->GetDimensions();

	cout << "Running my modified butterfly interpolation scheme ... " << endl
		 << "Slice = " << (z + 1) << endl
		 << "Parameter w = " << p1 << endl
		 << "Maximum kernel size = " << maxKernelSize << endl;

	for (int y = 1; y < (dims[1] - 1); y++)
	{
		for (int x = 1; x < (dims[0] - 1); x++)
		{
			unsigned char* pixel;
			int d0 = 0, d1 = 0;
			float distance = .0;
			float stickLength = .0;
			std::vector<float> stickLengths;
			std::vector<float> distBetweenEmptyA1;
			std::vector<float> ws;
			std::vector<ButterflyVertices> butterflyConfigs;

			ButterflyVertices butterflyCase;
			int k = 1;
			int k1 = 1, k2 = 1;

			// Config 1 & Config 2.
			while (k1 <= maxKernelSize && k2 <= maxKernelSize)
			{
				if ((z - k1) < 0 || (z + k2) > dims[2] - 1)
					break;

				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y, z - k1));
				d0 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y, z + k2));
				d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

				if (d0 == 0) k1++;
				if (d1 == 0) k2++;
				if (d0 != 0 && d1 != 0)
				{
					// Get the k1 (a1-empty) and k2 (a2-empty).
					butterflyCase.k1 = k1;
					butterflyCase.k2 = k2;

					// Get the coordinate of a1.
					butterflyCase.x1 = x;
					butterflyCase.y1 = y;
					butterflyCase.z1 = z - k1;

					// a1, a2.
					butterflyCase.a1 = d0;
					butterflyCase.a2 = d1;

					// d1, d2.
					if ((z - (k1 + 1)) < 0) butterflyCase.d1 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y, z - (k1 + 1)));
						butterflyCase.d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					if ((z + (k2 + 1)) > (dims[2] - 1)) butterflyCase.d2 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y, z + (k2 + 1)));
						butterflyCase.d2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					// Calculate distance from empty voxel to voxel a1.
					distance = sqrt(pow(x - x, 2) + pow(y - y, 2) + pow(z - (z - k1), 2));

					// Calculate the stick length.
					stickLength = sqrt(pow(((z - k1) - (z + k2)), 2));

					// b1, b2. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - 1, y, z));
					butterflyCase.b1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + 1, y, z));
					butterflyCase.b2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// c1, c2, c3, c4.
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - 1, y, z - k1));
					butterflyCase.c1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + 1, y, z - k1));
					butterflyCase.c2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - 1, y, z + k2));
					butterflyCase.c3 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + 1, y, z + k2));
					butterflyCase.c4 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// Config 1.
					butterflyConfigs.push_back(butterflyCase);
					distBetweenEmptyA1.push_back(distance);
					stickLengths.push_back(stickLength);

					// b1, b2. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - 1, z));
					butterflyCase.b1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + 1, z));
					butterflyCase.b2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// c1, c2, c3, c4.
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - 1, z - k1));
					butterflyCase.c1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + 1, z - k1));
					butterflyCase.c2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - 1, z + k2));
					butterflyCase.c3 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + 1, z + k2));
					butterflyCase.c4 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// Config 2.
					butterflyConfigs.push_back(butterflyCase);
					distBetweenEmptyA1.push_back(distance);
					stickLengths.push_back(stickLength);

					break;
				}
			}

			// Config 3 & Config 4.
			k = 1;
			k1 = 1;
			k2 = 1;

			while (k1 <= maxKernelSize && k2 <= maxKernelSize)
			{
				if ((x - k1) < 0 || (x + k2) > dims[0] - 1)
					break;

				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y, z));
				d0 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y, z));
				d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

				if (d0 == 0) k1++;
				if (d1 == 0) k2++;
				if (d0 != 0 && d1 != 0)
				{
					// Get the k1 (a1-empty) and k2 (a2-empty).
					butterflyCase.k1 = k1;
					butterflyCase.k2 = k2;

					// Get the coordinate of a1.
					butterflyCase.x1 = x - k1;
					butterflyCase.y1 = y;
					butterflyCase.z1 = z;

					// a1, a2.
					butterflyCase.a1 = d0;
					butterflyCase.a2 = d1;

					// d1, d2.
					if ((x - (k1 + 1)) < 0) butterflyCase.d1 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - (k1 + 1), y, z));
						butterflyCase.d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					if ((x + (k2 + 1)) > (dims[0] - 1)) butterflyCase.d2 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + (k2 + 1), y, z));
						butterflyCase.d2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					// Calculate distance from empty voxel to voxel a1.
					distance = sqrt(pow(x - (x - k1), 2) + pow(y - y, 2) + pow(z - z, 2));

					// Calculate the stick length.
					stickLength = sqrt(pow(((x - k1) - (x + k2)), 2));

					// b1, b2. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y, z - 1));
					butterflyCase.b1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y, z + 1));
					butterflyCase.b2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// c1, c2, c3, c4.
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y, z - 1));
					butterflyCase.c1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y, z + 1));
					butterflyCase.c2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y, z - 1));
					butterflyCase.c3 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y, z + 1));
					butterflyCase.c4 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// Config 3.
					butterflyConfigs.push_back(butterflyCase);
					distBetweenEmptyA1.push_back(distance);
					stickLengths.push_back(stickLength);

					// b1, b2. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - 1, z));
					butterflyCase.b1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + 1, z));
					butterflyCase.b2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// c1, c2, c3, c4. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y - 1, z));
					butterflyCase.c1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y + 1, z));
					butterflyCase.c2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y - 1, z));
					butterflyCase.c3 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y + 1, z));
					butterflyCase.c4 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// Config 4.
					butterflyConfigs.push_back(butterflyCase);
					distBetweenEmptyA1.push_back(distance);
					stickLengths.push_back(stickLength);

					break;
				}
			}

			// Config 5 & Config 6.
			k = 1;
			k1 = 1;
			k2 = 1;

			while (k1 <= maxKernelSize && k2 <= maxKernelSize)
			{
				if ((y - k1) < 0 || (y + k2) > dims[1] - 1)
					break;

				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - k1, z));
				d0 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + k2, z));
				d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

				if (d0 == 0) k1++;
				if (d1 == 0) k2++;
				if (d0 != 0 && d1 != 0)
				{
					// Get the k1 (a1-empty) and k2 (a2-empty).
					butterflyCase.k1 = k1;
					butterflyCase.k2 = k2;

					// Get the coordinate of a1.
					butterflyCase.x1 = x;
					butterflyCase.y1 = y - k1;
					butterflyCase.z1 = z;

					// a1, a2.
					butterflyCase.a1 = d0;
					butterflyCase.a2 = d1;

					// d1, d2.
					if ((y - (k1 + 1)) < 0) butterflyCase.d1 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - (k1 + 1), z));
						butterflyCase.d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					if ((y + (k2 + 1)) > (dims[1] - 1)) butterflyCase.d2 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + (k2 + 1), z));
						butterflyCase.d2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					// Calculate distance from empty voxel to voxel a1.
					distance = sqrt(pow(x - x, 2) + pow(y - (y - k1), 2) + pow(z - z, 2));

					// Calculate the stick length.
					stickLength = sqrt(pow(((y - k1) - (y + k2)), 2));

					// b1, b2. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y, z - 1));
					butterflyCase.b1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y, z + 1));
					butterflyCase.b2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// c1, c2, c3, c4.
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - k1, z - 1));
					butterflyCase.c1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - k1, z + 1));
					butterflyCase.c2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + k2, z - 1));
					butterflyCase.c3 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + k2, z + 1));
					butterflyCase.c4 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// Config 5.
					butterflyConfigs.push_back(butterflyCase);
					distBetweenEmptyA1.push_back(distance);
					stickLengths.push_back(stickLength);

					// b1, b2. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - 1, y, z));
					butterflyCase.b1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + 1, y, z));
					butterflyCase.b2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// c1, c2, c3, c4. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - 1, y - k1, z));
					butterflyCase.c1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + 1, y - k1, z));
					butterflyCase.c2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - 1, y + k2, z));
					butterflyCase.c3 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + 1, y + k2, z));
					butterflyCase.c4 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// Config 6.
					butterflyConfigs.push_back(butterflyCase);
					distBetweenEmptyA1.push_back(distance);
					stickLengths.push_back(stickLength);
					
					break;
				}
			}

			// Config 7.
			k = 1;
			k1 = 1;
			k2 = 1;

			while (k1 <= maxKernelSize && k2 <= maxKernelSize)
			{
				if ((y - k2) < 0 || (y + k1) > dims[1] - 1 || (z - k1) < 0 || (z + k2) > dims[2] - 1)
					break;

				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + k1, z - k1));
				d0 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - k2, z + k2));
				d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

				if (d0 == 0) k1++;
				if (d1 == 0) k2++;
				if (d0 != 0 && d1 != 0)
				{
					// Get the k1 (a1-empty) and k2 (a2-empty).
					butterflyCase.k1 = k1;
					butterflyCase.k2 = k2;

					// Get the coordinate of a1.
					butterflyCase.x1 = x;
					butterflyCase.y1 = y + k1;
					butterflyCase.z1 = z - k1;

					// a1, a2.
					butterflyCase.a1 = d0;
					butterflyCase.a2 = d1;

					// b1, b2. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - 1, y, z));
					butterflyCase.b1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + 1, y, z));
					butterflyCase.b2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// c1, c2, c3, c4.
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - 1, y + k1, z - k1));
					butterflyCase.c1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + 1, y + k1, z - k1));
					butterflyCase.c2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - 1, y - k2, z + k2));
					butterflyCase.c3 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + 1, y - k2, z + k2));
					butterflyCase.c4 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// d1, d2.
					if ((y + (k1 + 1)) > (dims[1] - 1) || (z - (k1 + 1)) < 0) butterflyCase.d1 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + (k1 + 1), z - (k1 + 1)));
						butterflyCase.d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					if ((y - (k2 + 1)) < 0 || (z + (k2 + 1)) > (dims[2] - 1)) butterflyCase.d2 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - (k2 + 1), z + (k2 + 1)));
						butterflyCase.d2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					// Config 7.
					butterflyConfigs.push_back(butterflyCase);

					// Calculate distance from empty voxel to voxel a1.
					distance = sqrt(pow(x - x, 2) + pow(y - (y + k1), 2) + pow(z - (z - k1), 2));
					distBetweenEmptyA1.push_back(distance);

					// Calculate the stick length.
					stickLength = sqrt(pow(((y + k1) - (y - k2)), 2) + pow(((z - k1) - (z + k2)), 2));
					stickLengths.push_back(stickLength);

					break;
				}
			}

			// Config 8.
			k = 1;
			k1 = 1;
			k2 = 1;

			while (k1 <= maxKernelSize && k2 <= maxKernelSize)
			{
				if ((y - k1) < 0 || (y + k2) > dims[1] - 1 || (z - k1) < 0 || (z + k2) > dims[2] - 1)
					break;

				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - k1, z - k1));
				d0 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + k2, z + k2));
				d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

				if (d0 == 0) k1++;
				if (d1 == 0) k2++;
				if (d0 != 0 && d1 != 0)
				{
					// Get the k1 (a1-empty) and k2 (a2-empty).
					butterflyCase.k1 = k1;
					butterflyCase.k2 = k2;

					// Get the coordinate of a1.
					butterflyCase.x1 = x;
					butterflyCase.y1 = y + k1;
					butterflyCase.z1 = z - k1;

					// a1, a2.
					butterflyCase.a1 = d0;
					butterflyCase.a2 = d1;

					// b1, b2. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - 1, y, z));
					butterflyCase.b1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + 1, y, z));
					butterflyCase.b2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// c1, c2, c3, c4.
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - 1, y - k1, z - k1));
					butterflyCase.c1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + 1, y - k1, z - k1));
					butterflyCase.c2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - 1, y + k2, z + k2));
					butterflyCase.c3 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + 1, y + k2, z + k2));
					butterflyCase.c4 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// d1, d2.
					if ((y - (k1 + 1)) < 0 || (z - (k1 + 1)) < 0) butterflyCase.d1 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - (k1 + 1), z - (k1 + 1)));
						butterflyCase.d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					if ((y + (k2 + 1)) > (dims[1] - 1) || (z + (k2 + 1)) > (dims[2] - 1)) butterflyCase.d2 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + (k2 + 1), z + (k2 + 1)));
						butterflyCase.d2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					// Config 8.
					butterflyConfigs.push_back(butterflyCase);

					// Calculate distance from empty voxel to voxel a1.
					distance = sqrt(pow(x - x, 2) + pow(y - (y - k1), 2) + pow(z - (z - k1), 2));
					distBetweenEmptyA1.push_back(distance);

					// Calculate the stick length.
					stickLength = sqrt(pow(((y - k1) - (y + k2)), 2) + pow(((z - k1) - (z + k2)), 2));
					stickLengths.push_back(stickLength);

					break;
				}
			}

			// Config 9.
			k = 1;
			k1 = 1;
			k2 = 1;

			while (k1 <= maxKernelSize && k2 <= maxKernelSize)
			{
				if ((x - k1) < 0 || (x + k2) > dims[0] - 1 || (z - k1) < 0 || (z + k2) > dims[2] - 1)
					break;

				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y, z - k1));
				d0 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y, z + k2));
				d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

				if (d0 == 0) k1++;
				if (d1 == 0) k2++;
				if (d0 != 0 && d1 != 0)
				{
					// Get the k1 (a1-empty) and k2 (a2-empty).
					butterflyCase.k1 = k1;
					butterflyCase.k2 = k2;

					// Get the coordinate of a1.
					butterflyCase.x1 = x - k1;
					butterflyCase.y1 = y;
					butterflyCase.z1 = z - k1;

					// a1, a2.
					butterflyCase.a1 = d0;
					butterflyCase.a2 = d1;

					// b1, b2. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - 1, z));
					butterflyCase.b1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + 1, z));
					butterflyCase.b2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// c1, c2, c3, c4.
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y - 1, z - k1));
					butterflyCase.c1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y + 1, z - k1));
					butterflyCase.c2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y - 1, z + k2));
					butterflyCase.c3 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y + 1, z + k2));
					butterflyCase.c4 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// d1, d2.
					if ((x - (k1 + 1)) < 0 || (z - (k1 + 1)) < 0) butterflyCase.d1 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - (k1 + 1), y, z - (k1 + 1)));
						butterflyCase.d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					if ((x + (k2 + 1)) > (dims[0] - 1) || (z + (k2 + 1)) > (dims[2] - 1)) butterflyCase.d2 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + (k2 + 1), y, z + (k2 + 1)));
						butterflyCase.d2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					// Config 9.
					butterflyConfigs.push_back(butterflyCase);

					// Calculate distance from empty voxel to voxel a1.
					distance = sqrt(pow(x - (x - k1), 2) + pow(y - y, 2) + pow(z - (z - k1), 2));
					distBetweenEmptyA1.push_back(distance);

					// Calculate the stick length.
					stickLength = sqrt(pow(((x - k1) - (x + k2)), 2) + pow(((z - k1) - (z + k2)), 2));
					stickLengths.push_back(stickLength);

					break;
				}
			}

			// Config 10.
			k = 1;
			k1 = 1;
			k2 = 1;

			while (k1 <= maxKernelSize && k2 <= maxKernelSize)
			{
				if ((x - k2) < 0 || (x + k1) > dims[0] - 1 || (z - k1) < 0 || (z + k2) > dims[2] - 1)
					break;

				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k1, y, z - k1));
				d0 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k2, y, z + k2));
				d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

				if (d0 == 0) k1++;
				if (d1 == 0) k2++;
				if (d0 != 0 && d1 != 0)
				{
					// Get the k1 (a1-empty) and k2 (a2-empty).
					butterflyCase.k1 = k1;
					butterflyCase.k2 = k2;

					// Get the coordinate of a1.
					butterflyCase.x1 = x + k1;
					butterflyCase.y1 = y;
					butterflyCase.z1 = z - k1;

					// a1, a2.
					butterflyCase.a1 = d0;
					butterflyCase.a2 = d1;

					// b1, b2. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y - 1, z));
					butterflyCase.b1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y + 1, z));
					butterflyCase.b2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// c1, c2, c3, c4.
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k1, y - 1, z - k1));
					butterflyCase.c1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k1, y + 1, z - k1));
					butterflyCase.c2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k2, y - 1, z + k2));
					butterflyCase.c3 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k2, y + 1, z + k2));
					butterflyCase.c4 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// d1, d2.
					if ((x + (k1 + 1)) > (dims[0] - 1) || (z - (k1 + 1)) < 0) butterflyCase.d1 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + (k1 + 1), y, z - (k1 + 1)));
						butterflyCase.d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					if ((x - (k2 + 1)) < 0 || (z + (k2 + 1)) > (dims[2] - 1)) butterflyCase.d2 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - (k2 + 1), y, z + (k2 + 1)));
						butterflyCase.d2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					// Config 10.
					butterflyConfigs.push_back(butterflyCase);

					// Calculate distance from empty voxel to voxel a1.
					distance = sqrt(pow(x - (x + k1), 2) + pow(y - y, 2) + pow(z - (z - k1), 2));
					distBetweenEmptyA1.push_back(distance);

					// Calculate the stick length.
					stickLength = sqrt(pow(((x + k1) - (x - k2)), 2) + pow(((z - k1) - (z + k2)), 2));
					stickLengths.push_back(stickLength);

					break;
				}
			}

			// Config 11.
			k = 1;
			k1 = 1;
			k2 = 1;

			while (k1 <= maxKernelSize && k2 <= maxKernelSize)
			{
				if ((x - k1) < 0 || (x + k2) > dims[0] - 1 || (y - k1) < 0 || (y + k2) > dims[1] - 1)
					break;

				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y - k1, z));
				d0 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y + k2, z));
				d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

				if (d0 == 0) k1++;
				if (d1 == 0) k2++;
				if (d0 != 0 && d1 != 0)
				{
					// Get the k1 (a1-empty) and k2 (a2-empty).
					butterflyCase.k1 = k1;
					butterflyCase.k2 = k2;

					// Get the coordinate of a1.
					butterflyCase.x1 = x - k1;
					butterflyCase.y1 = y - k1;
					butterflyCase.z1 = z;

					// a1, a2.
					butterflyCase.a1 = d0;
					butterflyCase.a2 = d1;

					// b1, b2. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y, z - 1));
					butterflyCase.b1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y, z + 1));
					butterflyCase.b2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// c1, c2, c3, c4.
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y - k1, z - 1));
					butterflyCase.c1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y - k1, z + 1));
					butterflyCase.c2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y + k2, z - 1));
					butterflyCase.c3 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y + k2, z + 1));
					butterflyCase.c4 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// d1, d2.
					if ((x - (k1 + 1)) < 0 || (y - (k1 + 1)) < 0) butterflyCase.d1 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - (k1 + 1), y - (k1 + 1), z));
						butterflyCase.d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					if ((x + (k2 + 1)) > (dims[0] - 1) || (y + (k2 + 1)) > (dims[1] - 1)) butterflyCase.d2 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + (k2 + 1), y + (k2 + 1), z));
						butterflyCase.d2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					// Config 11.
					butterflyConfigs.push_back(butterflyCase);

					// Calculate distance from empty voxel to voxel a1.
					distance = sqrt(pow(x - (x - k1), 2) + pow(y - (y - k1), 2) + pow(z - z, 2));
					distBetweenEmptyA1.push_back(distance);

					// Calculate the stick length.
					stickLength = sqrt(pow(((x - k1) - (x + k2)), 2) + pow(((y - k1) - (y + k2)), 2));
					stickLengths.push_back(stickLength);

					break;
				}
			}

			// Config 12.
			k = 1;
			k1 = 1;
			k2 = 1;

			while (k1 <= maxKernelSize && k2 <= maxKernelSize)
			{
				if ((x - k1) < 0 || (x + k2) > dims[0] - 1 || (y - k2) < 0 || (y + k1) > dims[1] - 1)
					break;

				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y + k1, z));
				d0 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
				pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y - k2, z));
				d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

				if (d0 == 0) k1++;
				if (d1 == 0) k2++;
				if (d0 != 0 && d1 != 0)
				{
					// Get the k1 (a1-empty) and k2 (a2-empty).
					butterflyCase.k1 = k1;
					butterflyCase.k2 = k2;

					// Get the coordinate of a1.
					butterflyCase.x1 = x - k1;
					butterflyCase.y1 = y + k1;
					butterflyCase.z1 = z;

					// a1, a2.
					butterflyCase.a1 = d0;
					butterflyCase.a2 = d1;

					// b1, b2. 
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y, z - 1));
					butterflyCase.b1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x, y, z + 1));
					butterflyCase.b2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// c1, c2, c3, c4.
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y + k1, z - 1));
					butterflyCase.c1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - k1, y + k1, z + 1));
					butterflyCase.c2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y - k2, z - 1));
					butterflyCase.c3 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + k2, y - k2, z + 1));
					butterflyCase.c4 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);

					// d1, d2.
					if ((x - (k1 + 1)) < 0 || (y + (k1 + 1)) > (dims[1] - 1)) butterflyCase.d1 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x - (k1 + 1), y + (k1 + 1), z));
						butterflyCase.d1 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					if ((x + (k2 + 1)) > (dims[0] - 1) || (y - (k2 + 1)) < 0) butterflyCase.d2 = 0;
					else
					{
						pixel = static_cast<unsigned char*>(o->GetScalarPointer(x + (k2 + 1), y - (k2 + 1), z));
						butterflyCase.d2 = (((int)pixel[1]) * twelveBitInt) + ((int)pixel[0]);
					}

					// Config 12.
					butterflyConfigs.push_back(butterflyCase);

					// Calculate distance from empty voxel to voxel a1.
					distance = sqrt(pow(x - (x - k1), 2) + pow(y - (y + k1), 2) + pow(z - z, 2));
					distBetweenEmptyA1.push_back(distance);

					// Calculate the stick length.
					stickLength = sqrt(pow(((x - k1) - (x + k2)), 2) + pow(((y + k1) - (y - k2)), 2));
					stickLengths.push_back(stickLength);

					break;
				}
			}

			float totalTop = .0, totalBottom = .0, averageVoxels = .0;
			int length = butterflyConfigs.size();
			int count = 0;

			if (length > 0)
			{
				// <Archive> Get the maximum vertices in the Butterfly configs. 
				//int max = getButterflyConfigsMaxVertexSize(butterflyConfigs);

				// Get the minimum stick length.
				std::vector<int> minimumStickLengthIndexes = getMinimumStickLengthIndex(stickLengths);

				// <Archive> Get the minimum range width in the Butterfly configs.
				//int minRangeWidth = getButterflyConfigsMinRangeWidth(butterflyConfigs);

				for (int i = 0; i < length; i++)
				{
					// <Archive>
					//int rangeWidth = findRangeWidth2(getButterflyCaseVertices(butterflyConfigs[i]));
					//if(rangeWidth == minRangeWidth)

					// <Archive>
					//if (getButterflyCaseVertexSize(butterflyConfigs[i]) == max)

					// Get only the stick(s) that has/have minimum length.
					if (std::find(minimumStickLengthIndexes.begin(), minimumStickLengthIndexes.end(), i) != minimumStickLengthIndexes.end())
					{
						float butterflyIntensity = .0;
						int valenceA1, valenceA2;
						std::vector<int> neighbourA1;
						std::vector<int> neighbourA2;

						// Get b1 and b2 by simple linear interpolation.
						if ((butterflyConfigs[i].k1 + butterflyConfigs[i].k2) <= p3)
						{
							if (butterflyConfigs[i].b1 <= 0 && butterflyConfigs[i].c1 > 0 && butterflyConfigs[i].c3 > 0)
							{
								float r = 0.5;
								butterflyConfigs[i].b1 = (butterflyConfigs[i].c1 * (1 - r)) + (butterflyConfigs[i].c3 * r);
							}
							if (butterflyConfigs[i].b2 <= 0 && butterflyConfigs[i].c2 > 0 && butterflyConfigs[i].c4 > 0)
							{
								float r = 0.5;
								butterflyConfigs[i].b2 = (butterflyConfigs[i].c2 * (1 - r)) + (butterflyConfigs[i].c4 * r);
							}
						}

						// 1. Based on a1, find valenceA1: a2 - b1 - c1 - d1 - c2 - b2.
						neighbourA1.push_back(butterflyConfigs[i].a2);
						if (butterflyConfigs[i].b1 > 0) neighbourA1.push_back(butterflyConfigs[i].b1);
						if (butterflyConfigs[i].c1 > 0) neighbourA1.push_back(butterflyConfigs[i].c1);
						if (butterflyConfigs[i].d1 > 0) neighbourA1.push_back(butterflyConfigs[i].d1);
						if (butterflyConfigs[i].c2 > 0) neighbourA1.push_back(butterflyConfigs[i].c2);
						if (butterflyConfigs[i].b2 > 0) neighbourA1.push_back(butterflyConfigs[i].b2);
						valenceA1 = neighbourA1.size();

						// 2. Based on a2, find valenceA2: a1 - b1 - c3 - d2 - c4 - b2.
						neighbourA2.push_back(butterflyConfigs[i].a1);
						if (butterflyConfigs[i].b1 > 0) neighbourA2.push_back(butterflyConfigs[i].b1);
						if (butterflyConfigs[i].c3 > 0) neighbourA2.push_back(butterflyConfigs[i].c3);
						if (butterflyConfigs[i].d2 > 0) neighbourA2.push_back(butterflyConfigs[i].d2);
						if (butterflyConfigs[i].c4 > 0) neighbourA2.push_back(butterflyConfigs[i].c4);
						if (butterflyConfigs[i].b2 > 0) neighbourA2.push_back(butterflyConfigs[i].b2);
						valenceA2 = neighbourA2.size();

						// Performing Butterfly cases:
						if ((valenceA1 == 6 && valenceA2 == 6))
						{
							float w = p1;
							float a = ((float)1.0 / 2.0) - w;
							float b = ((float)1.0 / 8.0) + (2.0 * w);
							float c = -((float)1.0 / 16.0) - w;
							float d = w;

							// 3. Case 1: If both valenceA1 and valenceA2 are 6:
							butterflyIntensity = (butterflyConfigs[i].a1 * a) + (butterflyConfigs[i].a2 * a) +
								(butterflyConfigs[i].b1 * b) + (butterflyConfigs[i].b2 * b) +
								(butterflyConfigs[i].c1 * c) + (butterflyConfigs[i].c2 * c) +
								(butterflyConfigs[i].c3 * c) + (butterflyConfigs[i].c4 * c) +
								(butterflyConfigs[i].d1 * d) + (butterflyConfigs[i].d2 * d);
						}
						else if (valenceA1 != 6 && valenceA2 == 6 && valenceA1 >= 3)
						{
							// 4. Case 2(a): If valenceA2 is 6 and valenceA1 is extraordinary points:
							butterflyIntensity = calculateButterflyIntensity(neighbourA1, butterflyConfigs[i].a1);
						}
						else if (valenceA1 == 6 && valenceA2 != 6 && valenceA2 >= 3)
						{
							// 4. Case 2(b): If valenceA1 is 6 and valenceA2 is extraordinary points:
							butterflyIntensity = calculateButterflyIntensity(neighbourA2, butterflyConfigs[i].a2);
						}
						else if (valenceA1 != 6 && valenceA2 != 6 && valenceA1 >= 3 && valenceA2 >= 3)
						{
							// Here is the problem part!!!
							// 5. Case 3: If both valenceA1 and valenceA2 are not 6:
							float b1 = calculateButterflyIntensity(neighbourA1, butterflyConfigs[i].a1);
							float b2 = calculateButterflyIntensity(neighbourA2, butterflyConfigs[i].a2);

							if (b1 >= 0 && b2 >= 0)
								butterflyIntensity = (b1 + b2) / 2;
							else
								butterflyIntensity = -1;
						}
						else
						{
							// Compute using linear interpolation, if there are valence = 2.
							butterflyIntensity = -1;
						}

						float estimatedVoxel = .0;

						// Perform linear interpolation only if butterflyIntensity is < 0.
						if (butterflyIntensity < 0)
						{ 
							//printButterflyVertices(butterflyConfigs[i]);
							float r = (float)butterflyConfigs[i].k1 / (butterflyConfigs[i].k1 + butterflyConfigs[i].k2);
							estimatedVoxel = (butterflyConfigs[i].a1 * (1 - r)) + (butterflyConfigs[i].a2 * r);
						}

						// Perform quadratic interpolation only if buttereflyIntensity is >= 0.
						else
						{
							// If k1 == k2 is the same:
							// Midpoint (butterflyIntensity) is the empty voxel.
							if (butterflyConfigs[i].k1 == butterflyConfigs[i].k2)
								estimatedVoxel = butterflyIntensity;
							else
							{
								// If k1 != k2: Quadratic interpolation.
								// Midpoint (butterflyIntensity) is not the empty voxel.
								float r = (float)butterflyConfigs[i].k1 / (butterflyConfigs[i].k1 + butterflyConfigs[i].k2);
								float lengthL = (float)stickLengths[i];
								float lengthB = (float)stickLengths[i] / 2;
								float m0 = (float)butterflyConfigs[i].a1;
								float m1 = (4 * butterflyIntensity) - (3 * butterflyConfigs[i].a1) - butterflyConfigs[i].a2;
								float m2 = (2 * butterflyConfigs[i].a1) - (4 * butterflyIntensity) + (2 * butterflyConfigs[i].a2);

								estimatedVoxel = (m2 * pow(r, 2)) + (m1 * r) + m0;
							}	
						}

						if (estimatedVoxel < 0)
						{
							estimatedVoxel = 0;
						}
						else if (estimatedVoxel > 4095)
						{
							estimatedVoxel = 4095;
						}

						totalTop += (estimatedVoxel / stickLengths[i]);
						totalBottom += ((float)1 / stickLengths[i]);
					}
				}

				// Averaged the distance weighted.
				averageVoxels = (float)totalTop / totalBottom;
			}

			unsigned char* finalPixel = static_cast<unsigned char*>(i->GetScalarPointer(x, y, z));
			*finalPixel++ = (unsigned char)((int)averageVoxels % twelveBitInt);
			*finalPixel++ = (unsigned char)((int)averageVoxels / twelveBitInt);
		}
	}
}

int main(int argc, char *argv[])
{
	int sliceNo = 0;
	int incrementSparsity = 0;
	int incrementLimit = 0;
	std::string datasetName = "";
	std::string method = "original";
	float parameter1 = .0, parameter2 = .0, parameter3 = .0;
	float averageRangeWidth = .0;

	if (argc < 3)
	{
		cout << "Usage: " << argv[0] << " *outputDatasetName(string) *sliceNo(int) method(string) parameter(float)" << endl
			 << "The methods are: mean, median, olympic-con, sticks, butterfly-my" << endl;
		return EXIT_FAILURE;
	}
	else if (argc >= 3)
	{
		datasetName = argv[1];
		sliceNo = atoi(argv[2]);

		if (argc > 3)
		{
			method = argv[3];

			if (strcmp(method.c_str(), "mean") == 0 || strcmp(method.c_str(), "median") == 0 ||
				strcmp(method.c_str(), "olympic-con") == 0 || strcmp(method.c_str(), "sticks") == 0 || 
				strcmp(method.c_str(), "butterfly-my") == 0)
			{
				if (argc < 5)
				{
					cout << "Usage: " << argv[0] << " *datasetName(string) *sliceNo(int) *method(string) *parameter(int, float)" << endl
						 << "The " << method << " method requires a parameter." << endl
						 << "{ Mean, Median = kernel size (recom: 3) }" << endl
						 << "{ Olympic = kernel size & removal percentage (recom: 20) }" << endl 
						 << "{ Sticks = maximum stick length (recom: 3) }" << endl 
						 << "{ Butterfly = w-parameter (recom: 0) & maximum stick length (recom: 3) & max neighbour interpolation distance (recom: 0) }" << endl;
					return EXIT_FAILURE;
				}
				else
				{
					parameter1 = atof(argv[4]);

					if(argc >= 6)
						parameter2 = atof(argv[5]);

					if (argc >= 7)
						parameter3 = atof(argv[6]);
				}
			}
		}
	}

	cout << "Data input: " << argv[0] << " " << datasetName << " " << sliceNo << " " << method << " " << parameter1 << endl;

	vtkSmartPointer<vtkNamedColors> colors =
		vtkSmartPointer<vtkNamedColors>::New();

	// Set the colors.
	std::array<unsigned char, 4> skinColor{ { 255, 125, 64 } };
	colors->SetColor("SkinColor", skinColor.data());
	std::array<unsigned char, 4> bkg{ { 51, 77, 102, 255 } };
	colors->SetColor("BkgColor", bkg.data());

	// Create the renderer, the render window, and the interactor. The renderer
	// draws into the render window, the interactor enables mouse- and
	// keyboard-based interaction with the data within the render window.
	vtkSmartPointer<vtkRenderer> aRenderer =
		vtkSmartPointer<vtkRenderer>::New();
	vtkSmartPointer<vtkRenderWindow> renWin =
		vtkSmartPointer<vtkRenderWindow>::New();
	renWin->AddRenderer(aRenderer);

	vtkSmartPointer<vtkRenderWindowInteractor> iren =
		vtkSmartPointer<vtkRenderWindowInteractor>::New();
	iren->SetRenderWindow(renWin);

	vtkSmartPointer<vtkVolume16Reader> v16 = vtkSmartPointer<vtkVolume16Reader>::New();
	v16->SetDataDimensions(256, 256);
	v16->SetImageRange(1, 93);
	v16->SetDataByteOrderToLittleEndian();
	v16->SetFilePrefix("fullHeadRaw/");
	v16->SetFilePattern("%sheadsq%d.raw");
	v16->SetDataSpacing(0.9375, 0.9375, 1.5);
	v16->Update();

	vtkSmartPointer<vtkImageData> imageData =
		vtkSmartPointer<vtkImageData>::New();
	imageData->GetPointData()->CopyAllOn();
	imageData->GetPointData()->CopyScalarsOn();
	imageData->DeepCopy(v16->GetOutput());

	vtkSmartPointer<vtkImageData> oriImageData =
		vtkSmartPointer<vtkImageData>::New();

	std::cout << "Image Data Object Type: " << imageData->GetDataObjectType() << std::endl;
	std::cout << "Number of Scalar Components: " << imageData->GetNumberOfScalarComponents() << std::endl;
	std::cout << "Scalar Type: " << imageData->GetScalarTypeAsString() << std::endl;
	std::cout << "Scalar Size: " << imageData->GetScalarSize() << std::endl;

	// NOTE: Access image data here ...
	int* dims = imageData->GetDimensions();
	int z = sliceNo - 1;
	int numSlice = 1;
	int offsetSlice = 0;

	do {
		std::cout << "Number of slice = ";
		std::cin >> numSlice;

		if (numSlice <= 0)
			std::cout << "Please input number greater than 0.";
	} while (numSlice <= 0);

	do {
		std::cout << "Increment sparsity = ";
		std::cin >> incrementSparsity;

		if (incrementSparsity < 0)
			std::cout << "Please input number greater than or equal to 0.";
	} while (incrementSparsity < 0);

	do {
		std::cout << "Increment limit = ";
		std::cin >> incrementLimit;

		if (incrementLimit <= 0)
			std::cout << "Please input number greater than 0.";
	} while (incrementLimit <= 0);

	if (numSlice == 1) offsetSlice = 0;
	else if (numSlice == 3) offsetSlice = 1;

	if (argc > 3)
	{
		// Clean the data first.
		for (int j = 0; j < incrementLimit; j++)
		{
			z = sliceNo + (incrementSparsity * j) - 1;

			for (int i = 0; i < numSlice; i++)
				cleanImageDataOnAxialZ(imageData, z - offsetSlice + i);
		}

		// Copy the bin-filled data.
		oriImageData->DeepCopy(imageData);

		for (int j = 0; j < incrementLimit; j++)
		{
			z = sliceNo + (incrementSparsity * j) - 1;

			/*for (int i = 0; i < numSlice; i++)
				cleanImageDataOnAxialZ(imageData, z - offsetSlice + i);*/

			if (strcmp(method.c_str(), "mean") == 0)
			{
				// PNN by using Mean operation.
				for (int i = 0; i < numSlice; i++)
					pnnMeanOperationOnAxialZ(imageData, oriImageData, z - offsetSlice + i, parameter1);
			}
			else if (strcmp(method.c_str(), "median") == 0)
			{
				// PNN by using Median operation.
				for (int i = 0; i < numSlice; i++)
					pnnMedianOperationOnAxialZ(imageData, oriImageData, z - offsetSlice + i, parameter1);
			}
			else if (strcmp(method.c_str(), "olympic-con") == 0)
			{
				if (parameter2 <= 0)
					parameter2 = 1;

				// PNN by using Olympic operation.
				for (int i = 0; i < numSlice; i++)
					pnnConventionalOlympicOperationOnAxialZ(imageData, oriImageData, z - offsetSlice + i, parameter1, parameter2);
			}
			else if (strcmp(method.c_str(), "sticks") == 0)
			{
				// PNN by using Sticks interpolation.
				for (int i = 0; i < numSlice; i++)
					pnnSticksHoleFillingOnAxialZ(imageData, oriImageData, z - offsetSlice + i, parameter1);
			}
			else if (strcmp(method.c_str(), "butterfly-my") == 0)
			{
				if (parameter2 <= 0)
					parameter2 = 1;

				// PNN by using my Modified Butterfly interpolation.
				for (int i = 0; i < numSlice; i++)
					pnnMyModifiedButterflyInterpolationOnAxialZ(imageData, oriImageData, z - offsetSlice + i, parameter1, parameter2, parameter3);
			}
		}
	}

	// An isosurface, or contour value of 500 is known to correspond to the
	// skin of the patient.
	// The triangle stripper is used to create triangle strips from the
	// isosurface; these render much faster on many systems.
	vtkSmartPointer<vtkMarchingCubes> skinExtractor =
		vtkSmartPointer<vtkMarchingCubes>::New();
	skinExtractor->SetInputData(imageData);
	skinExtractor->SetValue(0, 500);

	vtkSmartPointer<vtkStripper> skinStripper =
		vtkSmartPointer<vtkStripper>::New();
	skinStripper->SetInputConnection(skinExtractor->GetOutputPort());

	vtkSmartPointer<vtkPolyDataMapper> skinMapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
	skinMapper->SetInputConnection(skinStripper->GetOutputPort());
	skinMapper->ScalarVisibilityOff();

	vtkSmartPointer<vtkActor> skin =
		vtkSmartPointer<vtkActor>::New();
	skin->SetMapper(skinMapper);
	skin->GetProperty()->SetDiffuseColor(colors->GetColor3d("SkinColor").GetData());
	skin->GetProperty()->SetSpecular(.3);
	skin->GetProperty()->SetSpecularPower(20);
	//skin->GetProperty()->SetOpacity(.3);

	// An isosurface, or contour value of 1150 is known to correspond to the
	// bone of the patient.
	// The triangle stripper is used to create triangle strips from the
	// isosurface; these render much faster on may systems.
	vtkSmartPointer<vtkMarchingCubes> boneExtractor =
		vtkSmartPointer<vtkMarchingCubes>::New();
	boneExtractor->SetInputData(imageData);
	boneExtractor->SetValue(0, 1150);

	vtkSmartPointer<vtkStripper> boneStripper =
		vtkSmartPointer<vtkStripper>::New();
	boneStripper->SetInputConnection(boneExtractor->GetOutputPort());

	vtkSmartPointer<vtkPolyDataMapper> boneMapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
	boneMapper->SetInputConnection(boneStripper->GetOutputPort());
	boneMapper->ScalarVisibilityOff();

	vtkSmartPointer<vtkActor> bone =
		vtkSmartPointer<vtkActor>::New();
	bone->SetMapper(boneMapper);
	bone->GetProperty()->SetDiffuseColor(colors->GetColor3d("Ivory").GetData());
	//bone->GetProperty()->SetOpacity(.3);

	// An outline provides context around the data.
	vtkSmartPointer<vtkOutlineFilter> outlineData =
		vtkSmartPointer<vtkOutlineFilter>::New();
	outlineData->SetInputData(imageData);

	vtkSmartPointer<vtkPolyDataMapper> mapOutline =
		vtkSmartPointer<vtkPolyDataMapper>::New();
	mapOutline->SetInputConnection(outlineData->GetOutputPort());

	vtkSmartPointer<vtkActor> outline =
		vtkSmartPointer<vtkActor>::New();
	outline->SetMapper(mapOutline);
	outline->GetProperty()->SetColor(colors->GetColor3d("Black").GetData());

	// Now we are creating three orthogonal planes passing through the
	// volume. Each plane uses a different texture map and therefore has
	// different coloration.

	// Start by creating a black/white lookup table.
	vtkSmartPointer<vtkLookupTable> bwLut =
		vtkSmartPointer<vtkLookupTable>::New();
	//bwLut->SetTableRange(0, 2000);
	bwLut->SetTableRange(0, 4095);
	//bwLut->SetNumberOfColors(256);
	bwLut->SetSaturationRange(0, 0);
	bwLut->SetHueRange(0, 0);
	bwLut->SetValueRange(0, 1);
	bwLut->Build(); //effective built

	// Now create a lookup table that consists of the full hue circle
	// (from HSV).
	vtkSmartPointer<vtkLookupTable> hueLut =
		vtkSmartPointer<vtkLookupTable>::New();
	hueLut->SetTableRange(0, 2000);
	hueLut->SetHueRange(0, 1);
	hueLut->SetSaturationRange(1, 1);
	hueLut->SetValueRange(1, 1);
	hueLut->Build(); //effective built

	// Finally, create a lookup table with a single hue but having a range
	// in the saturation of the hue.
	vtkSmartPointer<vtkLookupTable> satLut =
		vtkSmartPointer<vtkLookupTable>::New();
	satLut->SetTableRange(0, 2000);
	satLut->SetHueRange(.6, .6);
	satLut->SetSaturationRange(0, 1);
	satLut->SetValueRange(1, 1);
	satLut->Build(); //effective built

	// Create the first of the three planes. The filter vtkImageMapToColors
	// maps the data through the corresponding lookup table created above.  The
	// vtkImageActor is a type of vtkProp and conveniently displays an image on
	// a single quadrilateral plane. It does this using texture mapping and as
	// a result is quite fast. (Note: the input image has to be unsigned char
	// values, which the vtkImageMapToColors produces.) Note also that by
	// specifying the DisplayExtent, the pipeline requests data of this extent
	// and the vtkImageMapToColors only processes a slice of data.
	vtkSmartPointer<vtkImageMapToColors> sagittalColors =
		vtkSmartPointer<vtkImageMapToColors>::New();
	sagittalColors->SetInputData(imageData);
	sagittalColors->SetLookupTable(bwLut);
	sagittalColors->Update();

	vtkSmartPointer<vtkImageActor> sagittal =
		vtkSmartPointer<vtkImageActor>::New();
	sagittal->GetMapper()->SetInputConnection(sagittalColors->GetOutputPort());
	sagittal->SetDisplayExtent(128, 128, 0, 255, 0, 92);

	// Create the second (axial) plane of the three planes. We use the
	// same approach as before except that the extent differs.
	vtkSmartPointer<vtkImageMapToColors> axialColors =
		vtkSmartPointer<vtkImageMapToColors>::New();
	axialColors->SetInputData(imageData);
	axialColors->SetLookupTable(bwLut);
	axialColors->Update();

	vtkSmartPointer<vtkImageActor> axial =
		vtkSmartPointer<vtkImageActor>::New();
	axial->GetMapper()->SetInputConnection(axialColors->GetOutputPort());
	axial->SetDisplayExtent(0, 255, 0, 255, 21, 21);
	
	// EXPORT IMAGE:
	std::string fileName = "";
	std::string filePath = "figures/";
	std::string fileExt = "-k3.raw.png";

	// Extract and write VOI of axial's plane into PNG file.
	vtkSmartPointer<vtkExtractVOI> axialVOI =
		vtkSmartPointer<vtkExtractVOI>::New();
	vtkSmartPointer<vtkPNGWriter> axialWriter =
		vtkSmartPointer<vtkPNGWriter>::New();
	
	for (int j = 0; j < incrementLimit; j++)
	{
		z = sliceNo + (incrementSparsity * j) - 1;

		for (int i = 0; i < numSlice; i++)
		{
			fileName = filePath + method + "-" + datasetName + std::to_string(z - offsetSlice + i + 1) + fileExt;

			axialVOI->SetInputConnection(axialColors->GetOutputPort());
			axialVOI->SetVOI(0, 255, 0, 255, z - offsetSlice + i, z - offsetSlice + i);
			axialVOI->Update();

			axialWriter->SetInputConnection(axialVOI->GetOutputPort());
			axialWriter->SetFileName(fileName.c_str());
			axialWriter->Write();
		}
	}

	// Create the third (coronal) plane of the three planes. We use
	// the same approach as before except that the extent differs.
	vtkSmartPointer<vtkImageMapToColors> coronalColors =
		vtkSmartPointer<vtkImageMapToColors>::New();
	coronalColors->SetInputData(imageData);
	coronalColors->SetLookupTable(bwLut);
	coronalColors->Update();

	vtkSmartPointer<vtkImageActor> coronal =
		vtkSmartPointer<vtkImageActor>::New();
	coronal->GetMapper()->SetInputConnection(coronalColors->GetOutputPort());
	coronal->SetDisplayExtent(0, 255, 150, 150, 0, 92);

	// It is convenient to create an initial view of the data. The FocalPoint
	// and Position form a vector direction. Later on (ResetCamera() method)
	// this vector is used to position the camera to look at the data in
	// this direction.
	vtkSmartPointer<vtkCamera> aCamera = vtkSmartPointer<vtkCamera>::New();
	aCamera->SetViewUp(0, 0, -1);
	//aCamera->SetPosition(0, 0.5, 0); // View coronal slice
	//aCamera->SetPosition(0.5, 0, 0); // View sagittal slice
	aCamera->SetPosition(0, 1, 0); // View surface rendering
	aCamera->SetFocalPoint(0, 0, 0);
	aCamera->ComputeViewPlaneNormal();
	//aCamera->Azimuth(20.0);
	//aCamera->Elevation(20.0);

	// Actors are added to the renderer. An initial camera view is created.
	// The Dolly() method moves the camera towards the FocalPoint,
	// thereby enlarging the image.
	aRenderer->AddActor(outline);
	//aRenderer->AddActor(sagittal);
	//aRenderer->AddActor(axial);
	//aRenderer->AddActor(coronal);
	aRenderer->AddActor(skin);
	aRenderer->AddActor(bone);
	aRenderer->SetActiveCamera(aCamera);
	aRenderer->ResetCamera();
	aCamera->Dolly(1.5);

	// Set a background color for the renderer and set the size of the
	// render window (expressed in pixels).
	aRenderer->SetBackground(colors->GetColor3d("BkgColor").GetData());
	renWin->SetSize(640, 480);

	// Note that when camera movement occurs (as it does in the Dolly()
	// method), the clipping planes often need adjusting. Clipping planes
	// consist of two planes: near and far along the view direction. The
	// near plane clips out objects in front of the plane; the far plane
	// clips out objects behind the plane. This way only what is drawn
	// between the planes is actually rendered.
	aRenderer->ResetCameraClippingRange();

	// Initialize the event loop and then start it.
	iren->Initialize();
	iren->Start();

	return EXIT_SUCCESS;
}
