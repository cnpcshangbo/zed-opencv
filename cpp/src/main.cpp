///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2020, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/***********************************************************************************************
 ** This sample demonstrates how to use the ZED SDK with OpenCV. 					  	      **
 ** Depth and images are captured with the ZED SDK, converted to OpenCV format and displayed. **
 ***********************************************************************************************/

 // ZED includes
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//using namespace cv;

// Sample includes
#include <SaveDepth.hpp>

using namespace sl;

cv::Mat slMat2cvMat(Mat& input);
void printHelp();

int main(int argc, char **argv) {

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_params;
    //init_params.input.setFromSVOFile("/home/nvidia/Documents/ZED/HD720_SN2353053_14-02-19.svo");
    //init_params.input.setFromSVOFile("/home/nvidia/Documents/ZED/HD720_SN2353053_14-05-18.svo");
    //init_params.camera_resolution = RESOLUTION::HD1080;
    init_params.depth_mode = DEPTH_MODE::ULTRA;
    init_params.coordinate_units = UNIT::METER;
    //if (argc > 1) init_params.input.setFromSVOFile(argv[1]);
    init_params.coordinate_units = UNIT::MILLIMETER; // Use millimeter units (for depth measurements)

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        printf("%s\n", toString(err).c_str());
        zed.close();
        return 1; // Quit if an error occurred
    }

    // Display help in console
    printHelp();

    // Set runtime parameters after opening the camera
    RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE::STANDARD;

    // Prepare new image size to retrieve half-resolution images
    Resolution image_size = zed.getCameraInformation().camera_resolution;
    int new_width = image_size.width / 2;
    int new_height = image_size.height / 2;

    Resolution new_image_size(new_width, new_height);

    // To share data between sl::Mat and cv::Mat, use slMat2cvMat()
    // Only the headers and pointer to the sl::Mat are copied, not the data itself
    Mat image_zed(new_width, new_height, MAT_TYPE::U8_C4);
    cv::Mat image_ocv = slMat2cvMat(image_zed);
    Mat depth_image_zed(new_width, new_height, MAT_TYPE::U8_C4);
    cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);
    Mat depth, point_cloud;

    // Loop until 'q' is pressed
    char key = ' ';
    while (key != 'q') {

        if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS) {

            // Retrieve the left image, depth image in half-resolution
            zed.retrieveImage(image_zed, VIEW::LEFT, MEM::CPU, new_image_size);
            zed.retrieveImage(depth_image_zed, VIEW::DEPTH, MEM::CPU, new_image_size);
            zed.retrieveMeasure(depth, MEASURE::DEPTH, MEM::CPU, new_image_size);
            // Retrieve the RGBA point cloud in half-resolution
            // To learn how to manipulate and display point clouds, see Depth Sensing sample
            zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA, MEM::CPU, new_image_size);

            //Detecting wood from depth_image_zed
            int n_min = 0;
            int n = 0;
            float depth_value;
            float depth_value_min = 2500;
            //depth.getValue(0,depth_image_zed.getHeight() / 2, &depth_value_min);
            printf("image_zed.Width = %d, height = %d.", image_zed.getWidth(), image_zed.getHeight());
            printf("depth_image_zed.Width = %d, height = %d.", depth_image_zed.getWidth(), depth_image_zed.getHeight());
            printf("depth.Width = %d, height = %d.", depth.getWidth(), depth.getHeight());
            while (n < depth_image_zed.getWidth()){
                        //int x = image.getWidth() / 2;
                        int x = n;
                        int y = depth_image_zed.getHeight() / 2;

                        depth.getValue(x,y, &depth_value);
                        //printf("Depth to Camera at (%d, %d): %f mm\n", x, y, depth_value);
                        //printf("");
                        if ((isnan(depth_value)==0) && (depth_value < depth_value_min) && (depth_value > 0)) {
                          n_min = n; depth_value_min = depth_value;
                        }
            n++;
            }
            printf("Min value is at %d. Value is %f mm. \n", n_min, depth_value_min);

            cv::circle( depth_image_ocv, cv::Point( n_min, depth_image_zed.getHeight() / 2 ), 32.0, cv::Scalar( 0, 0, 255 ), 1, 8 );
            // Display image and depth using cv:Mat which share sl:Mat data
            cv::imshow("Image", image_ocv);
            cv::imshow("Depth", depth_image_ocv);

            // Handle key event
            key = cv::waitKey(10);
            processKeyEvent(zed, key);
        }
    }
    zed.close();
    return 0;
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM::CPU));
}

/**
* This function displays help in console
**/
void printHelp() {
    std::cout << " Press 's' to save Side by side images" << std::endl;
    std::cout << " Press 'p' to save Point Cloud" << std::endl;
    std::cout << " Press 'd' to save Depth image" << std::endl;
    std::cout << " Press 'm' to switch Point Cloud format" << std::endl;
    std::cout << " Press 'n' to switch Depth format" << std::endl;
}
