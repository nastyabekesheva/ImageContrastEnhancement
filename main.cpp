#include </usr/local/include/opencv4>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

cv::Mat simpleSeidel(cv::Mat U, cv::Mat Ax, cv::Mat Ay, cv::Mat Cx, cv::Mat Cy, cv::Mat B, cv::Mat D, int nx, int ny, double eps = 1e-4) {
    double err = 1;
    int iter = 0;

    while (err > eps) {
        iter += 1;
        if (iter % 100 == 0) {
            std::cout << "iter = " << iter << "; err = " << err << std::endl;
        }
        err = 0;
        for (int i = 1; i <= nx; ++i) {
            for (int j = 1; j <= ny; ++j) {
                double t = (Ax.at<double>(i, j) * U.at<double>(i - 1, j) + Cx.at<double>(i, j) * U.at<double>(i + 1, j) +
                            Ay.at<double>(i, j) * U.at<double>(i, j - 1) + Cy.at<double>(i, j) * U.at<double>(i, j + 1) - D.at<double>(i, j)) / B.at<double>(i, j);
                double div = std::abs(t - U.at<double>(i, j));
                if (err < div) {
                    err = div;
                }
                U.at<double>(i, j) = t;
            }
        }
    }
    return U;
}

cv::Mat mainImgFuncDip(double Lambda, cv::Mat f) {
    int nx = f.rows - 2;
    int ny = f.cols - 2;

    cv::Mat Ax = cv::Mat::ones(nx + 2, ny + 2, CV_64F);
    cv::Mat Cx = Ax.clone();
    cv::Mat Ay = Ax.clone();
    cv::Mat Cy = Ax.clone();
    cv::Mat B = Ax + Cx + Ay + Cy + Lambda;
    cv::Mat D = cv::Mat::zeros(nx + 2, ny + 2, CV_64F);

    for (int i = 1; i <= nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            D.at<double>(i, j) = (f.at<double>(i - 1, j) - 2 * f.at<double>(i, j) + f.at<double>(i + 1, j)) +
                                 (f.at<double>(i, j - 1) - 2 * f.at<double>(i, j) + f.at<double>(i, j + 1));
        }
    }

    cv::Mat U = cv::Mat::zeros(nx + 2, ny + 2, CV_64F);

    for (int j = 1; j <= ny; ++j) {
        B.at<double>(1, j) = B.at<double>(1, j) - Ax.at<double>(1, j);
        Ax.at<double>(1, j) = 0;
        B.at<double>(nx, j) = B.at<double>(nx, j) - Cx.at<double>(nx, j);
        Cx.at<double>(nx, j) = 0;
    }

    for (int i = 1; i <= nx; ++i) {
        B.at<double>(i, 1) = B.at<double>(i, 1) - Ay.at<double>(i, 1);
        Ay.at<double>(i, 1) = 0;
        B.at<double>(i, ny) = B.at<double>(i, ny) - Cy.at<double>(i, ny);
        Cy.at<double>(i, ny) = 0;
    }

    U = simpleSeidel(U, Ax, Ay, Cx, Cy, B, D, nx, ny);

    for (int j = 1; j <= ny; ++j) {
        U.at<double>(0, j) = U.at<double>(1, j);
        U.at<double>(nx + 1, j) = U.at<double>(nx, j);
    }

    for (int i = 1; i <= nx; ++i) {
        U.at<double>(i, 0) = U.at<double>(i, 1);
        U.at<double>(i, ny + 1) = U.at<double>(i, ny);
    }

    U.at<double>(0, 0) = (U.at<double>(0, 1) + U.at<double>(1, 0)) / 2;
    U.at<double>(nx + 1, ny + 1) = (U.at<double>(nx, ny + 1) + U.at<double>(nx + 1, ny)) / 2;
    U.at<double>(0, ny + 1) = (U.at<double>(0, ny) + U.at<double>(1, ny + 1)) / 2;
    U.at<double>(nx + 1, 0) = (U.at<double>(nx, 0) + U.at<double>(nx + 1, 1)) / 2;

    double maxU, minU;
    cv::minMaxLoc(U, &minU, &maxU);
    U = (U - minU) / (maxU - minU);

    return U;
}

void mainTester() {
    std::string path_to_pictures = "./input/im*.png";
    std::string output_directory = "./output";
    std::vector<double> LamArray = {0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2};

    if (!fs::exists(output_directory)) {
        fs::create_directory(output_directory);
    }

    // Table header
    std::vector<std::string> header = {"Image Path", "Vertical Pixels", "Horizontal Pixels", "Total Pixels"};
    for (auto Lam : LamArray) {
        header.push_back("Time (λ=" + std::to_string(Lam) + ")");
    }

    std::vector<std::vector<std::string>> table;
    table.push_back(header);

    for (const auto& entry : fs::directory_iterator("./input")) {
        std::string image_path = entry.path().string();
        cv::Mat f = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (f.empty()) {
            std::cerr << "Error: Image file '" << image_path << "' not found." << std::endl;
            continue;
        }

        f.convertTo(f, CV_64F, 1.0 / 255);
        double maxf, minf;
        cv::minMaxLoc(f, &minf, &maxf);
        f = (f - minf) / (maxf - minf);

        std::vector<std::string> row = {image_path, std::to_string(f.rows), std::to_string(f.cols), std::to_string(f.rows * f.cols)};

        for (auto Lam : LamArray) {
            auto start_time = std::chrono::high_resolution_clock::now();
            cv::Mat U = mainImgFuncDip(Lam, f);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> execution_time = end_time - start_time;
            row.push_back(std::to_string(execution_time.count()));

            // Save the output image
            std::string output_filename = output_directory + "/" + fs::path(image_path).stem().string() + "_lambda_" + std::to_string(Lam) + ".png";
            cv::imwrite(output_filename, U * 255);
        }

        table.push_back(row);
    }

    // Print the table
    for (const auto& row : table) {
        for (const auto& col : row) {
            std::cout << col << " | ";
        }
        std::cout << std::endl;
    }
}

int main() {
    mainTester();
    return 0;
}
