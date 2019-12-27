#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>
#include "log.h"

class curve_fitting_cost {
  public:
    curve_fitting_cost(double x, double y) :
        _x(x), _y(y) {
    }

    template <typename T>
    bool operator()(const T *const a, const T *const b, const T *const c, T *residual) const {
        residual[0] = T(_y) - ceres::exp(a[0] * T(_x) * T(_x) + b[0] * T(_x) + c[0]); // y-exp(ax^2+bx+c)
        return true;
    }

    const double _x, _y;
};

class curve_fitting_cost1 {
  public:
    curve_fitting_cost1(double x, double y) :
        _x(x), _y(y) {
    }

    template <typename T>
    bool operator()(const T *const a, T *residual) const {
        residual[0] = T(_y) - ceres::exp(a[0] * T(_x) * T(_x) + a[1] * T(_x) + a[2]); // y-exp(ax^2+bx+c)
        return true;
    }

    const double _x, _y;
};

class curve_fitting_cost2 : public ceres::SizedCostFunction<1, 1, 1, 1> {
  public:
    curve_fitting_cost2(double x, double y) :
        _x(x), _y(y) {
    }

    bool Evaluate(const double *const *parameters, double *residual, double **jacobians) const override {
        residual[0] = _y - ceres::exp(parameters[0][0] * _x * _x + parameters[1][0] * _x + parameters[2][0]); // y-exp(ax^2+bx+c)
        if (jacobians) {
            if (jacobians[0]) {
                jacobians[0][0] = -_x * _x * ceres::exp(parameters[0][0] * _x * _x + parameters[1][0] * _x + parameters[2][0]);
            }
            if (jacobians[1]) {
                jacobians[1][0] = -_x * ceres::exp(parameters[0][0] * _x * _x + parameters[1][0] * _x + parameters[2][0]);
            }
            if (jacobians[2]) {
                jacobians[2][0] = -ceres::exp(parameters[0][0] * _x * _x + parameters[1][0] * _x + parameters[2][0]);
            }
        }
        return true;
    }

    const double _x, _y;
};

int main() {
    double a = 1.0, b = 3.0, c = 1.0; // 真实参数值
    const int N = 100;
    double w_sigma = 0.8; // 噪声Sigma值
    cv::RNG rng;          // OpenCV随机数
    bool brief_report = false;
    bool cout_progress = false;

    std::vector<double> x_data, y_data;

    log_debug("generating data...");
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(
            exp(a * x * x + b * x + c) + rng.gaussian(w_sigma));
    }
    log_debug("done");
    {
        log_debug("");
        double para1, para2, para3;
        ceres::Problem problem;
        for (int i = 0; i < N; i++) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<curve_fitting_cost, 1, 1, 1, 1>(
                    new curve_fitting_cost(x_data[i], y_data[i])),
                nullptr, &para1, &para2, &para3);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.minimizer_progress_to_stdout = cout_progress;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (brief_report)
            std::cout << summary.BriefReport() << std::endl;
        log_debug("estimated a: %lf, b: %lf, c: %lf", para1, para2, para3);
    }
    {
        log_debug("");
        double paras[3] = {0, 0, 0};
        ceres::Problem problem;
        for (int i = 0; i < N; i++) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<curve_fitting_cost1, 1, 3>(
                    new curve_fitting_cost1(x_data[i], y_data[i])),
                nullptr, paras);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.minimizer_progress_to_stdout = cout_progress;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (brief_report)
            std::cout << summary.BriefReport() << std::endl;
        log_debug("estimated a: %lf, b: %lf, c: %lf", paras[0], paras[1], paras[2]);
    }
    {
        log_debug("");
        double para1, para2, para3;
        ceres::Problem problem;
        for (int i = 0; i < N; i++) {
            ceres::CostFunction *cost_fun = new curve_fitting_cost2(x_data[i], y_data[i]);
            problem.AddResidualBlock(cost_fun, nullptr, &para1, &para2, &para3);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.minimizer_progress_to_stdout = cout_progress;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (brief_report)
            std::cout << summary.BriefReport() << std::endl;
        log_debug("estimated a: %lf, b: %lf, c: %lf", para1, para2, para3);
    }

    return 0;
}
