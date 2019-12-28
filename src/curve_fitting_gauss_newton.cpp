#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include "log.h"

using namespace Eigen;

class curve_fitting_cost : public ceres::SizedCostFunction<1, 1, 1, 1> {
  public:
    curve_fitting_cost(double x, double y) :
        x_(x), y_(y) {
    }

    virtual ~curve_fitting_cost() {
    }

    bool Evaluate(const double *const *parameters, double *residual, double **jacobians) const override {
        residual[0] = parameters[0][0] * x_ * x_ + parameters[1][0] * x_ + parameters[2][0] - y_;

        if (jacobians) {
            if (jacobians[0]) {
                jacobians[0][0] = x_ * x_;
            }
            if (jacobians[1]) {
                jacobians[1][0] = x_;
            }
            if (jacobians[2]) {
                jacobians[2][0] = 1;
            }
        }

        return true;
    }

  private:
    double x_;
    double y_;
};

// y = a * x^2 + b * x + c;
int main() {
    double a = 2.0, b = 1.0, c = 3.0; // 真实参数
    double para1 = 0, para2 = 0, para3 = 0;
    double para4 = 0, para5 = 0, para6 = 0;
    const int N = 30;

    std::default_random_engine e;
    std::normal_distribution<double> n(0, 0.2);

    std::vector<double> x_data, y_data;
    x_data.reserve(N);
    y_data.reserve(N);

    log_debug("generating data...");
    for (double x = 0; x < N; x += 0.5) {
        double y = a * x * x + b * x + c + n(e);
        x_data.push_back(x);
        y_data.push_back(y);
    }
    log_debug("done");

    // estimate using ceres
    {
        ceres::Problem problem;
        problem.AddParameterBlock(&para1, 1);
        problem.AddParameterBlock(&para2, 1);
        problem.AddParameterBlock(&para3, 1);
        for (int i = 0; i < N; i++) {
            ceres::CostFunction *cost_fun = new curve_fitting_cost(x_data[i], y_data[i]);
            problem.AddResidualBlock(cost_fun, nullptr, &para1, &para2, &para3);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::DOGLEG;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }
    //
    {
        size_t iterations = 100;
        double cost = 0, lastCost = 0; // 本次的cost和上一次迭代的cost
        for (int iter = 0; iter < iterations; iter++) {
            Matrix3d H = Matrix3d::Zero(); // Hessian = J^T J
            Vector3d b = Vector3d::Zero();
            cost = 0;

            for (int i = 0; i < N; i++) {
                double xi = x_data[i], yi = y_data[i]; // 第i个数据点
                double error = 0;
                error = para4 * xi * xi + para5 * xi + para6 - yi;

                Vector3d J;
                J[0] = xi * xi;
                J[1] = xi;
                J[2] = 1.0;

                H += J * J.transpose();
                b += -error * J;

                cost += error * error;
            }

            Vector3d dx;
            dx = H.inverse() * b;

            if (isnan(dx[0])) {
                log_debug("result is nan!");
                break;
            }

            if (iter > 0 && cost > lastCost) {
                // 误差增长了
                log_debug("cost: %lf, last cost: %lf", cost, lastCost);
                break;
            }
            // update
            para4 += dx[0];
            para5 += dx[1];
            para6 += dx[2];

            double dt = std::max(std::max(dx[0], dx[1]), dx[2]);
            if (std::abs(dt) < 1e-8) {
                log_debug("converged!");
                break;
            }

            lastCost = cost;
        }
    }

    log_debug("estimate result:");
    log_debug("ceres estimated para1: %lf, para2: %lf, para3: %lf", para1, para2, para3);
    log_debug("   GN estimated para4: %lf, para5: %lf, para6: %lf", para4, para5, para6);
    return 0;
}
