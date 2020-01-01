#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include "log.h"
#include "LS_solver.h"

class curve_fitting_cost : public CostFunction {
  public:
    curve_fitting_cost(double x, double y) {
        x_ = x;
        y_ = y;
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        residuals[0] = parameters[0][0] * x_ * x_ + parameters[1][0] * x_ + parameters[2][0] - y_;

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
    double x_, y_;
};

// y = a * x^2 + b * x + c;
int main() {
    double a = 2.0, b = 1.0, c = 3.0; // 真实参数
    double para1 = 0.0, para2 = 0.0, para3 = 0.0;
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

    LS_solver solver(3, 1);
    solver.add_parameter_block(&para1);
    solver.add_parameter_block(&para2);
    solver.add_parameter_block(&para3);
    for (int i = 0; i < N; i++) {
        CostFunction *cost = new curve_fitting_cost(x_data[i], y_data[i]);
        solver.add_residual_block(cost);
    }
    solver.Optimize(50);

    log_debug("estimate result:");
    log_debug("solver estimated para1: %lf, para2: %lf, para3: %lf", para1, para2, para3);

    return 0;
}
