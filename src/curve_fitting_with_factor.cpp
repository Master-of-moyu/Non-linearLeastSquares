#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include "log.h"

using namespace Eigen;

class ParaFactor : public ceres::SizedCostFunction<1, 1> {
  public:
    ParaFactor(double x) :
        x_(x) {
    }

    virtual ~ParaFactor() {
    }

    bool Evaluate(const double *const *parameters, double *residual, double **jacobians) const override {
        residual[0] = x_ - parameters[0][0];

        double weight;
        weight = 5.0;
        residual[0] *= weight;

        if (jacobians) {
            if (jacobians[0]) {
                jacobians[0][0] = -1 * weight;
            }
        }

        return true;
    }

  private:
    double x_;
};

class AnaCostFun : public ceres::SizedCostFunction<1, 1, 1, 1> {
  public:
    AnaCostFun(double x, double y) :
        x_(x), y_(y) {
    }

    virtual ~AnaCostFun() {
    }

    bool Evaluate(const double *const *parameters, double *residual, double **jacobians) const override {
        residual[0] = parameters[0][0] * x_ * x_ + parameters[1][0] * x_ + parameters[2][0] - y_;

        double weight;
        weight = 1.0;
        residual[0] *= weight;

        if (jacobians) {
            if (jacobians[0]) {
                jacobians[0][0] = x_ * x_ * weight;
            }
            if (jacobians[1]) {
                jacobians[1][0] = x_ * weight;
            }
            if (jacobians[2]) {
                jacobians[2][0] = 1 * weight;
            }
        }

        return true;
    }

  private:
    double x_;
    double y_;
};

// y = a * x^2 + b * x + c;  a = 2, b = 1, c = 3
int main() {
    double a = 2.0, b = 1.0, c = 3.0; // 真实参数值
    double para1 = 0, para2 = 0, para3 = 0;
    double para4 = 0, para5 = 0, para6 = 0;
    const int N = 30;

    std::default_random_engine e;
    std::normal_distribution<double> n(0, 0.2);

    std::vector<double> x_data, y_data;
    x_data.reserve(N);
    y_data.reserve(N);

    for (double x = 0; x < N; x += 0.5) {
        double y = a * x * x + b * x + c + n(e);
        x_data.push_back(x);
        y_data.push_back(y);
    }

    // using ceres
    {
        ceres::Problem problem;
        for (int i = 0; i < N; i++) {
            ceres::CostFunction *cost_fun = new AnaCostFun(x_data[i], y_data[i]);
            problem.AddResidualBlock(cost_fun, nullptr, &para1, &para2, &para3);
        }

        ceres::CostFunction *para1_cost = new ParaFactor(a);
        problem.AddResidualBlock(para1_cost, nullptr, &para1);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }
    //
    {
        size_t iterations = 100;
        double cost = 0, lastCost = 0;
        for (int iter = 0; iter < iterations; iter++) {
            Matrix3d H = Matrix3d::Zero(); // Hessian = J^T J
            Vector3d b = Vector3d::Zero();
            cost = 0;

            for (int i = 0; i < N; i++) {
                double weight = 1.0;
                double xi = x_data[i], yi = y_data[i];
                double error = 0;
                error = para4 * xi * xi + para5 * xi + para6 - yi;

                Vector3d J;
                J[0] = xi * xi;
                J[1] = xi;
                J[2] = 1.0;
                // 1.
                // H += J * J.transpose();
                // b += -error * J;
                // 2.
                for (int i = 0; i < H.rows(); i++) {
                    for (int j = 0; j < H.cols(); j++)
                        H(i, j) += J[i] * J[j];
                }

                b += -error * J;

                cost += error * error;
            }

            double error3 = a - para4;
            double j3 = -1;
            double weight3 = 5.0;

            error3 *= weight3;
            j3 *= weight3;

            Vector3d J;
            J[0] = j3;
            J[1] = 0;
            J[2] = 0;
            // 1.
            // H += J * J.transpose();
            // b += -error3 * J;
            // 2.
            for (int i = 0; i < H.rows(); i++) {
                for (int j = 0; j < H.cols(); j++)
                    H(i, j) += J[i] * J[j];
            }

            b += -error3 * J;

            cost += error3 * error3;

            Vector3d dx;
            dx = H.inverse() * b;

            if (isnan(dx[0])) {
                log_debug("result is nan!");
                break;
            }

            if (iter > 0 && cost > lastCost) {
                // error increased
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
            // log_debug("total cost: %lf", cost);
        }
    }

    log_debug("estimate result:");
    log_debug("ceres estimated para1: %lf, para2: %lf, para3: %lf", para1, para2, para3);
    log_debug("   GN estimated para4: %lf, para5: %lf, para6: %lf", para4, para5, para6);

    return 0;
}
