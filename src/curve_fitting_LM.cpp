#include <iostream>
#include <ctime>
#include <vector>
#include <random>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "log.h"

using namespace std;
using namespace Eigen;

class CostBlock {
  public:
    CostBlock(double x, double y) :
        x_(x), y_(y) {
    }
    // residual
    double function(double m, double c) {
        return (exp(m * x_ + c) - y_);
    }
    // jacobian
    Eigen::Matrix<double, 1, 2> Jacobian(double m, double c) {
        Eigen::Matrix<double, 1, 2> J;
        J[0] = x_ * exp(m * x_ + c);
        J[1] = exp(m * x_ + c);
        return J;
    }

    double x_, y_;
};

class LMOptimization {
  public:
    void setInitValue(double m, double c) {
        m_ = m;
        c_ = c;
    }

    void OptimizeLM(int niters) {
        double lambda = 0.01;
        for (int iter = 0; iter < niters; iter++) {
            MatrixXd Jacobian(errorTerms.size(), 2);
            VectorXd err(errorTerms.size());

            for (int k = 0; k < errorTerms.size(); k++) {
                err[k] = errorTerms[k]->function(m_, c_);
                Jacobian.row(k) = errorTerms[k]->Jacobian(m_, c_);
            }

            Matrix2d JTJ = Jacobian.transpose() * Jacobian;                    // hessian
            Matrix2d H = JTJ + lambda * Matrix2d(JTJ.diagonal().asDiagonal()); // LM
            Vector2d b = -Jacobian.transpose() * err;
            //solve
            Vector2d delta = H.colPivHouseholderQr().solve(b);

            if (delta.norm() < 1e-8) {
                log_debug("converged!");
                break;
            }

            double err_before = 0.5 * err.squaredNorm();
            double err_after = 0;
            for (int k = 0; k < errorTerms.size(); k++) {
                double e = errorTerms[k]->function(m_ + delta[0], c_ + delta[1]);
                err_after += e * e;
            }
            err_after *= 0.5;
            if (err_after < err_before) {
                // accept and update
                m_ += delta[0];
                c_ += delta[1];
                lambda /= 2;
                err_before = err_after;
            } else {
                lambda *= 4;
            }

            log_debug("iter: %d, cost: %lf, lambda: %lf", iter, err_before, lambda);
        }
    }
    void addErrorTerm(CostBlock* e) {
        errorTerms.push_back(e);
    }

    double m_, c_;
    std::vector<CostBlock*> errorTerms;
};

int main() {
    double a = 0.5, b = 0.1;
    std::default_random_engine e;
    std::normal_distribution<double> n(0, 0.2);

    std::vector<std::pair<double, double>> data;

    for (double x = 0; x < 10; x += 0.05) {
        double y = exp(a * x + b) + n(e);
        data.push_back(std::make_pair(x, y));
    }

    LMOptimization opt;
    // construct problem
    for (int k = 0; k < data.size(); k++) {
        CostBlock* e = new CostBlock(data[k].first, data[k].second);
        opt.addErrorTerm(e);
    }

    opt.setInitValue(0, 0);
    opt.OptimizeLM(50);

    log_debug("estimated a: %lf, b: %lf", opt.m_, opt.c_);
    return 0;
}
