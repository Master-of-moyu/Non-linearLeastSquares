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

    void compute_residual_jacobian(MatrixXd& Jacobian, VectorXd& errors, double m, double c) {
        Jacobian.setZero();
        errors.setZero();
        for (int k = 0; k < errorTerms.size(); k++) {
            errors[k] = errorTerms[k]->function(m, c);
            Jacobian.row(k) = errorTerms[k]->Jacobian(m, c);
        }
    }

    void OptimizeLM(int niters) {
        double cost_old = 0.0, cost_new = 0.0;
        double lambda = 0.01;
        MatrixXd Jacobian(errorTerms.size(), 2);
        VectorXd errors(errorTerms.size());

        compute_residual_jacobian(Jacobian, errors, m_, c_);
        cost_old = 0.5 * errors.squaredNorm();

        for (int iter = 0; iter < niters; iter++) {
            Matrix2d JTJ = Jacobian.transpose() * Jacobian; // hessian
            Matrix2d H = JTJ;
            Vector2d b = -Jacobian.transpose() * errors;
            for (int i = 0; i < H.cols(); i++)
                H(i, i) *= (1 + lambda);
            //solve
            Vector2d delta = H.colPivHouseholderQr().solve(b);

            if ((bool)std::isnan((double)delta[0])) {
                // T_new = T_old;
                break;
            }

            if (delta.maxCoeff() < 1e-8) {
                log_debug("converged!");
                break;
            }

            // update
            cost_new = 0.0;
            m_ += delta[0];
            c_ += delta[1];
            compute_residual_jacobian(Jacobian, errors, m_, c_);
            cost_new = 0.5 * errors.squaredNorm();
            if (cost_new < cost_old) {
                // accept and update
                lambda /= 2;
                cost_old = cost_new;
                if (lambda < 1e-6) lambda = 1e-6;
            } else {
                m_ -= delta[0];
                c_ -= delta[1];
                compute_residual_jacobian(Jacobian, errors, m_, c_);
                lambda *= 3;
                if (lambda > 1000) lambda = 1000;
            }

            log_debug("iter: %d, cost: %lf, lambda: %lf", iter, cost_old, lambda);
        }
    }

    void OptimizeDogLeg(int niters) {
        double cost_old = 0.0, cost_new = 0.0;
        MatrixXd Jacobian(errorTerms.size(), 2);
        VectorXd errors(errorTerms.size());

        compute_residual_jacobian(Jacobian, errors, m_, c_);
        cost_old = 0.5 * errors.squaredNorm();

        for (int iter = 0; iter < niters; iter++) {
            Matrix2d JTJ = Jacobian.transpose() * Jacobian; // hessian
            Matrix2d H = JTJ;
            Vector2d b = -Jacobian.transpose() * errors;
            //solve
            Vector2d delta = H.colPivHouseholderQr().solve(b);

            if ((bool)std::isnan((double)delta[0])) {
                // T_new = T_old;
                break;
            }

            if (delta.maxCoeff() < 1e-8) {
                log_debug("converged!");
                break;
            }

            // update
            cost_new = 0.0;
            m_ += delta[0];
            c_ += delta[1];
            compute_residual_jacobian(Jacobian, errors, m_, c_);
            cost_new = 0.5 * errors.squaredNorm();
            if (cost_new < cost_old) {
                // accept and update
                cost_old = cost_new;
            } else {
                m_ -= delta[0];
                c_ -= delta[1];
                log_debug("cost increased. iter: %d", iter);
                break;
            }

            log_debug("iter: %d, cost: %lf", iter, cost_old);
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
    log_debug("");
    log_debug("opt again:");

    opt.setInitValue(0, 0);
    opt.OptimizeDogLeg(50);
    log_debug("estimated a: %lf, b: %lf", opt.m_, opt.c_);
    return 0;
}
