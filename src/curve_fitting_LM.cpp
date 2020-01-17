#include <iostream>
#include <ctime>
#include <vector>
#include <random>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "log.h"

// notice:
// new_model_cost
//  = 1/2 [f + J * step]^2
//  = 1/2 [ f'f + 2f'J * step + step' * J' * J * step ]
// model_cost_change
//  = cost - new_model_cost
//  = f'f/2  - 1/2 [ f'f + 2f'J * step + step' * J' * J * step]
//  = -f'J * step - step' * J' * J * step / 2
//  = -(J * step)'(f + J * step / 2)

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
            Vector2d delta = H.inverse() * b;

            if ((bool)std::isnan((double)delta[0])) {
                // T_new = T_old;
                break;
            }

            if (delta.norm() < 1e-10) {
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
        //initial parameter tao v epsilon1 epsilon2
        double eps1 = 1e-12, eps2 = 1e-12;
        double radius = 100.0;

        for (int iter = 0; iter < niters; iter++) {
            log_debug("iter: %d", iter);

            compute_residual_jacobian(Jacobian, errors, m_, c_);
            cost_old = errors.squaredNorm() * 0.5;
            VectorXd gradient = Jacobian.transpose() * errors; //gradient

            if (gradient.norm() <= eps1) {
                log_debug("stop F'(x) = g(x) = 0 for a global minimizer optimizer.");
                break;
            }
            if (errors.norm() <= eps1) {
                log_debug("stop f(x) = 0 for f(x) is so small");
                break;
            }

            //compute how far go along stepest descent direction.
            double alpha = gradient.squaredNorm() / (Jacobian * gradient).squaredNorm();
            //compute gauss newton step and stepest descent step.
            VectorXd stepest_descent = -alpha * gradient;
            VectorXd gauss_newton = (Jacobian.transpose() * Jacobian).inverse() * Jacobian.transpose() * errors * (-1);
            log_debug("  radius: %lf, stepest_descent: %lf, GN: %lf", radius, alpha * stepest_descent.norm(), gauss_newton.norm());

            double beta = 0;
            //compute dog-leg step.
            Vector2d dog_leg;
            if (gauss_newton.norm() <= radius) {
                dog_leg = gauss_newton;
                log_debug("  use GN");
            } else if (gradient.norm() * alpha >= radius) {
                dog_leg = (-radius / gradient.norm()) * gradient;
                log_debug("  use stepest_descent");
            } else {
                VectorXd a = alpha * stepest_descent;
                VectorXd b = gauss_newton;
                double c = a.transpose() * (b - a);
                beta = (sqrt(c * c + (b - a).squaredNorm() * (radius * radius - a.squaredNorm())) - c)
                       / (b - a).squaredNorm();

                dog_leg = alpha * stepest_descent + beta * (gauss_newton - alpha * stepest_descent);
                log_debug("  use both");
            }

            if (dog_leg.norm() <= 1e-8) {
                log_debug("stop because change in x is small");
                break;
            }

            double m_new, c_new;
            m_new = m_ + dog_leg[0];
            c_new = c_ + dog_leg[1];

            double model_cost_change = 0;
            double a = 0.5 * dog_leg.transpose() * Jacobian.transpose() * Jacobian * dog_leg;
            model_cost_change = -(gradient.transpose() * dog_leg + a);

            log_debug("  model_cost_change: %lf", model_cost_change);

            compute_residual_jacobian(Jacobian, errors, m_new, c_new);
            cost_new = errors.squaredNorm() * 0.5;
            if (cost_new > cost_old) { // rejected
                radius *= 0.5;
                continue;
            }

            //compute delta F = F(x) - F(x_new)
            double deltaF = cost_old - cost_new;
            log_debug("  dogleg: %lf %lf, norm: %lf", dog_leg[0], dog_leg[1], dog_leg.norm());
            log_debug("  cost old: %lf, cost new: %lf, para old: %lf %lf, para new: %lf %lf", cost_old, cost_new, m_, c_, m_new, c_new);

            double roi = deltaF / model_cost_change;
            if (roi > 0) {
                m_ = m_new;
                c_ = c_new;
                // log_debug("new para: %lf  %lf  %lf  %lf", params[0], params[1], params[2], params[3]);
            }
            if (roi > 0.75) {
                radius = max(radius, 3.0 * dog_leg.norm());
            } else if (roi < 0.25) {
                if (gauss_newton.norm() <= radius && roi <= 0) {
                    radius = gauss_newton.norm() / 5.0;
                } else {
                    radius = radius / 2.0;
                }
            }
            log_debug("  roi: %lf", roi);
        }
    }

    void addErrorTerm(CostBlock* e) {
        errorTerms.push_back(e);
    }

    double m_, c_;
    std::vector<CostBlock*> errorTerms;
};

// y = exp(a * x + b)
class curve_fitting_cost : public ceres::SizedCostFunction<1, 1, 1> {
  public:
    curve_fitting_cost(double x, double y) :
        _x(x), _y(y) {
    }

    bool Evaluate(const double* const* parameters, double* residual, double** jacobians) const override {
        residual[0] = _y - ceres::exp(parameters[0][0] * _x + parameters[1][0]);
        if (jacobians) {
            if (jacobians[0]) {
                jacobians[0][0] = -_x * ceres::exp(parameters[0][0] * _x + parameters[1][0]);
            }
            if (jacobians[1]) {
                jacobians[1][0] = -ceres::exp(parameters[0][0] * _x + parameters[1][0]);
            }
        }
        return true;
    }

    const double _x, _y;
};

int main() {
    // generate data
    double a = 0.5, b = 0.1;
    std::default_random_engine e;
    std::normal_distribution<double> n(0, 0.2);

    std::vector<std::pair<double, double>> data;

    for (double x = 0; x < 10; x += 0.05) {
        double y = exp(a * x + b) + n(e);
        data.push_back(std::make_pair(x, y));
    }

    double init_m = 0;
    double init_c = 0;
    // LM
    double LM_est_m, LM_est_c;
    LMOptimization opt;
    for (int k = 0; k < data.size(); k++) {
        CostBlock* e = new CostBlock(data[k].first, data[k].second);
        opt.addErrorTerm(e);
    }

    opt.setInitValue(init_m, init_c);
    opt.OptimizeLM(50);
    LM_est_m = opt.m_;
    LM_est_c = opt.c_;

    log_debug("");
    log_debug("opt use dogleg:");

    // dogleg
    opt.setInitValue(init_m, init_c);
    opt.OptimizeDogLeg(50);

    // ceres
    double para1 = init_m, para2 = init_c;
    {
        ceres::Problem problem;
        for (int i = 0; i < data.size(); i++) {
            ceres::CostFunction* cost_fun = new curve_fitting_cost(data[i].first, data[i].second);
            problem.AddResidualBlock(cost_fun, nullptr, &para1, &para2);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::DOGLEG;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }
    log_debug("    LM estimated a: %lf, b: %lf", LM_est_m, LM_est_c);
    log_debug("dogleg estimated a: %lf, b: %lf", opt.m_, opt.c_);
    log_debug(" ceres estimated a: %lf, b: %lf", para1, para2);
    return 0;
}
