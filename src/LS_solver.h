#pragma once

#include "types.h"
#include "log.h"

class CostFunction {
  public:
    CostFunction() = default;
    virtual ~CostFunction() = default;

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const = 0;
};

class LS_solver {
  public:
    LS_solver(size_t num_para, size_t resi_dim) {
        parameter_num = num_para;
        residual_dim = resi_dim;
    }

    void add_residual_block(CostFunction* e) {
        cost_blocks.push_back(e);
    }

    void add_parameter_block(double* data) {
        parameters.push_back(data);
    }

    void Optimize(int niters) {
        double cost_old = 0.0, cost_new = 0.0;
        matrix<> A;
        vector<> b;
        x.resize(parameter_num);
        A.resize(parameter_num, parameter_num);
        b.resize(parameter_num);
        // copy initial parameters
        for (int i = 0; i < parameter_num; i++) {
            x[i] = *parameters[i];
        }

        vector<> x_old = x;
        cost_old = linearize(A, b, x);
        // log_debug("[LS solver] a1");
        double lambda = 0.02;
        for (int iter = 0; iter < niters; iter++) {
            // LM
            for (int i = 0; i < parameter_num; i++)
                A(i, i) *= (1 + lambda);
            //solve
            const vector<> dT = A.ldlt().solve(b);

            if ((bool)std::isnan((double)dT[0])) {
                x = x_old;
                log_debug("[LS solver] dt0 nan.");
                break;
            }
            if (dT.maxCoeff() < 1e-8) {
                log_debug("[LS solver] converged.");
                break;
            }
            // update
            x += dT;
            cost_new = linearize(A, b, x);
            if (cost_new < cost_old) {
                // accept
                cost_old = cost_new;
                lambda *= 0.5;
                if (lambda < 1e-6) lambda = 1e-6;
            } else {
                x = x_old;
                log_debug("[LS solver] cost increased. iter: %d", iter);
                lambda *= 3;
                if (lambda > 1000) lambda = 1000;
            }
        }

        // copy estimated parameters
        for (int i = 0; i < parameter_num; i++) {
            *parameters[i] = x[i];
        }
    }

    double linearize(matrix<>& A, vector<>& b, vector<>& x) {
        double energy = 0.0;
        A.setZero();
        b.setZero();
        for (int k = 0; k < cost_blocks.size(); k++) {
            // evaluate every cost function
            std::vector<matrix<Eigen::Dynamic, Eigen::Dynamic, true>> jacobians(parameter_num);
            std::vector<const double*> parameter_ptrs(parameter_num);
            std::vector<double*> jacobian_ptrs(parameter_num);
            vector<> residual;
            residual.resize(residual_dim);
            // data prepare
            for (int i = 0; i < parameter_num; i++) {
                jacobians[i].resize(1, 1);
                parameter_ptrs[i] = &x[i];
                jacobian_ptrs[i] = jacobians[i].data();
            }
            CostFunction* cost = cost_blocks[k];
            cost->Evaluate(parameter_ptrs.data(), residual.data(), jacobian_ptrs.data());
            // fill A & b
            std::vector<matrix<>> J(parameter_num);
            for (int i = 0; i < parameter_num; i++) {
                matrix<>& j = J[i];
                j.resize(1, 1);
                j = jacobians[i];
            }

            for (int i = 0; i < parameter_num; i++) {
                for (int j = 0; j < parameter_num; j++) {
                    A.block<1, 1>(i, j) += J[i].transpose() * J[j];
                }
                b.segment<1>(i) += -residual[0] * J[i];
            }
            // add energy
            for (int i = 0; i < residual.size(); i++) {
                energy += residual[i] * residual[i];
            }
        }
        return energy * 0.5;
    }

    std::vector<CostFunction*> cost_blocks;
    std::vector<double*> parameters;
    size_t parameter_num = 0;
    size_t residual_dim = 0;
    vector<> x;
};
