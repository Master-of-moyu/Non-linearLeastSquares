#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <ceres/ceres.h>
#include "log.h"

using namespace std;
using namespace Eigen;

const double DERIV_STEP = 1e-5;
const int MAX_ITER = 200;

#define max(a, b) (((a) > (b)) ? (a) : (b))

class curve_fitting_cost {
  public:
    curve_fitting_cost(double x, double y) :
        _x(x), _y(y) {
    }

    template <typename T>
    bool operator()(const T* const a, const T* const b, const T* const c, const T* const d, T* residual) const {
        residual[0] = T(_y) - (a[0] * sin(b[0] * T(_x)) + c[0] * cos(d[0] * T(_x)));
        return true;
    }

    const double _x, _y;
};

double func(const VectorXd& input, const VectorXd& output, const VectorXd& params, double objIndex) {
    // obj = A * sin(Bx) + C * cos(D*x) - F
    double x1 = params(0);
    double x2 = params(1);
    double x3 = params(2);
    double x4 = params(3);

    double t = input(objIndex);
    double f = output(objIndex);

    return x1 * sin(x2 * t) + x3 * cos(x4 * t) - f;
}

//return vector make up of func() element.
VectorXd objF(const VectorXd& input, const VectorXd& output, const VectorXd& params) {
    VectorXd obj(input.rows());
    for (int i = 0; i < input.rows(); i++)
        obj(i) = func(input, output, params, i);

    return obj;
}

//F = (f ^t * f)/2
double Func(const VectorXd& obj) {
    return obj.squaredNorm() / 2;
}

double Deriv(const VectorXd& input, const VectorXd& output, int objIndex, const VectorXd& params,
             int paraIndex) {
    VectorXd para1 = params;
    VectorXd para2 = params;

    para1(paraIndex) -= DERIV_STEP;
    para2(paraIndex) += DERIV_STEP;

    double obj1 = func(input, output, para1, objIndex);
    double obj2 = func(input, output, para2, objIndex);

    return (obj2 - obj1) / (2 * DERIV_STEP);
}

MatrixXd Jacobin(const VectorXd& input, const VectorXd& output, const VectorXd& params) {
    int rowNum = input.rows();
    int colNum = params.rows();

    MatrixXd Jac(rowNum, colNum);

    for (int i = 0; i < rowNum; i++) {
        for (int j = 0; j < colNum; j++) {
            Jac(i, j) = Deriv(input, output, i, params, j);
        }
    }
    return Jac;
}

void dogLeg(const VectorXd& input, const VectorXd& output, VectorXd& params) {
    int errNum = input.rows();   //error num
    int paraNum = params.rows(); //parameter num

    VectorXd obj = objF(input, output, params);    // residuals
    MatrixXd Jac = Jacobin(input, output, params); //jacobin
    VectorXd gradient = Jac.transpose() * obj;     //gradient

    //initial parameter tao v epsilon1 epsilon2
    double eps1 = 1e-12, eps2 = 1e-12;
    double radius = 1.0;

    bool found = obj.norm() <= eps1 || gradient.norm() <= eps1;
    if (found) return;

    double last_sum = 0;
    int iterCnt = 0;
    while (iterCnt < MAX_ITER) {
        log_debug(" ");
        log_debug("iter: %d", iterCnt);
        VectorXd obj = objF(input, output, params);
        MatrixXd Jac = Jacobin(input, output, params); //jacobin
        VectorXd gradient = Jac.transpose() * obj;     //gradient

        if (gradient.norm() <= eps1) {
            cout << "stop F'(x) = g(x) = 0 for a global minimizer optimizer." << endl;
            break;
        }
        if (obj.norm() <= eps1) {
            cout << "stop f(x) = 0 for f(x) is so small";
            break;
        }

        //compute how far go along stepest descent direction.
        double alpha = gradient.squaredNorm() / (Jac * gradient).squaredNorm();
        //compute gauss newton step and stepest descent step.
        VectorXd stepest_descent = -alpha * gradient;
        VectorXd gauss_newton = (Jac.transpose() * Jac).inverse() * Jac.transpose() * obj * (-1);
        log_debug("radius: %lf, stepest_descent: %lf, GN: %lf", radius, alpha * stepest_descent.norm(), gauss_newton.norm());

        double beta = 0;

        //compute dog-leg step.
        VectorXd dog_leg(params.rows());
        if (gauss_newton.norm() <= radius) {
            dog_leg = gauss_newton;
            log_debug("use GN");
        } else if (alpha * stepest_descent.norm() >= radius) {
            dog_leg = (radius / stepest_descent.norm()) * stepest_descent;
            log_debug("use stepest_descent");
        } else {
            VectorXd a = alpha * stepest_descent;
            VectorXd b = gauss_newton;
            double c = a.transpose() * (b - a);
            beta = (sqrt(c * c + (b - a).squaredNorm() * (radius * radius - a.squaredNorm())) - c)
                   / (b - a).squaredNorm();

            dog_leg = alpha * stepest_descent + beta * (gauss_newton - alpha * stepest_descent);
            log_debug("use both");
        }
        // log_debug("dogleg step: %lf  %lf  %lf  %lf ", dog_leg[0], dog_leg[1], dog_leg[2], dog_leg[3]);

        if (dog_leg.norm() <= 1e-8) {
            cout << "stop because change in x is small" << endl;
            break;
        }

        VectorXd new_params(params.rows());
        new_params = params + dog_leg;

        // cout << "new parameter is: " << endl
        //      << new_params << endl;

        //compute f(x)
        obj = objF(input, output, params);
        //compute f(x_new)
        VectorXd obj_new = objF(input, output, new_params);

        //compute delta F = F(x) - F(x_new)
        double deltaF = Func(obj) - Func(obj_new);
        log_debug("cost old: %lf, cost new: %lf", Func(obj), Func(obj_new));

        //compute delat L =L(0)-L(dog_leg)
        double deltaL = 0;
        if (gauss_newton.norm() <= radius)
            deltaL = Func(obj);
        else if (alpha * stepest_descent.norm() >= radius)
            deltaL = radius * (2 * alpha * gradient.norm() - radius) / (2.0 * alpha);
        else {
            VectorXd a = alpha * stepest_descent;
            VectorXd b = gauss_newton;
            double c = a.transpose() * (b - a);
            beta = (sqrt(c * c + (b - a).squaredNorm() * (radius * radius - a.squaredNorm())) - c)
                   / (b - a).squaredNorm();

            deltaL = alpha * (1 - beta) * (1 - beta) * gradient.squaredNorm() / 2.0 + beta * (2.0 - beta) * Func(obj);
        }

        double roi = deltaF / deltaL;
        if (roi > 0) {
            params = new_params;
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

            if (radius <= eps2 * (params.norm() + eps2)) {
                cout << "trust region radius is too small." << endl;
                break;
            }
        }
        log_debug("roi: %lf, dogleg norm: %lf", roi, dog_leg.norm());
        iterCnt++;
    }
}

void LM(const VectorXd& input, const VectorXd& output, VectorXd& params) {
    double cost_old = 0.0, cost_new = 0.0;
    double lambda = 0.01;
    int errNum = input.rows();   //error num
    int paraNum = params.rows(); //parameter num

    VectorXd obj = objF(input, output, params);    // residuals
    MatrixXd Jac = Jacobin(input, output, params); //jacobin
    VectorXd b = -Jac.transpose() * obj;           //gradient
    cost_old = Func(obj);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // log_debug(" ");
        // log_debug("iter: %d, lambda: %lf", iter, lambda);
        Matrix4d H = Jac.transpose() * Jac;
        for (int i = 0; i < H.cols(); i++)
            H(i, i) *= (1 + lambda);

        //solve
        Vector4d delta = H.inverse() * b;

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
        params += delta;

        obj = objF(input, output, params);
        cost_new = Func(obj);
        if (cost_new < cost_old) {
            // accept and update
            lambda /= 2;
            cost_old = cost_new;
            if (lambda < 1e-6) lambda = 1e-6;
        } else {
            params -= delta;
            lambda *= 3;
            obj = objF(input, output, params);    // residuals
            Jac = Jacobin(input, output, params); //jacobin
            b = -Jac.transpose() * obj;           //gradient
            if (lambda > 1000) lambda = 1000;
        }
        log_debug("iter: %d, cost: %lf, lambda: %lf", iter, cost_old, lambda);
    }
}

int main() {
    // obj = A * sin(Bx) + C * cos(D*x) - F
    //there are 4 parameter: A, B, C, D.
    int num_params = 4;
    //generate random data using these parameter
    int total_data = 100;

    VectorXd input(total_data);
    VectorXd output(total_data);

    double A = 5.0, B = 1.0, C = 10.0, D = 2.0;
    //load observation data
    for (int i = 0; i < total_data; i++) {
        //generate a random variable [-10 10]
        double x = 20.0 * ((random() % 1000) / 1000.0) - 10.0;
        double deltaY = 2.0 * (random() % 1000) / 1000.0;
        double y = A * sin(B * x) + C * cos(D * x) + deltaY;

        input(i) = x;
        output(i) = y;
    }

    //gauss the parameters
    VectorXd params_gaussNewton(num_params);
    //init gauss
    params_gaussNewton << 1.6, 1.4, 6.2, 1.7;

    VectorXd params_levenMar = params_gaussNewton;
    VectorXd params_dogLeg = params_gaussNewton;

    dogLeg(input, output, params_dogLeg);
    LM(input, output, params_levenMar);
    //ceres
    double para1, para2, para3, para4;
    para1 = params_gaussNewton[0];
    para2 = params_gaussNewton[1];
    para3 = params_gaussNewton[2];
    para4 = params_gaussNewton[3];
    {
        ceres::Problem problem;
        for (int i = 0; i < total_data; i++) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<curve_fitting_cost, 1, 1, 1, 1, 1>(
                    new curve_fitting_cost(input[i], output[i])),
                nullptr, &para1, &para2, &para3, &para4);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::DOGLEG;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }

    log_debug("ground  truth: %lf  %lf  %lf  %lf", A, B, C, D);
    log_debug("dogleg  estim: %lf  %lf  %lf  %lf", params_dogLeg[0], params_dogLeg[1], params_dogLeg[2], params_dogLeg[3]);
    log_debug("     LM estim: %lf  %lf  %lf  %lf", params_levenMar[0], params_levenMar[1], params_levenMar[2], params_levenMar[3]);
    log_debug("  ceres estim: %lf  %lf  %lf  %lf", para1, para2, para3, para4);
}
