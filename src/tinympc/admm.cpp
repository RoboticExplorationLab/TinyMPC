#include <iostream>

#include "admm.hpp"

#include <tinympc/admm.hpp>
// #include "problem_data/quadrotor_50hz_params.hpp"
// #include "problem_data/quadrotor_50hz_params_3.hpp"
#include "problem_data/quadrotor_50hz_params_unconstrained.hpp"
#include "problem_data/quadrotor_50hz_params_constrained.hpp"
#include "trajectory_data/quadrotor_50hz_line_5s.hpp"
// #include "trajectory_data/quadrotor_50hz_line_9s.hpp"
using Eigen::Matrix;

// Global variables
struct tiny_cache cache;
struct tiny_params params;
struct tiny_problem problem;


#define DEBUG_MODULE "TINYALG"

extern "C" {

// #include "debug.h"

static uint64_t startTimestamp;

void c_call_test(float x[12][10]){
    x[0][0] = -85.0;
    x[1][0] = -99.0;
}

void __attribute__ ((constructor)) initLibrary(void) {
    //
    // Function that is called when the library is loaded
    //
    // Copy data from problem_data/quadrotor*.hpp
    cache.Adyn[0] = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Adyn_data);
    cache.Bdyn[0] = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(Bdyn_data);
    cache.rho[0] = rho_value;
    cache.Kinf[0] = Eigen::Map<Matrix<tinytype, NINPUTS, NSTATES, Eigen::RowMajor>>(Kinf_data);
    cache.Pinf[0] = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Pinf_data);
    cache.Quu_inv[0] = Eigen::Map<Matrix<tinytype, NINPUTS, NINPUTS, Eigen::RowMajor>>(Quu_inv_data);
    cache.AmBKt[0] = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(AmBKt_data);
    cache.coeff_d2p[0] = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(coeff_d2p_data);

    cache.Adyn[1] = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Adyn_data);
    cache.Bdyn[1] = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(Bdyn_data);
    cache.rho[1] = rho_constrained_value;
    cache.Kinf[1] = Eigen::Map<Matrix<tinytype, NINPUTS, NSTATES, Eigen::RowMajor>>(Kinf_constrained_data);
    cache.Pinf[1] = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Pinf_constrained_data);
    cache.Quu_inv[1] = Eigen::Map<Matrix<tinytype, NINPUTS, NINPUTS, Eigen::RowMajor>>(Quu_inv_constrained_data);
    cache.AmBKt[1] = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(AmBKt_constrained_data);
    cache.coeff_d2p[1] = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(coeff_d2p_constrained_data);

    params.Q[0] = Eigen::Map<tiny_VectorNx>(Q_data);
    params.Qf[0] = Eigen::Map<tiny_VectorNx>(Qf_data);
    params.R[0] = Eigen::Map<tiny_VectorNu>(R_data);
    params.Q[1] = Eigen::Map<tiny_VectorNx>(Q_constrained_data);
    params.Qf[1] = Eigen::Map<tiny_VectorNx>(Qf_constrained_data);
    params.R[1] = Eigen::Map<tiny_VectorNu>(R_constrained_data);
    tinytype u_hover[4] = {.65, .65, .65, .65};
    params.u_min = tiny_VectorNu(-u_hover[0], -u_hover[1], -u_hover[2], -u_hover[3]).replicate<1, NHORIZON-1>();
    params.u_max = tiny_VectorNu(1 - u_hover[0], 1 - u_hover[1], 1 - u_hover[2], 1 - u_hover[3]).replicate<1, NHORIZON-1>();
    for (int i=0; i<NHORIZON; i++) {
        params.x_min[i] = tiny_VectorNc::Constant(-1000); // Currently unused
        // params.x_max[i] = tiny_VectorNc::Zero();
        params.x_max[i] = tiny_VectorNc::Constant(1000);
        // params.x_max[i](0) = x_max_given[i];
        params.A_constraints[i] = tiny_MatrixNcNx::Zero();
        // for (int j=0; j<3; j++) {
        //     params.A_constraints[i](j) = A_ineq_given[j][i];
        // }
    }
    params.Xref = tiny_MatrixNxNh::Zero();
    params.Uref = tiny_MatrixNuNhm1::Zero();
    params.cache = cache;

    problem.x = tiny_MatrixNxNh::Zero();
    problem.q = tiny_MatrixNxNh::Zero();
    problem.p = tiny_MatrixNxNh::Zero();
    problem.v = tiny_MatrixNxNh::Zero();
    problem.vnew = tiny_MatrixNxNh::Zero();
    problem.g = tiny_MatrixNxNh::Zero();

    problem.u = tiny_MatrixNuNhm1::Zero();
    problem.r = tiny_MatrixNuNhm1::Zero();
    problem.d = tiny_MatrixNuNhm1::Zero();
    problem.z = tiny_MatrixNuNhm1::Zero();
    problem.znew = tiny_MatrixNuNhm1::Zero();
    problem.y = tiny_MatrixNuNhm1::Zero();

    problem.primal_residual_state = 0;
    problem.primal_residual_input = 0;
    problem.dual_residual_state = 0;
    problem.dual_residual_input = 0;
    problem.abs_tol = 0.001;
    problem.status = 0;
    problem.iter = 0;
    problem.max_iter = 20;
    problem.cache_level = 0; // 0 to use rho corresponding to inactive constraints (1 to use rho corresponding to active constraints)

    printf("Problem is initialized\n");
}

void julia_sim_wrapper_solve_lqr(float x[12][10], float u[4][9]){
    // Copy data from problem_data/quadrotor*.hpp
    struct tiny_cache cache;
    cache.Adyn[0] = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Adyn_data);
    cache.Bdyn[0] = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(Bdyn_data);
    cache.rho[0] = rho_value;
    cache.Kinf[0] = Eigen::Map<Matrix<tinytype, NINPUTS, NSTATES, Eigen::RowMajor>>(Kinf_data);
    cache.Pinf[0] = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Pinf_data);
    cache.Quu_inv[0] = Eigen::Map<Matrix<tinytype, NINPUTS, NINPUTS, Eigen::RowMajor>>(Quu_inv_data);
    cache.AmBKt[0] = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(AmBKt_data);
    cache.coeff_d2p[0] = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(coeff_d2p_data);

    struct tiny_params params;
    params.Q[0] = Eigen::Map<tiny_VectorNx>(Q_data);
    params.Qf[0] = Eigen::Map<tiny_VectorNx>(Qf_data);
    params.R[0] = Eigen::Map<tiny_VectorNu>(R_data);
    params.u_min = tiny_MatrixNuNhm1::Constant(-0.5);
    params.u_max = tiny_MatrixNuNhm1::Constant(0.5);
    for (int i=0; i<NHORIZON; i++) {
        params.x_min[i] = tiny_VectorNc::Constant(-99999); // Currently unused
        params.x_max[i] = tiny_VectorNc::Zero();
        params.A_constraints[i] = tiny_MatrixNcNx::Zero();
    }
    params.Xref = tiny_MatrixNxNh::Zero();
    params.Uref = tiny_MatrixNuNhm1::Zero();
    params.cache = cache;

    struct tiny_problem problem;
    problem.x = tiny_MatrixNxNh::Zero();
    problem.q = tiny_MatrixNxNh::Zero();
    problem.p = tiny_MatrixNxNh::Zero();
    problem.v = tiny_MatrixNxNh::Zero();
    problem.vnew = tiny_MatrixNxNh::Zero();
    problem.g = tiny_MatrixNxNh::Zero();

    problem.u = tiny_MatrixNuNhm1::Zero();
    problem.r = tiny_MatrixNuNhm1::Zero();
    problem.d = tiny_MatrixNuNhm1::Zero();
    problem.z = tiny_MatrixNuNhm1::Zero();
    problem.znew = tiny_MatrixNuNhm1::Zero();
    problem.y = tiny_MatrixNuNhm1::Zero();

    problem.primal_residual_state = 0;
    problem.primal_residual_input = 0;
    problem.dual_residual_state = 0;
    problem.dual_residual_input = 0;
    problem.abs_tol = 0.001;
    problem.status = 0;
    problem.iter = 0;
    problem.max_iter = 100;
    problem.iters_check_rho_update = 10;
    problem.cache_level = 0; // 0 to use rho corresponding to inactive constraints (1 to use rho corresponding to active constraints)

    // Copy reference trajectory into Eigen matrix
    // Matrix<tinytype, NSTATES, NTOTAL, Eigen::ColMajor> Xref_total = Eigen::Map<Matrix<tinytype, NTOTAL, NSTATES, Eigen::RowMajor>>(Xref_data).transpose();
    Matrix<tinytype, NSTATES, 1> Xref_origin;
    Xref_origin << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    // params.Xref = Xref_total.block<NSTATES, NHORIZON>(0,0);
    params.Xref = Xref_origin.replicate<1,NHORIZON>();
    // problem.x.col(0) = params.Xref.col(0);
    problem.x.col(0) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    int Nx = 12;
    int Nh = 10;
    int Nu = 4;
    Eigen::Map<tiny_MatrixNxNh> problem_x(&x[0][0], Nx, Nh);
    problem.x = problem_x;
    Eigen::Map<tiny_MatrixNuNhm1> problem_u(&u[0][0], Nu, Nh-1);
    problem.u = problem_u;

    // std::cout << params.Xref << std::endl;

    solve_lqr(&problem, &params);
    Eigen::Map<tiny_MatrixNuNhm1>(&u[0][0], Nu, Nh-1) = problem.u;
    Eigen::Map<tiny_MatrixNxNh>(&x[0][0], Nx+1, Nh) = problem.x;

    // std::cout << problem.iter << std::endl;
    std::cout << "LQR RESULTS ";
    std::cout << problem.u.col(0)(0) << " ";
    std::cout << problem.u.col(0)(1) << " ";
    std::cout << problem.u.col(0)(2) << " ";
    std::cout << problem.u.col(0)(3) << std::endl;

}


void julia_sim_wrapper_solve_admm(float x[NSTATES][NHORIZON], float u[NINPUTS][NHORIZON-1], int mpc_iter, float x_max_given[NHORIZON], float A_ineq_given[3][NHORIZON]){
    // Copy reference trajectory into Eigen matrix
    Matrix<tinytype, NSTATES, NTOTAL, Eigen::ColMajor> Xref_total = Eigen::Map<Matrix<tinytype, NTOTAL, NSTATES, Eigen::RowMajor>>(Xref_data).transpose();
    Matrix<tinytype, NSTATES, 1> Xref_origin;
    Xref_origin << Xref_total.col(0).head(3), 0, 0, 0, 0, 0, 0, 0, 0, 0; // Go to xyz start of traj
    
    params.Xref = Xref_total.block<NSTATES, NHORIZON>(0, mpc_iter);
    // params.Xref = Xref_origin.replicate<1,NHORIZON>();
    // problem.x.col(0) = params.Xref.col(0);
    // problem.x.col(0) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    // int Nx = 12;
    // int Nh = 25;
    // int Nu = 4;
    Eigen::Map<tiny_MatrixNxNh> problem_x(&x[0][0], NSTATES, NHORIZON);
    problem.x = problem_x;
    Eigen::Map<tiny_MatrixNuNhm1> problem_u(&u[0][0], NINPUTS, NHORIZON-1);
    problem.u = problem_u;

    // std::cout << params.Xref << std::endl;

    Eigen::Matrix<tinytype, 3, 1> obs_center;
    obs_center << 0, 0, .8;
    float r_obs = .5;

    Eigen::Matrix<tinytype, 3, 1> xc;
    Eigen::Matrix<tinytype, 3, 1> a_norm;
    Eigen::Matrix<tinytype, 3, 1> q_c;

    // Update constraint parameters
    for (int i=0; i<NHORIZON; i++) {
        xc = obs_center - problem.x.col(i).head(3);
        a_norm = xc / xc.norm();
        // params.A_constraints[i].block<1, 3>(0, 0) = a_norm.transpose();
        params.A_constraints[i].head(3) = a_norm.transpose();
        q_c = obs_center - r_obs*a_norm;
        params.x_max[i](0) = a_norm.transpose() * q_c;
    }
    solve_admm(&problem, &params);

    // std::cout << "done" << std::endl;
    // std::cout << problem.iter << std::endl;

    Eigen::Map<tiny_MatrixNuNhm1>(&u[0][0], NINPUTS, NHORIZON-1) = problem.u;
    Eigen::Map<tiny_MatrixNxNh>(&x[0][0], NSTATES, NHORIZON) = problem.x;
    Eigen::Map<Eigen::Matrix<tinytype, 3, NHORIZON>>(&A_ineq_given[0][0], 3, NHORIZON) = problem.xyz_news;
}


void multAdyn(tiny_VectorNx &Ax, const tiny_MatrixNxNx &A, const tiny_VectorNx &x) {
    Ax(0) = (x(0) + A(0,4)*x(4) + A(0,6)*x(6) + A(0,10)*x(10));
    Ax(1) = (x(1) + A(1,3)*x(3) + A(1,7)*x(7) + A(1,9)*x(9));
    Ax(2) = x(2) + A(2,8)*x(8);
    Ax(3) = x(3) + A(3,9)*x(9);
    Ax(4) = x(4) + A(4,10)*x(10);
    Ax(5) = x(5) + A(5,11)*x(11);
    Ax(6) = (x(6) + A(6,4)*x(4) + A(6,10)*x(10));
    Ax(7) = (x(7) + A(7,3)*x(3) + A(7,9)*x(9));
    Ax(8) = x(8);
    Ax(9) = x(9);
    Ax(10) = x(10);
    Ax(11) = x(11);
}

void solve_lqr(struct tiny_problem *problem, const struct tiny_params *params) {
    problem->u.col(0) = -params->cache.Kinf[problem->cache_level] * (problem->x.col(0) - params->Xref.col(0));
}


void solve_admm(struct tiny_problem *problem, const struct tiny_params *params) {

    problem->status = 0;
    problem->iter = 0;

    forward_pass(problem, params);
    update_slack(problem, params);
    update_dual(problem, params);
    update_linear_cost(problem, params);
    // std::cout << problem->r << std::endl;
    // std::cout << problem->q << std::endl;
    // std::cout << problem->p << std::endl;
    for (int i=0; i<problem->max_iter; i++) {
    // for (int i=0; i<1; i++) {
    // while (usecTimestamp() + 300 < maxTime) {

        // Solve linear system with Riccati and roll out to get new trajectory
        update_primal(problem, params);
        // backward_pass_grad(problem, params);
        // std::cout << problem->d << std::endl;
        // std::cout << problem->p << std::endl;
        // forward_pass(problem, params);
        // std::cout << problem->u << std::endl;
        // std::cout << problem->x << std::endl;

        // Project slack variables into feasible domain
        update_slack(problem, params);

        // Compute next iteration of dual variables
        update_dual(problem, params);

        // Update linear control cost terms using reference trajectory, duals, and slack variables
        update_linear_cost(problem, params);

        problem->primal_residual_state = (problem->x - problem->vnew).cwiseAbs().maxCoeff();
        problem->dual_residual_state = ((problem->v - problem->vnew).cwiseAbs().maxCoeff()) * params->cache.rho[problem->cache_level];
        problem->primal_residual_input = (problem->u - problem->znew).cwiseAbs().maxCoeff();
        problem->dual_residual_input = ((problem->z - problem->znew).cwiseAbs().maxCoeff()) * params->cache.rho[problem->cache_level];

        // TODO: convert arrays of Eigen vectors into one Eigen matrix
        // Save previous slack variables
        problem->v = problem->vnew;
        problem->z = problem->znew;

        problem->iter += 1;

        // TODO: remove convergence check and just return when allotted runtime is up
        // Check for convergence
        if (problem->primal_residual_state < problem->abs_tol &&
            problem->primal_residual_input < problem->abs_tol &&
            problem->dual_residual_state < problem->abs_tol &&
            problem->dual_residual_input < problem->abs_tol)
        {
            problem->status = 1;
            break;
        }

        // TODO: add rho scaling


        // std::cout << problem->primal_residual_state << std::endl;
        // std::cout << problem->dual_residual_state << std::endl;
        // std::cout << problem->primal_residual_input << std::endl;
        // std::cout << problem->dual_residual_input << "\n" << std::endl;
    }
}

/**
 * Do backward Riccati pass then forward roll out
*/
void update_primal(struct tiny_problem *problem, const struct tiny_params *params) {
    backward_pass_grad(problem, params);
    forward_pass(problem, params);
}

/**
 * Update linear terms from Riccati backward pass
*/
void backward_pass_grad(struct tiny_problem *problem, const struct tiny_params *params) {
    for (int i=NHORIZON-2; i>=0; i--) {
        // problem->Qu.noalias() = params->cache.Bdyn.transpose().lazyProduct(problem->p.col(i+1));
        // problem->Qu += problem->r.col(i);
        // (problem->d.col(i)).noalias() = params->cache.Quu_inv.lazyProduct(problem->Qu);
        (problem->d.col(i)).noalias() = params->cache.Quu_inv[problem->cache_level] * (params->cache.Bdyn[problem->cache_level].transpose() * problem->p.col(i+1) + problem->r.col(i));
        (problem->p.col(i)).noalias() = problem->q.col(i) + params->cache.AmBKt[problem->cache_level].lazyProduct(problem->p.col(i+1)) - (params->cache.Kinf[problem->cache_level].transpose()).lazyProduct(problem->r.col(i)) + params->cache.coeff_d2p[problem->cache_level] * problem->d.col(i);
    }
}

/**
 * Use LQR feedback policy to roll out trajectory
*/
void forward_pass(struct tiny_problem *problem, const struct tiny_params *params) {
    for (int i=0; i<NHORIZON-1; i++) {
        (problem->u.col(i)).noalias() = -params->cache.Kinf[problem->cache_level].lazyProduct(problem->x.col(i)) - problem->d.col(i);
        // problem->u.col(i) << .001, .02, .3, 4;
        // DEBUG_PRINT("u(0): %f\n", problem->u.col(0)(0));
        multAdyn(problem->Ax, params->cache.Adyn[problem->cache_level], problem->x.col(i));
        (problem->x.col(i+1)).noalias() = problem->Ax + params->cache.Bdyn[problem->cache_level].lazyProduct(problem->u.col(i));
        // (problem->x.col(i+1)).noalias() = params->cache.Adyn.lazyProduct(problem->x.col(i)) + params->cache.Bdyn.lazyProduct(problem->u.col(i));
    }
}

/**
 * Project slack (auxiliary) variables into their feasible domain, defined by
 * projection functions related to each constraint
 * TODO: pass in meta information with each constraint assigning it to a
 * projection function
*/
void update_slack(struct tiny_problem *problem, const struct tiny_params *params) {
    // Box constraints on input
    // Get current time

    problem->znew = params->u_max.cwiseMin(params->u_min.cwiseMax(problem->u));

    // Half space constraints on state
    // TODO: support multiple half plane constraints per knot point
    //      currently this only works for one constraint per knot point
    // TODO: can potentially take advantage of the fact that A_constraints[3:end] is zero and just do
    //      v.col(i) = x.col(i) - dist*A_constraints[i] since we have to copy x[3:end] into v anyway
    //      downside is it's not clear this is happening externally and so values of A_constraints
    //      not set to zero (other than the first three) can cause the algorithm to fail
    // TODO: the only state values changing here are the first three (x, y, z) so it doesn't make sense
    //      to do operations on the remaining 9 when projecting (or doing anything related to the dual
    //      or auxiliary variables). v and g could be of size (3) and everything would work the same.
    //      The only reason this doesn't break is because in the update_linear_cost function subtracts
    //      g from v and so the last nine entries are always zero.
    problem->xg = problem->x + problem->g;
    // problem->dists = (params->A_constraints.transpose().cwiseProduct(problem->xg)).colwise().sum();
    // problem->dists -= params->x_max;
    // startTimestamp = usecTimestamp();
    problem->cache_level = 0;
    for (int i=0; i<NHORIZON; i++) {
        problem->dist = (params->A_constraints[i].head(3)).lazyProduct(problem->xg.col(i).head(3)); // Distances can be computed in one step outside the for loop
        problem->dist -= params->x_max[i](0);
        problem->xyz_news.col(i) = problem->xg.col(i).head(3) - problem->dist*params->A_constraints[i].head(3).transpose();
        // DEBUG_PRINT("dist: %f\n", dist);
        if (problem->dist <= 0) {
            problem->vnew.col(i) = problem->xg.col(i);
        }
        else {
            problem->cache_level = 1; // Constraint violated, use second cache level
            problem->xyz_new = problem->xg.col(i).head(3) - problem->dist*params->A_constraints[i].head(3).transpose();
            problem->vnew.col(i) << problem->xyz_new, problem->xg.col(i).tail(NSTATES-3);
        }
    }
    // problem->vnew = problem->xg;
    // DEBUG_PRINT("s: %d\n", usecTimestamp() - startTimestamp);
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
*/
void update_dual(struct tiny_problem *problem, const struct tiny_params *params) {
    problem->y = problem->y + problem->u - problem->znew;
    problem->g = problem->g + problem->x - problem->vnew;
}

/**
 * Update linear control cost terms in the Riccati feedback using the changing
 * slack and dual variables from ADMM
*/
void update_linear_cost(struct tiny_problem *problem, const struct tiny_params *params) {
    // problem->r = -(params->Uref.array().colwise() * params->R[problem->cache_level].array()); // Uref = 0 so commented out for speed up. Need to uncomment if using Uref
    problem->r = -params->cache.rho[problem->cache_level] * (problem->znew - problem->y);
    problem->q = -(params->Xref.array().colwise() * params->Q[problem->cache_level].array());
    problem->q -= params->cache.rho[problem->cache_level] * (problem->vnew - problem->g);
    // problem->p.col(NHORIZON-1) = -(params->Xref.col(NHORIZON-1).array().colwise() * params->Qf.array());
    problem->p.col(NHORIZON-1) = -(params->Xref.col(NHORIZON-1).transpose().lazyProduct(params->cache.Pinf[problem->cache_level]));
    problem->p.col(NHORIZON-1) -= params->cache.rho[problem->cache_level] * (problem->vnew.col(NHORIZON-1) - problem->g.col(NHORIZON-1));

    // for (int i=0; i<NHORIZON-1; i++) {
    //     problem->r.col(i) = -params->cache.rho * (problem->znew.col(i) - problem->y.col(i)) - params->R * params->Uref.col(i);
    //     problem->q.col(i) = -params->cache.rho * (problem->vnew.col(i) - problem->g.col(i)) - params->Q * params->Xref.col(i);
    // }
    // problem->p.col(NHORIZON-1) = -params->cache.rho * (problem->vnew.col(NHORIZON-1) - problem->g.col(NHORIZON-1)) - params->Qf * params->Xref.col(NHORIZON-1);
}


// void solve_admm(struct tiny_problem *problem, const struct tiny_params *params) {

//     problem->status = 0;
//     problem->iter = 1;

//     forward_pass(problem, params);
//     update_slack(problem, params);
//     update_dual(problem, params);
//     update_linear_cost(problem, params);
//     for (int i=0; i<problem->max_iter; i++) {

//         // Solve linear system with Riccati and roll out to get new trajectory
//         update_primal(problem, params);

//         // Project slack variables into feasible domain
//         update_slack(problem, params);

//         // Compute next iteration of dual variables
//         update_dual(problem, params);

//         // Update linear control cost terms using reference trajectory, duals, and slack variables
//         update_linear_cost(problem, params);

//         problem->primal_residual_state = (problem->x - problem->vnew).cwiseAbs().maxCoeff();
//         problem->dual_residual_state = ((problem->v - problem->vnew).cwiseAbs().maxCoeff()) * params->cache.rho;
//         problem->primal_residual_input = (problem->u - problem->znew).cwiseAbs().maxCoeff();
//         problem->dual_residual_input = ((problem->z - problem->znew).cwiseAbs().maxCoeff()) * params->cache.rho;

//         // TODO: convert arrays of Eigen vectors into one Eigen matrix
//         // Save previous slack variables
//         problem->v = problem->vnew;
//         problem->z = problem->znew;

//         // TODO: remove convergence check and just return when allotted runtime is up
//         // Check for convergence
//         if (problem->primal_residual_state < problem->abs_tol &&
//             problem->primal_residual_input < problem->abs_tol &&
//             problem->dual_residual_state < problem->abs_tol &&
//             problem->dual_residual_input < problem->abs_tol)
//         {
//             problem->status = 1;
//             break;
//         }

//         // TODO: add rho scaling

//         problem->iter += 1;

//         // std::cout << problem->primal_residual_state << std::endl;
//         // std::cout << problem->dual_residual_state << std::endl;
//         // std::cout << problem->primal_residual_input << std::endl;
//         // std::cout << problem->dual_residual_input << "\n" << std::endl;
//     }
// }

// /**
//  * Do backward Riccati pass then forward roll out
// */
// void update_primal(struct tiny_problem *problem, const struct tiny_params *params) {
//     backward_pass_grad(problem, params);
//     forward_pass(problem, params);
// }

// /**
//  * Update linear terms from Riccati backward pass
// */
// void backward_pass_grad(struct tiny_problem *problem, const struct tiny_params *params) {
//     for (int i=NHORIZON-2; i>=0; i--) {
//         // problem->Qu.noalias() = params->cache.Bdyn.transpose().lazyProduct(problem->p.col(i+1));
//         // problem->Qu += problem->r.col(i);
//         // (problem->d.col(i)).noalias() = params->cache.Quu_inv.lazyProduct(problem->Qu);
//         (problem->d.col(i)).noalias() = params->cache.Quu_inv * (params->cache.Bdyn.transpose() * problem->p.col(i+1) + problem->r.col(i));
//         (problem->p.col(i)).noalias() = problem->q.col(i) + params->cache.AmBKt.lazyProduct(problem->p.col(i+1)) - (params->cache.Kinf.transpose()).lazyProduct(problem->r.col(i)); // + params->cache.coeff_d2p * problem->d.col(i); // coeff_d2p always appears to be zeros
//     }
// }

// /**
//  * Use LQR feedback policy to roll out trajectory
// */
// void forward_pass(struct tiny_problem *problem, const struct tiny_params *params) {
//     for (int i=0; i<NHORIZON-1; i++) {
//         (problem->u.col(i)).noalias() = -params->cache.Kinf.lazyProduct(problem->x.col(i)) - problem->d.col(i);
//         // problem->u.col(i) << .001, .02, .3, 4;
//         // DEBUG_PRINT("u(0): %f\n", problem->u.col(0)(0));
//         multAdyn(problem->Ax, params->cache.Adyn, problem->x.col(i));
//         (problem->x.col(i+1)).noalias() = problem->Ax + params->cache.Bdyn.lazyProduct(problem->u.col(i));
//     }
// }

// /**
//  * Project slack (auxiliary) variables into their feasible domain, defined by
//  * projection functions related to each constraint
//  * TODO: pass in meta information with each constraint assigning it to a
//  * projection function
// */
// void update_slack(struct tiny_problem *problem, const struct tiny_params *params) {
//     // Box constraints on input
//     // Get current time

//     problem->znew = params->u_max.cwiseMin(params->u_min.cwiseMax(problem->u));

//     // Half space constraints on state
//     // TODO: support multiple half plane constraints per knot point
//     //      currently this only works for one constraint per knot point
//     // TODO: can potentially take advantage of the fact that A_constraints[3:end] is zero and just do
//     //      v.col(i) = x.col(i) - dist*A_constraints[i] since we have to copy x[3:end] into v anyway
//     //      downside is it's not clear this is happening externally and so values of A_constraints
//     //      not set to zero (other than the first three) can cause the algorithm to fail
//     // TODO: the only state values changing here are the first three (x, y, z) so it doesn't make sense
//     //      to do operations on the remaining 9 when projecting (or doing anything related to the dual
//     //      or auxiliary variables). v and g could be of size (3) and everything would work the same.
//     //      The only reason this doesn't break is because in the update_linear_cost function subtracts
//     //      g from v and so the last nine entries are always zero.
//     // startTimestamp = usecTimestamp();
//     problem->xg = problem->x + problem->g;
//     // problem->dists = (params->A_constraints.transpose().cwiseProduct(problem->xg)).colwise().sum();
//     // problem->dists -= params->x_max;
//     for (int i=0; i<NHORIZON; i++) {
//         problem->dist = (params->A_constraints[i].head(3)).lazyProduct(problem->xg.col(i).head(3)); // Distances can be computed in one step outside the for loop
//         problem->dist -= params->x_max[i](0);
//         // DEBUG_PRINT("dist: %f\n", dist);
//         if (problem->dist <= 0) {
//             problem->vnew.col(i) = problem->xg.col(i);
//         }
//         else {
//             problem->xyz_new = problem->xg.col(i).head(3) - problem->dist*params->A_constraints[i].head(3).transpose();
//             problem->vnew.col(i) << problem->xyz_new, problem->xg.col(i).tail(NSTATES-3);

//             // if (i == NHORIZON-1) {
//             // if (i == 0) {
//             //     std::cout << "prev xyz: "   << problem->x.col(i).head(3)   << std::endl;
//             //     std::cout << "vnew: "       << problem->vnew.col(i).head(3) << std::endl;
//             // }
//         }
        
//         problem->xyz_news.col(i) = problem->xg.col(i).head(3) - problem->dist*params->A_constraints[i].head(3).transpose();
//     }
//     // DEBUG_PRINT("slack: %d\n", usecTimestamp() - startTimestamp);
// }

// /**
//  * Update next iteration of dual variables by performing the augmented
//  * lagrangian multiplier update
// */
// void update_dual(struct tiny_problem *problem, const struct tiny_params *params) {
//     problem->y = problem->y + problem->u - problem->znew;
//     problem->g = problem->g + problem->x - problem->vnew;
// }

// /**
//  * Update linear control cost terms in the Riccati feedback using the changing
//  * slack and dual variables from ADMM
// */
// void update_linear_cost(struct tiny_problem *problem, const struct tiny_params *params) {
//     // problem->r = -(params->Uref.array().colwise() * params->R.array()); // Uref = 0 so commented out for speed up. Need to uncomment if using Uref
//     problem->r = -params->cache.rho * (problem->znew - problem->y);
//     problem->q = -(params->Xref.array().colwise() * params->Q.array());
//     problem->q -= params->cache.rho * (problem->vnew - problem->g);
//     // problem->p.col(NHORIZON-1) = -(params->Xref.col(NHORIZON-1).array().colwise() * params->Qf.array());
//     problem->p.col(NHORIZON-1) = -(params->Xref.col(NHORIZON-1).transpose().lazyProduct(params->cache.Pinf));
//     problem->p.col(NHORIZON-1) -= params->cache.rho * (problem->vnew.col(NHORIZON-1) - problem->g.col(NHORIZON-1));

//     // for (int i=0; i<NHORIZON-1; i++) {
//     //     problem->r.col(i) = -params->cache.rho * (problem->znew.col(i) - problem->y.col(i)) - params->R * params->Uref.col(i);
//     //     problem->q.col(i) = -params->cache.rho * (problem->vnew.col(i) - problem->g.col(i)) - params->Q * params->Xref.col(i);
//     // }
//     // problem->p.col(NHORIZON-1) = -params->cache.rho * (problem->vnew.col(NHORIZON-1) - problem->g.col(NHORIZON-1)) - params->Qf * params->Xref.col(NHORIZON-1);
// }



} /* extern "C" */