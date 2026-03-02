#include <tests/config.hpp>
#include <tests/utils.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <ipc/potentials/barrier_potential.hpp>
#include <ipc/high_order_contact/high_order_contact_potential.hpp>
#include <ipc/distance/line_line.hpp>

#include <finitediff.hpp>
#include <igl/edges.h>
#include <igl/readCSV.h>
#include <ipc/ipc.hpp>

#include "igl/read_triangle_mesh.h"
#include "igl/write_triangle_mesh.h"
#include "ipc/distance/edge_edge.hpp"

#include "ipc/high_order_contact/quadrature_potential.hpp"
#include "ipc/smooth_contact/distance/mollifier.hpp"

using namespace ipc;

// When the edge-edge closest point approaches the end points of the edge, the potential should converge to a finite number
TEST_CASE("Convergent Quadrature Edge Edge Limit", "[high_order_potential]")
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F, E;

    double epsilon = GENERATE(1e-3, 1e-4, 1e-6, 1e-12, 1e-16);
    {
        V.resize(8, 3);
        V <<
            0, 0, 0,
            1, 0, 0,
            0.5, -0.5, 1,
            0.5, 0.5, 1,
            epsilon, 0.5, -0.01,
            epsilon, -0.5, -0.01,
            epsilon + 0.5, 0, -1.01,
            epsilon - 0.5, 0, -1.01;

        F.resize(8, 3);
        F <<
            0, 1, 2,
            0, 1, 3,
            0, 2, 3,
            1, 2, 3,
            4, 5, 6,
            4, 6, 7,
            4, 5, 7,
            5, 6, 7;

        E.resize(12, 2);
        E <<
            0, 1,
            0, 2,
            0, 3,
            1, 2,
            1, 3,
            2, 3,
            4, 5,
            4, 6,
            4, 7,
            5, 6,
            5, 7,
            6, 7;
    }

    CollisionMesh mesh(V, E, F);

    const double dhat = 0.1;

    HighOrderContactParameters params(dhat, 0., 2, 0);

    HighOrderCollisions collisions;
    collisions.build(mesh, V, params);

    HighOrderContactPotential potential(params);

    const index_t e0 = 0;
    const index_t e1 = 6;

    auto dtype = edge_edge_distance_type(
        V.row(mesh.edges()(e0, 0)),
        V.row(mesh.edges()(e0, 1)),
        V.row(mesh.edges()(e1, 0)),
        V.row(mesh.edges()(e1, 1)));

    REQUIRE(dtype == EdgeEdgeDistanceType::EA_EB);

    Eigen::VectorXd g = potential.gradient(collisions, mesh, V);

    double x = potential(collisions, mesh, V);

    // These numbers can be changed as the formulation changes, but they shouldn't be extremely large
    REQUIRE(abs(x) < 2);
    REQUIRE(g.norm() < 200);
}

TEST_CASE("Convergent Quadrature Hessian", "[high_order_potential]")
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F, E;
    igl::read_triangle_mesh((tests::DATA_DIR / "../src/tests/potential/wrapped_sphere.obj").string(), V, F);

    igl::edges(F, E);
    CollisionMesh mesh(V, E, F);

    const double dhat = 0.15;
    HighOrderContactParameters params(dhat, 0., 2, 0);

    HighOrderCollisions collisions;
    collisions.build(mesh, V, params);

    HighOrderContactPotential potential(params);

    Eigen::MatrixXd h = potential.hessian(collisions, mesh, V);

    // full finite difference is too expensive, verify directional derivative only
    Eigen::VectorXd test_dir(V.size(), 1);
    for (int i = 0; i < test_dir.size(); i++) {
        test_dir(i) = i;
    }

    Eigen::MatrixXd fh;
    fd::finite_jacobian(
        Eigen::VectorXd::Zero(1), [&](const Eigen::VectorXd& y) {
            Eigen::MatrixXd V_ = V + fd::unflatten(test_dir, 3) * y(0);
            HighOrderCollisions collisions_;
            collisions_.build(mesh, V_, params);
            return potential.gradient(collisions_, mesh, V_);
        }, fh, fd::AccuracyOrder::SECOND, 1e-8);

    REQUIRE((fh.col(0) - h * test_dir).norm() < fh.norm() * 1e-4);
}

TEST_CASE("Convergent Quadrature Gradient", "[high_order_potential]")
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F, E;
    igl::read_triangle_mesh((tests::DATA_DIR / "../src/tests/potential/wrapped_sphere.obj").string(), V, F);

    igl::edges(F, E);
    CollisionMesh mesh(V, E, F);

    const double dhat = 0.15;
    HighOrderContactParameters params(dhat, 0., 2, 0);

    HighOrderCollisions collisions;
    collisions.build(mesh, V, params);

    HighOrderContactPotential potential(params);

    Eigen::VectorXd g = potential.gradient(collisions, mesh, V);

    // full finite difference is too expensive, verify directional derivative only
    Eigen::VectorXd test_dir(V.size(), 1);
    for (int i = 0; i < test_dir.size(); i++) {
        test_dir(i) = i;
    }

    Eigen::VectorXd fg;
    fd::finite_gradient(
        Eigen::VectorXd::Zero(1), [&](const Eigen::VectorXd& y) {
            Eigen::MatrixXd V_ = V + fd::unflatten(test_dir, 3) * y(0);
            HighOrderCollisions collisions_;
            collisions_.build(mesh, V_, params);
            return potential(collisions_, mesh, V_);
        }, fg, fd::AccuracyOrder::SECOND, 1e-8);

    REQUIRE(abs(fg(0) - g.dot(test_dir)) < fg.norm() * 1e-6);
}


TEST_CASE("Formulation Discontinuity", "[high_order_potential]")
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F, E;
    {
        V.resize(12, 3);
        V << 0.50367684867164086437,  2.93648750371529354553, -1.00004231848768010416, -0.49633255318502039755,  2.94141655389999590042, -1.00001938593733363803, -0.01013768181206405744, -0.06105867429777982885, -1.00004343059451317188,  0.50370983406540503768,  2.93645083403066831096,  1.00003860343532968713, -0.01014685591357166079, -0.06103229407236412246,  1.00001543282668614587, -0.49635840713946227654,  2.94139600967862779868,  1.00001842376878191665, -0.50000000000000000000, -3.00000000000000000000,  1.00000000000000000000,  0.00000000000000000000,  0.00000000000000000000,  1.00000000000000000000, -0.50000000000000000000, -3.00000000000000000000, -1.00000000000000000000,  0.00000000000000000000,  0.00000000000000000000, -1.00000000000000000000,  0.50000000000000000000, -3.00000000000000000000,  1.00000000000000000000,  0.50000000000000000000, -3.00000000000000000000, -1.00000000000000000000;
        F.resize(3, 16);
        F <<
            0, 1, 3, 1, 5, 5, 4, 4, 7, 7, 11, 7, 8, 7, 7, 11,
            2, 3, 2, 5, 1, 4, 1, 2, 8, 11, 6, 6, 9, 9, 10, 10,
            1, 0, 0, 3, 4, 3, 2, 3, 6, 9, 8, 10, 11, 8, 11, 6;
        F.transposeInPlace();
    }

    igl::edges(F, E);
    CollisionMesh mesh(V, E, F);

    const double dhat = 1e-3;
    HighOrderContactParameters params(dhat, 0., 2, 0);

    HighOrderCollisions collisions;
    collisions.build(mesh, V, params);

    HighOrderContactPotential potential(params);

    Eigen::VectorXd g = potential.gradient(collisions, mesh, V);

    {
        Eigen::MatrixXd Vy = V;
        Vy << 0.50366769744905004469,  2.93648907641392797885, -1.00004231830756462607, -0.49633255318502422782,  2.94141655389999634451, -1.00001938593733430416, -0.01057866928878470510, -0.06098288765878417256, -1.00004342191498118986,  0.50370073867773879073,  2.93645239726239815070,  1.00003860342200678879, -0.01058437405459194611, -0.06095709746440926280,  1.00001543217581212453, -0.49635840713946338676,  2.94139600967862779868,  1.00001842376878236074, -0.50000000000000000000, -3.00000000000000000000,  1.00000000000000000000,  0.00000000000000000000,  0.00000000000000000000,  1.00000000000000000000, -0.50000000000000000000, -3.00000000000000000000, -1.00000000000000000000,  0.00000000000000000000,  0.00000000000000000000, -1.00000000000000000000,  0.50000000000000000000, -3.00000000000000000000,  1.00000000000000000000,  0.50000000000000000000, -3.00000000000000000000, -1.00000000000000000000;
        HighOrderCollisions collisions_;
        collisions_.build(mesh, Vy, params);

        std::cout << "distance between x and y " << (V - Vy).cwiseAbs().maxCoeff() << std::endl;
        std::cout << "value at x " << potential(collisions, mesh, V) << std::endl;
        std::cout << "value at y " << potential(collisions_, mesh, Vy) << std::endl;
    }

    Eigen::VectorXd fg;
    fd::finite_gradient(
        fd::flatten(V), [&](const Eigen::VectorXd& y) {
            Eigen::MatrixXd V_ = fd::unflatten(y, 3);
            HighOrderCollisions collisions_;
            collisions_.build(mesh, V_, params);
            return potential(collisions_, mesh, V_);
        }, fg, fd::AccuracyOrder::SECOND, 1e-6);

    REQUIRE((fg - g).norm() < fg.norm() * 1e-6);
}

TEST_CASE("Convergent Quadrature Zero on Sphere", "[high_order_potential]")
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F, E;
    igl::read_triangle_mesh((tests::DATA_DIR / "../src/tests/potential/sphere.obj").string(), V, F);

    igl::edges(F, E);
    CollisionMesh mesh(V, E, F);

    const double dhat = 0.2;
    HighOrderContactParameters params(dhat, 0., 2, 0);

    HighOrderCollisions collisions;
    collisions.build(mesh, V, params);

    HighOrderContactPotential potential(params);
    double val = potential(collisions, mesh, V);
    REQUIRE(val == 0);

    auto g = potential.gradient(collisions, mesh, V);
    REQUIRE(g.norm() == 0);

    auto H = potential.hessian(collisions, mesh, V);
    REQUIRE(H.norm() == 0);
}


TEST_CASE("Convergent Quadrature Vertex Hessian", "[high_order_potential]")
{
    const auto method = make_default_broad_phase();
    Eigen::MatrixXd V;
    Eigen::MatrixXi F, E;
    igl::read_triangle_mesh((tests::DATA_DIR / "../src/tests/potential/wrapped_sphere.obj").string(), V, F);

    igl::edges(F, E);
    CollisionMesh mesh(V, E, F);

    const double dhat = 0.15;
    HighOrderContactParameters params(dhat, 0., 2, 0);

    Candidates candidates;
    candidates.build(mesh, V, dhat / 2, method.get(), true);
    candidates.convert_candidates_to_sets();
    PointPotential point_potential(mesh, candidates, params);

    for (int vid = 0; vid < V.rows(); ++vid) {
        const auto collisions = point_potential.build_collisions_at_vertex(V, vid);

        if (collisions.size() == 0) {
            continue;
        }

        std::vector<int> indices;
        {
            Eigen::VectorXd local_grad =
                PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(
                    V, collisions, params);
            indices = collisions.dofs();

            if (local_grad.norm() < 1e-10) {
                continue;
            }
        }

        Eigen::MatrixXd h = PointPotentialHelper::evaluate_potential_hessian_at_vertex_with_cached_collisions(
            V, collisions, params, PSDProjectionMethod::NONE);
        h = h(indices, indices).eval();

        Eigen::MatrixXd fh;
        fd::finite_jacobian(
            fd::flatten(V)(indices), [&](const Eigen::VectorXd& y) {
                Eigen::VectorXd y_ = fd::flatten(V);
                y_(indices) = y;
                Eigen::MatrixXd V_fd = fd::unflatten(y_, 3);

                return PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(
                    V_fd, collisions, params);
            }, fh, fd::AccuracyOrder::SECOND, 1e-8);

        REQUIRE((h - fh).norm() < 1e-6 * std::max({h.norm(), fh.norm(), 1e-8}));
    }
}

TEST_CASE("Convergent Quadrature Face Hessian", "[high_order_potential]")
{
    const auto method = make_default_broad_phase();
    Eigen::MatrixXd V;
    Eigen::MatrixXi F, E;
    igl::read_triangle_mesh((tests::DATA_DIR / "../src/tests/potential/wrapped_sphere.obj").string(), V, F);

    igl::edges(F, E);
    CollisionMesh mesh(V, E, F);

    const double dhat = 0.15;
    HighOrderContactParameters params(dhat, 0., 2, 0);

    Candidates candidates;
    candidates.build(mesh, V, dhat / 2, method.get(), true);
    candidates.convert_candidates_to_sets();
    PointPotential point_potential(mesh, candidates, params);

    for (int fid = 0; fid < F.rows(); ++fid) {

        const auto collisions = point_potential.build_collisions_at_face_center(V, fid);

        if (collisions.size() == 0) {
            continue;
        }

        Eigen::Vector3<index_t> vids;
        vids << F(fid, 0), F(fid, 1), F(fid, 2);

        Eigen::RowVector3d face_center = (V.row(vids[0]) + V.row(vids[1]) + V.row(vids[2])) / 3.;

        ConcatMatrixView<3> V_extended(V, face_center);

        std::vector<int> indices;
        {
            Eigen::VectorXd local_grad =
                PointPotentialHelper::evaluate_potential_gradient_at_face_center_with_cached_collisions(
                    V_extended, collisions, params);
            indices = collisions.dofs();

            if (local_grad.norm() < 1e-10) {
                continue;
            }
        }

        Eigen::MatrixXd h = PointPotentialHelper::evaluate_potential_hessian_at_face_center_with_cached_collisions(V_extended, vids, collisions, params, PSDProjectionMethod::NONE);
        h = h(indices, indices).eval();

        Eigen::MatrixXd fh;
        fd::finite_jacobian(
            fd::flatten(V)(indices), [&](const Eigen::VectorXd& y) {
                Eigen::VectorXd y_ = fd::flatten(V);
                y_(indices) = y;
                Eigen::MatrixXd V_fd = fd::unflatten(y_, 3);
                Eigen::RowVector3d face_center_fd = (V_fd.row(vids[0]) + V_fd.row(vids[1]) + V_fd.row(vids[2])) / 3.;
                ConcatMatrixView<3> V_fd_extended(V_fd, face_center_fd);

                return PointPotentialHelper::evaluate_potential_gradient_at_face_center_with_cached_collisions(V_fd_extended, collisions, params);
            }, fh, fd::AccuracyOrder::SECOND, 1e-8);

        REQUIRE((h - fh).norm() < 1e-6 * std::max({h.norm(), fh.norm(), 1e-8}));
    }
}


// 2D TESTS //

TEST_CASE("High order potential 2D no forces", "[high_order_potential], [high_order_potential_2d]")
{
    const auto method = make_default_broad_phase();
    Eigen::MatrixXd V;
    Eigen::MatrixXi E;
    double dhat = 1.;
    const int quadrature_order = GENERATE(2, 4, 10, 20);
    HighOrderContactParameters params(dhat, 0., 1, quadrature_order);

    SECTION("single_square")
    {
        INFO("single_square");
        V.resize(4, 2);
        E.resize(4, 2);
        V <<
            -1., -1.,
            1., -1.,
            1., 1.,
            -1., 1.;
        E <<
            0, 1,
            1, 2,
            2, 3,
            3, 0;
    }
    SECTION("single_square_2")
    {
        INFO("single_square_2");
        V.resize(8, 2);
        E.resize(8, 2);
        V <<
            -1., -1.,
            0., -1.,
            1., -1.,
            1., 0.,
            1., 1.,
            0., 1.,
            -1., 1.,
            -1., 0.;
        E <<
            0, 1,
            1, 2,
            2, 3,
            3, 4,
            4, 5,
            5, 6,
            6, 7,
            7, 0;
    }
    SECTION("circle") {
        INFO("circle");
        const int n = GENERATE(10, 50, 100, 200);
        V.resize(n, 2);
        E.resize(n, 2);
        for (int i = 0; i < n; i++) {
            V(i, 0) = std::cos(2 * M_PI * i / n);
            V(i, 1) = std::sin(2 * M_PI * i / n);
        }
        for (int i = 0; i < n; i++) {
            E(i, 0) = i;
            E(i, 1) = (i + 1) % n;
        }
    }

    Eigen::MatrixXi F;
    CollisionMesh mesh(
        std::vector<bool>(V.rows(), true),
        std::vector<bool>(V.rows(), false), V, E, F);

    HighOrderCollisions collisions;
    collisions.build(mesh, V, params, false, method.get());

    REQUIRE(!has_intersections(mesh, V));

    HighOrderContactPotential potential(params);
    double energy = potential(collisions, mesh, V);
    CAPTURE(quadrature_order);
    CHECK(energy == 0);

    Eigen::VectorXd grad = potential.gradient(collisions, mesh, V);
    CHECK(grad.squaredNorm() == 0);

    Eigen::MatrixXd hess = potential.hessian(collisions, mesh, V);
    CHECK(hess.squaredNorm() == 0);
}

TEST_CASE("High order potential 2D finite differences", "[high_order_potential], [high_order_potential_2d]")
{
    const auto method = make_default_broad_phase();
    Eigen::MatrixXd V;
    Eigen::MatrixXi E;
    double dhat = 0.6;
    constexpr double BA = 1e-7; // a small constant to break perfect alignments
    const int quadrature_order = GENERATE(2, 4, 10, 20);
    HighOrderContactParameters params(dhat, 0., 1, quadrature_order);
    CAPTURE(quadrature_order);

    auto run_checks = [&]() {
        Eigen::MatrixXi F;
        CollisionMesh mesh(
            std::vector<bool>(V.rows(), true),
            std::vector<bool>(V.rows(), false), V, E, F);

        HighOrderCollisions collisions;
        collisions.build(mesh, V, params, false, method.get());

        REQUIRE(!collisions.empty());
        REQUIRE(!has_intersections(mesh, V));

        HighOrderContactPotential potential(params);
        double energy = potential(collisions, mesh, V);
        CHECK(energy > 0);

        Eigen::VectorXd grad = potential.gradient(collisions, mesh, V);
        REQUIRE(grad.squaredNorm() > 1e-8);
        Eigen::VectorXd fgrad;
        fd::finite_gradient(
            fd::flatten(V),
            [&](const Eigen::VectorXd& x) {
                return potential(collisions, mesh, fd::unflatten(x, V.cols()));
            },
            fgrad, fd::AccuracyOrder::SECOND, 1e-8);
        CAPTURE(grad.norm());
        CAPTURE(fgrad.norm());
        CHECK((grad - fgrad).norm() < 1e-4 * std::max({grad.norm(), fgrad.norm(), 1e-8}));

        Eigen::MatrixXd hess = potential.hessian(collisions, mesh, V);
        REQUIRE(hess.squaredNorm() > 1e-3);
        Eigen::MatrixXd fhess;
        fd::finite_jacobian(
            fd::flatten(V),
            [&](const Eigen::VectorXd& x) {
                return potential.gradient(collisions, mesh, fd::unflatten(x, V.cols()));
            },
            fhess, fd::AccuracyOrder::SECOND, 1e-10);
        CAPTURE(hess.norm());
        CAPTURE(fhess.norm());
        CHECK((hess - fhess).norm() < 1e-3 * std::max({hess.norm(), fhess.norm(), 1e-8}));
    };

    SECTION("Corners")
    {
        const double P0x = GENERATE(.01, -.01, 0.0);
        const double P1y = GENERATE(.49, .5, .51);
        CAPTURE(P0x, P1y);
        V.resize(8, 2);
        E.resize(8, 2);
        V <<
            -1., 1.,
            -1., 0.,
            0., 0.,
            P0x, .5 + BA,
            0., 1.,
            1., 0.,
            1., 1.,
            .02, P1y;
        E <<
            0, 1,
            1, 2,
            2, 3,
            3, 4,
            4, 0,
            5, 6,
            6, 7,
            7, 5;
        run_checks();
    }

    SECTION("squares") {
        V.resize(8, 2);
        E.resize(8, 2);
        E <<
            0, 1,
            1, 2,
            2, 3,
            3, 0,
            4, 5,
            5, 6,
            6, 7,
            7, 4;
        SECTION("horizontal_squares") {
            INFO("horizontal_squares");
            V <<
                -1., 1. + BA,
                -1., 0. + BA,
                -.1, 0. + BA,
                -.1, 1. + BA,
                .1, 1.,
                .1, 0.,
                1., 0.,
                1., 1.;
            run_checks();
        }
        SECTION("vertical_squares") {
            INFO("vertical_squares");
            V <<
                0. + BA, -1.,
                1. + BA, -1.,
                1. + BA, -.1,
                0. + BA, -.1,
                0., .1,
                1., .1,
                1., 1.,
                0., 1.;
            run_checks();
        }
    }

    SECTION("mesh_1")
    {
        INFO("mesh 1");
        std::string mesh_name = (tests::DATA_DIR / "gcp" / "nonlinear_solve_iter020.obj").string();
        bool success = igl::readCSV(mesh_name + "-v.csv", V);
        success = success && igl::readCSV(mesh_name + "-e.csv", E);
        REQUIRE(success);
        V.col(0) += Eigen::VectorXd::Random(V.rows()) * BA;
        run_checks();
    }

    SECTION("mesh_2")
    {
        INFO("mesh 2");
        std::string mesh_name = (tests::DATA_DIR / "gcp" / "simple_2d.obj").string();
        bool success = igl::readCSV(mesh_name + "-v.csv", V);
        success = success && igl::readCSV(mesh_name + "-e.csv", E);
        REQUIRE(success);
        V.col(0) += Eigen::VectorXd::Random(V.rows()) * BA;
        run_checks();
    }
}