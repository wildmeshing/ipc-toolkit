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

    REQUIRE((fh.col(0) - h * test_dir).norm() < fh.norm() * 1e-6);
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
    REQUIRE(abs(val) < 1e-12);

    auto g = potential.gradient(collisions, mesh, V);
    REQUIRE(g.norm() < 1e-8);
}

TEST_CASE("Convergent Quadrature Vertex Hessian", "[high_order_potential]")
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F, E;
    igl::read_triangle_mesh((tests::DATA_DIR / "../src/tests/potential/wrapped_sphere.obj").string(), V, F);

    igl::edges(F, E);
    CollisionMesh mesh(V, E, F);

    const double dhat = 0.15;
    HighOrderContactParameters params(dhat, 0., 2, 0);

    Candidates candidates;
    candidates.build(mesh, V, dhat / 2, make_default_broad_phase(), true);
    candidates.convert_candidates_to_sets();
    PointPotential point_potential(mesh, candidates, params);

    for (int vid = 0; vid < V.rows(); ++vid) {
        const auto collisions = point_potential.build_collisions_at_vertex(V, vid);

        if (collisions.size() == 0) {
            continue;
        }

        std::vector<int> indices;
        {
            Eigen::SparseMatrix<double> g_sparse =
                PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(
                    V, collisions, params);
            for (index_t k = 0; k < g_sparse.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(g_sparse, k); it; ++it) {
                    assert(it.col() == 0);
                    indices.push_back(it.row());
                }
            }

            if (g_sparse.norm() < 1e-10) {
                continue;
            }
        }

        Eigen::MatrixXd h = PointPotentialHelper::evaluate_potential_hessian_at_vertex_with_cached_collisions(
            V, collisions, params);
        h = h(indices, indices).eval();

        Eigen::MatrixXd fh;
        fd::finite_jacobian(
            fd::flatten(V)(indices), [&](const Eigen::VectorXd& y) {
                Eigen::VectorXd y_ = fd::flatten(V);
                y_(indices) = y;
                Eigen::MatrixXd V_fd = fd::unflatten(y_, 3);

                Eigen::VectorXd g = Eigen::MatrixXd(PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(
                    V_fd, collisions, params)).col(0);
                return g(indices);
            }, fh, fd::AccuracyOrder::SECOND, 1e-8);

        REQUIRE((h - fh).norm() < 1e-6 * std::max({h.norm(), fh.norm(), 1e-8}));
    }
}

TEST_CASE("Convergent Quadrature Face Hessian", "[high_order_potential]")
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F, E;
    igl::read_triangle_mesh((tests::DATA_DIR / "../src/tests/potential/wrapped_sphere.obj").string(), V, F);

    igl::edges(F, E);
    CollisionMesh mesh(V, E, F);

    const double dhat = 0.15;
    HighOrderContactParameters params(dhat, 0., 2, 0);

    Candidates candidates;
    candidates.build(mesh, V, dhat / 2, make_default_broad_phase(), true);
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
            Eigen::SparseMatrix<double> g_sparse = PointPotentialHelper::evaluate_potential_gradient_at_face_center_with_cached_collisions(V_extended, vids, collisions, params);
            for (index_t k = 0; k < g_sparse.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(g_sparse, k); it; ++it) {
                    assert(it.col() == 0);
                    indices.push_back(it.row());
                }
            }

            if (g_sparse.norm() < 1e-10) {
                continue;
            }
        }

        Eigen::MatrixXd h = PointPotentialHelper::evaluate_potential_hessian_at_face_center_with_cached_collisions(V_extended, vids, collisions, params);
        h = h(indices, indices).eval();

        Eigen::MatrixXd fh;
        fd::finite_jacobian(
            fd::flatten(V)(indices), [&](const Eigen::VectorXd& y) {
                Eigen::VectorXd y_ = fd::flatten(V);
                y_(indices) = y;
                Eigen::MatrixXd V_fd = fd::unflatten(y_, 3);
                Eigen::RowVector3d face_center_fd = (V_fd.row(vids[0]) + V_fd.row(vids[1]) + V_fd.row(vids[2])) / 3.;
                ConcatMatrixView<3> V_fd_extended(V_fd, face_center_fd);

                Eigen::VectorXd g = PointPotentialHelper::evaluate_potential_gradient_at_face_center_with_cached_collisions(V_fd_extended, vids, collisions, params);
                return g(indices);
            }, fh, fd::AccuracyOrder::SECOND, 1e-8);

        REQUIRE((h - fh).norm() < 1e-6 * std::max({h.norm(), fh.norm(), 1e-8}));
    }
}

/*
TEST_CASE("High Order barrier potential codim", "[high_order_potential]")
{
    const auto method = make_default_broad_phase();
    double dhat = 2;
    std::string mesh_name;

    Eigen::MatrixXd vertices(4, 2);
    Eigen::MatrixXi edges(2, 2), faces;

    vertices << -1, 0, 0, 0, 1, 0, 1.5, 0.2;
    edges << 0, 1, 1, 2;

    CollisionMesh mesh;

    HighOrderCollisions collisions;
    mesh = CollisionMesh(
        std::vector<bool>(vertices.rows(), true),
        std::vector<bool>(vertices.rows(), false), vertices, edges, faces);
    HighOrderContactParameters params(dhat, 0.85, 0.15, 2, 4);
    collisions.build(mesh, vertices, params, false, method);
    CAPTURE(dhat, method);
    CHECK(!collisions.empty());
    CHECK(!has_intersections(mesh, vertices));

    HighOrderContactPotential potential(params);
    std::cout << "energy: " << potential(collisions, mesh, vertices) << "\n";

    // -------------------------------------------------------------------------
    // Minimum distance
    // -------------------------------------------------------------------------

    CHECK(
        collisions.compute_minimum_distance(mesh, vertices)
        <= collisions.compute_active_minimum_distance(mesh, vertices)
            * (1. + 1e-15));

    // -------------------------------------------------------------------------
    // Gradient
    // -------------------------------------------------------------------------

    const Eigen::VectorXd grad_b =
        potential.gradient(collisions, mesh, vertices);

    // Compute the gradient using finite differences
    Eigen::VectorXd fgrad_b;
    {
        auto f = [&](const Eigen::VectorXd& x) {
            return potential(
                collisions, mesh, fd::unflatten(x, vertices.cols()));
        };
        fd::finite_gradient(
            fd::flatten(vertices), f, fgrad_b, fd::AccuracyOrder::SECOND, 1e-8);
    }

    // REQUIRE(grad_b.squaredNorm() > 1e-8);
    std::cout << "grad relative error "
              << (grad_b - fgrad_b).norm() / grad_b.norm() << ", norms "
              << grad_b.norm() << " " << fgrad_b.norm() << "\n";
    CHECK((grad_b - fgrad_b).norm() / grad_b.norm() < 1e-5);

    // -------------------------------------------------------------------------
    // Hessian
    // -------------------------------------------------------------------------

    Eigen::MatrixXd hess_b = potential.hessian(collisions, mesh, vertices);

    // Compute the gradient using finite differences
    Eigen::MatrixXd fhess_b;
    {
        auto f = [&](const Eigen::VectorXd& x) {
            return potential.gradient(
                collisions, mesh, fd::unflatten(x, vertices.cols()));
        };
        fd::finite_jacobian(
            fd::flatten(vertices), f, fhess_b, fd::AccuracyOrder::SECOND, 1e-8);
    }

    REQUIRE(hess_b.squaredNorm() > 1e-8);
    std::cout << "hess relative error "
              << (hess_b - fhess_b).norm() / hess_b.norm() << ", norms "
              << hess_b.norm() << " " << fhess_b.norm() << "\n";
    CHECK((hess_b - fhess_b).norm() / hess_b.norm() < 1e-5);
}

#if defined(NDEBUG) && !defined(WIN32)
std::string tagsopt_ho = "[high_order_potential]";
#else
std::string tagsopt_ho = "[.][high_order_potential]";
#endif

TEST_CASE("High Order barrier potential full gradient and hessian 3D", tagsopt_ho)
{
    const auto method = make_default_broad_phase();
    const bool adaptive_dhat = GENERATE(true, false);
    const bool orientable = GENERATE(true, false);
    double dhat = -1;
    std::string mesh_name;
    bool all_vertices_on_surface = true;

    SECTION("two cubes far")
    {
        dhat = 1;
        mesh_name = "two-cubes-far.ply";
        all_vertices_on_surface = false;
    }
    SECTION("two cubes close")
    {
        dhat = 1e-1;
        mesh_name = "two-cubes-close.ply";
        all_vertices_on_surface = false;
    }

    double min_dist_ratio = 1.5;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi edges, faces;
    bool success = tests::load_mesh(mesh_name, vertices, edges, faces);
    vertices +=
        Eigen::MatrixXd::Random(vertices.rows(), vertices.cols()) * 1e-3;
    CAPTURE(mesh_name);
    REQUIRE(success);

    CollisionMesh mesh;

    HighOrderCollisions collisions;
    if (all_vertices_on_surface) {
        mesh = CollisionMesh(
            std::vector<bool>(vertices.rows(), true),
            std::vector<bool>(vertices.rows(), orientable), vertices, edges,
            faces);
    } else {
        mesh = CollisionMesh(
            ipc::CollisionMesh::construct_is_on_surface(vertices.rows(), edges),
            std::vector<bool>(vertices.rows(), orientable), vertices, edges,
            faces);

        vertices = mesh.vertices(vertices);
    }

    HighOrderContactParameters params(dhat, 0.85, 0.15, 2, 4);
    params.set_adaptive_dhat_ratio(min_dist_ratio);
    collisions.compute_adaptive_dhat(mesh, vertices, params, method);
    collisions.build(mesh, vertices, params, adaptive_dhat, method);
    CAPTURE(dhat, method, adaptive_dhat, all_vertices_on_surface);
    CHECK(!collisions.empty());
    CHECK(!has_intersections(mesh, vertices));

    HighOrderContactPotential potential(params);
    std::cout << "energy: " << potential(collisions, mesh, vertices) << "\n";

    // -------------------------------------------------------------------------
    // Minimum distance
    // -------------------------------------------------------------------------

    CHECK(
        collisions.compute_minimum_distance(mesh, vertices)
        <= collisions.compute_active_minimum_distance(mesh, vertices)
            * (1. + 1e-15));

    // -------------------------------------------------------------------------
    // Gradient
    // -------------------------------------------------------------------------

    const Eigen::VectorXd grad_b =
        potential.gradient(collisions, mesh, vertices);

    // Compute the gradient using finite differences
    Eigen::VectorXd fgrad_b;
    {
        auto f = [&](const Eigen::VectorXd& x) {
            return potential(
                collisions, mesh, fd::unflatten(x, vertices.cols()));
        };
        fd::finite_gradient(
            fd::flatten(vertices), f, fgrad_b, fd::AccuracyOrder::SECOND, 1e-8);
    }

    // REQUIRE(grad_b.squaredNorm() > 1e-8);
    std::cout << "grad relative error "
              << (grad_b - fgrad_b).norm() / grad_b.norm() << ", norms "
              << grad_b.norm() << " " << fgrad_b.norm() << "\n";
    CHECK((grad_b - fgrad_b).norm() / grad_b.norm() < 1e-5);

    // -------------------------------------------------------------------------
    // Hessian
    // -------------------------------------------------------------------------

    Eigen::MatrixXd hess_b = potential.hessian(collisions, mesh, vertices);

    // Compute the gradient using finite differences
    Eigen::MatrixXd fhess_b;
    {
        auto f = [&](const Eigen::VectorXd& x) {
            return potential.gradient(
                collisions, mesh, fd::unflatten(x, vertices.cols()));
        };
        fd::finite_jacobian(
            fd::flatten(vertices), f, fhess_b, fd::AccuracyOrder::SECOND, 1e-8);
    }

    REQUIRE(hess_b.squaredNorm() > 1e-8);
    std::cout << "hess relative error "
              << (hess_b - fhess_b).norm() / hess_b.norm() << ", norms "
              << hess_b.norm() << " " << fhess_b.norm() << "\n";
    CHECK((hess_b - fhess_b).norm() / hess_b.norm() < 1e-5);
}
*/

void test_high_order_potential(
    Eigen::MatrixXd& vertices,
    Eigen::MatrixXi& edges,
    double dhat,
    bool shouldbe0 = false)
{
    const bool adaptive_dhat = false;
    const bool orientable = false;
    const auto method = make_default_broad_phase();
    const double min_dist_ratio = 1.5;
    Eigen::MatrixXi faces;

    CollisionMesh mesh;
    HighOrderContactParameters params(dhat, 0.1, 1, 2);
    params.set_adaptive_dhat_ratio(min_dist_ratio);
    HighOrderCollisions collisions;
    mesh = CollisionMesh(
        std::vector<bool>(vertices.rows(), true),
        std::vector<bool>(vertices.rows(), orientable), vertices, edges, faces);
    collisions.compute_adaptive_dhat(mesh, vertices, params, method);
    collisions.build(mesh, vertices, params, adaptive_dhat, method);
    CAPTURE(dhat, method, adaptive_dhat);
    CHECK(!collisions.empty());
    /*
    std::cout << "high order collision candidate size " << collisions.size()
        << "\n";
    for (const auto& c : collisions.collisions) {
        std::cout << "  - Collision type: " << c->name() << ", primitives: ("
                  << (*c)[0] << ", " << (*c)[1] << ")\n";
    }
    */
    CHECK(!has_intersections(mesh, vertices));

    HighOrderContactPotential potential(params);
    const auto energy = potential(collisions, mesh, vertices);
    std::cout << "energy: " << energy << "\n";
    if (shouldbe0) CHECK(energy == 0);
    else CHECK(energy > 0);

    // -------------------------------------------------------------------------
    // Gradient
    // -------------------------------------------------------------------------

    const Eigen::VectorXd grad_b =
        potential.gradient(collisions, mesh, vertices);

    // Compute the gradient using finite differences
    Eigen::VectorXd fgrad_b;
    {
        auto f = [&](const Eigen::VectorXd& x) {
            return potential(
                collisions, mesh, fd::unflatten(x, vertices.cols()));
        };
        fd::finite_gradient(
            fd::flatten(vertices), f, fgrad_b, fd::AccuracyOrder::SECOND, 1e-8);
    }

    if (shouldbe0) REQUIRE(grad_b.squaredNorm() == 0);
    else {
        REQUIRE(grad_b.squaredNorm() > 1e-8);
        std::cout << "grad relative error "
              << (grad_b - fgrad_b).norm() / grad_b.norm() << "\n";
        CHECK((grad_b - fgrad_b).norm() < 1e-6 * grad_b.norm());
    }
    // CHECK(fd::compare_gradient(grad_b, fgrad_b));

    // -------------------------------------------------------------------------
    // Hessian
    // -------------------------------------------------------------------------

    Eigen::MatrixXd hess_b = potential.hessian(collisions, mesh, vertices);

    // Compute the gradient using finite differences
    Eigen::MatrixXd fhess_b;
    {
        auto f = [&](const Eigen::VectorXd& x) {
            return potential.gradient(
                collisions, mesh, fd::unflatten(x, vertices.cols()));
        };
        fd::finite_jacobian(
            fd::flatten(vertices), f, fhess_b, fd::AccuracyOrder::SECOND, 1e-8);
    }

    if (shouldbe0) REQUIRE(hess_b.squaredNorm() == 0);
    else {
        REQUIRE(hess_b.squaredNorm() > 1e-3);
        std::cout << "hess relative error "
              << (hess_b - fhess_b).norm() / hess_b.norm() << "\n";
        CHECK((hess_b - fhess_b).norm() < 1e-6 * hess_b.norm());
    }
    // CHECK(fd::compare_hessian(hess_b, fhess_b, 1e-3));
}

TEST_CASE("High Order barrier potential no forces at rest", "[high_order_potential]")
{
    double dhat = -1;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi edges;
    SECTION("single_square")
    {
        dhat = 2.0;
        vertices.resize(4, 2);
        edges.resize(4, 2);
        vertices <<
            0., 0.,
            1., 0.,
            1., 1.,
            0., 1.;
        edges <<
            0, 1,
            1, 2,
            2, 3,
            3, 0;
    }
    SECTION("single_square_2")
    {
        dhat = 2.0;
        vertices.resize(8, 2);
        edges.resize(8, 2);
        vertices <<
            0., 0.,
            .5, 0.,
            1., 0.,
            1., .5,
            1., 1.,
            .5, 1.,
            0., 1.,
            0., .5;
        edges <<
            0, 1,
            1, 2,
            2, 3,
            3, 4,
            4, 5,
            5, 6,
            6, 7,
            7, 0;
    }

    test_high_order_potential(vertices, edges, dhat, true);
}

TEST_CASE("High Order barrier potential real sim 2D C^2", "[high_order_potential]")
{
    double dhat = -1;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi edges;

    /*
    SECTION("simple_2_edges")
    {
        dhat = 2.0;
        vertices.resize(4, 2);
        edges.resize(2, 2);
        vertices << -100., 0.,
            200., 0.,
            1., 1.,
            0., 1.;
        edges << 0, 1,
            2, 3;
    }
    */

    /*
    SECTION("wedge")
    {
        dhat = 0.4;
        vertices.resize(8, 2);
        edges.resize(8, 2);
        vertices <<
            -1., 1.,
            -1., 0.,
            0., 0.,
            0., 1.,
            .02, .5,
            1., 0.,
            1., 1.,
            .01, .5;
        edges <<
            0, 1,
            1, 2,
            2, 3,
            3, 7,
            7, 0,
            4, 5,
            5, 6,
            6, 4;
    }
    */

    /*
    SECTION("horizontal_squares")
    {
        dhat = 0.4;
        vertices.resize(8, 2);
        edges.resize(8, 2);
        vertices <<
            -1., 1.1,
            -1., 0.1,
            -.1, 0.1,
            -.1, 1.1,
            .1, 1.,
            .1, 0.,
            1., 0.,
            1., 1.;
        edges <<
            0, 1,
            1, 2,
            2, 3,
            3, 0,
            4, 5,
            5, 6,
            6, 7,
            7, 4;
    }

    SECTION("vertical_squares")
    {
        dhat = 0.4;
        vertices.resize(8, 2);
        edges.resize(8, 2);
        vertices <<
            -1., 1.,
            -1., 0.,
            -.1, 0.,
            -.1, 1.,
            -1., -.1,
            -1., -1.,
            -.1, -1.,
            -.1, -.1;
        edges <<
            0, 1,
            1, 2,
            2, 3,
            3, 0,
            4, 5,
            5, 6,
            6, 7,
            7, 4;
    }

    SECTION("debug1")
    {
        std::string mesh_name =
            (tests::DATA_DIR / "gcp" / "nonlinear_solve_iter020.obj").string();
        dhat = 3e-2;
        bool success = igl::readCSV(mesh_name + "-v.csv", vertices);
        success = success && igl::readCSV(mesh_name + "-e.csv", edges);
        CAPTURE(mesh_name);
        REQUIRE(success);
    }
    */

    test_high_order_potential(vertices, edges, dhat, false);
}

TEST_CASE("High Order barrier potential real sim 2D C^1", "[high_order_potential]")
{
    const auto method = make_default_broad_phase();
    //const bool adaptive_dhat = GENERATE(true, false);
    const bool adaptive_dhat = false;

    double dhat = -1;
    std::string mesh_name;
    SECTION("debug2")
    {
        mesh_name = (tests::DATA_DIR / "gcp" / "simple_2d.obj").string();
        dhat = 0.1;
    }

    double min_dist_ratio = 1.5;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi edges, faces;
    bool success = igl::readCSV(mesh_name + "-v.csv", vertices);
    success = success && igl::readCSV(mesh_name + "-e.csv", edges);
    CAPTURE(mesh_name);
    REQUIRE(success);

    // std::cout << "\n" <<  vertices << "\n" << edges << "\n";

    CollisionMesh mesh;
    HighOrderContactParameters params(dhat, 0.9, 1, 4);
    params.set_adaptive_dhat_ratio(min_dist_ratio);
    HighOrderCollisions collisions;
    mesh = CollisionMesh(vertices, edges, faces);
    collisions.compute_adaptive_dhat(mesh, vertices, params, method);
    collisions.build(mesh, vertices, params, adaptive_dhat, method);
    CAPTURE(dhat, method, adaptive_dhat);
    CHECK(!collisions.empty());
    std::cout << "high order collision candidate size " << collisions.size()
              << "\n";
    //std::cout << collisions.to_string(mesh, vertices, params) << "\n";

    CHECK(!has_intersections(mesh, vertices));

    HighOrderContactPotential potential(params);
    std::cout << "energy: " << potential(collisions, mesh, vertices) << "\n";

    // -------------------------------------------------------------------------
    // Gradient
    // -------------------------------------------------------------------------

    const Eigen::VectorXd grad_b =
        potential.gradient(collisions, mesh, vertices);

    // Compute the gradient using finite differences
    Eigen::VectorXd fgrad_b;
    {
        auto f = [&](const Eigen::VectorXd& x) {
            return potential(
                collisions, mesh, fd::unflatten(x, vertices.cols()));
        };
        fd::finite_gradient(
            fd::flatten(vertices), f, fgrad_b, fd::AccuracyOrder::SECOND, 1e-8);
    }

    REQUIRE(grad_b.squaredNorm() > 1e-8);
    std::cout << "grad relative error "
              << (grad_b - fgrad_b).norm() / grad_b.norm() << "\n";
    CHECK((grad_b - fgrad_b).norm() < 1e-7 * grad_b.norm());
    // CHECK(fd::compare_gradient(grad_b, fgrad_b));
}
