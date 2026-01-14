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

using namespace ipc;

TEST_CASE("Good Quadrature Gradient", "[high_order_potential]")
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F, E;
    igl::read_triangle_mesh((tests::DATA_DIR / "../src/tests/potential/wrapped_sphere.obj").string(), V, F);

    igl::edges(F, E);
    CollisionMesh mesh(V, E, F);

    const double dhat = 0.2;
    QuadraturePotential potential(mesh, V, dhat);

    for (int vid = 0; vid < V.rows(); ++vid) {
        double x = potential.point_potential->evaluate_potential_at_vertex(V, vid);
        Eigen::VectorXd g = potential.point_potential->evaluate_potential_gradient_at_vertex(V, vid);

        if (abs(x) < 1e-12) {
            continue;
        }

        Eigen::VectorXd fg;
        fd::finite_gradient(
            fd::flatten(V), [&](const Eigen::VectorXd& y) {
                return potential.point_potential->evaluate_potential_at_vertex(fd::unflatten(y, 3), vid);
            }, fg, fd::AccuracyOrder::SECOND, 1e-8);

        // std::cout << (g - fg).norm() << " " << g.norm() << std::endl;
        REQUIRE((g - fg).norm() < 1e-6 * std::max({g.norm(), fg.norm(), 1e-8}));
    }

    for (int fid = 0; fid < F.rows(); ++fid) {
        double x = potential.point_potential->evaluate_potential_at_face_center(V, fid);
        Eigen::VectorXd g = potential.point_potential->evaluate_potential_gradient_at_face_center(V, fid);

        if (abs(x) < 1e-12) {
            continue;
        }

        Eigen::VectorXd fg;
        fd::finite_gradient(
            fd::flatten(V), [&](const Eigen::VectorXd& y) {
                return potential.point_potential->evaluate_potential_at_face_center(fd::unflatten(y, 3), fid);
            }, fg, fd::AccuracyOrder::SECOND, 1e-8);

        // std::cout << (g - fg).norm() << " " << g.norm() << std::endl;
        REQUIRE((g - fg).norm() < 1e-6 * std::max({g.norm(), fg.norm(), 1e-8}));
    }

    for (const auto &ee : potential.point_potential->collisions.m_candidates.ee_candidates) {
        auto dtype = edge_edge_distance_type(
            V.row(mesh.edges()(ee.edge0_id, 0)),
            V.row(mesh.edges()(ee.edge0_id, 1)),
            V.row(mesh.edges()(ee.edge1_id, 0)),
            V.row(mesh.edges()(ee.edge1_id, 1)));
        if (dtype != EdgeEdgeDistanceType::EA_EB)
            continue;

        double mollifier = Math<double>::cubic_spline(sqrt(edge_edge_distance(
            V.row(mesh.edges()(ee.edge0_id, 0)),
            V.row(mesh.edges()(ee.edge0_id, 1)),
            V.row(mesh.edges()(ee.edge1_id, 0)),
            V.row(mesh.edges()(ee.edge1_id, 1)), dtype)) / dhat) * 1.5;

        double x = potential.point_potential->evaluate_potential_at_edge_edge_closest_point(
            V, ee.edge0_id, ee.edge1_id) * mollifier;

        Eigen::MatrixXd g = potential.point_potential->evaluate_potential_gradient_at_edge_edge_closest_point(
            V, ee.edge0_id, ee.edge1_id) * mollifier;

        if (abs(x) < 1e-12) {
            continue;
        }

        Eigen::VectorXd fg;
        fd::finite_gradient(
            fd::flatten(V), [&](const Eigen::VectorXd& y) {
                return potential.point_potential->evaluate_potential_at_edge_edge_closest_point(
                    fd::unflatten(y, 3), ee.edge0_id, ee.edge1_id);
            }, fg, fd::AccuracyOrder::SECOND, 1e-8);
        fg *= mollifier;

        // std::cout << (g - fg).norm() << " " << g.norm() << std::endl;
        REQUIRE((g - fg).norm() < 2e-6 * std::max({g.norm(), fg.norm(), 1e-8}));
    }

    for (int face_id = 0; face_id < F.rows(); face_id++) {
        double x = potential.evaluate_per_face(V, face_id);
        Eigen::VectorXd g = potential.evaluate_per_face_gradient(V, face_id);

        if (abs(x) < 1e-12) {
            continue;
        }

        Eigen::VectorXd fg;
        fd::finite_gradient(
            fd::flatten(V), [&](const Eigen::VectorXd& y) {
                return potential.evaluate_per_face(fd::unflatten(y, 3), face_id);
            }, fg, fd::AccuracyOrder::SECOND, 1e-7);

        std::cout << (g - fg).norm() << " " << g.norm() << std::endl;
        REQUIRE((g - fg).norm() < 1e-6 * std::max({g.norm(), fg.norm(), 1e-8}));
    }
}

TEST_CASE("Good Quadrature", "[high_order_potential]")
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F, E;
    igl::read_triangle_mesh((tests::DATA_DIR / "../src/tests/potential/sphere.obj").string(), V, F);

    igl::edges(F, E);
    CollisionMesh mesh(V, E, F);

    const double dhat = 0.2;
    QuadraturePotential potential(mesh, V, dhat);

    for (int vid = 0; vid < V.rows(); ++vid) {
        double x = potential.point_potential->evaluate_potential_at_vertex(V, vid);
        REQUIRE(abs(x) < 1e-12);

        Eigen::MatrixXd g = potential.point_potential->evaluate_potential_gradient_at_vertex(V, vid);
        REQUIRE(g.norm() < 1e-8);
    }

    for (int fid = 0; fid < F.rows(); ++fid) {
        double x = potential.point_potential->evaluate_potential_at_face_center(V, fid);
        REQUIRE(abs(x) < 1e-12);

        Eigen::MatrixXd g = potential.point_potential->evaluate_potential_gradient_at_face_center(V, fid);
        REQUIRE(g.norm() < 1e-8);
    }

    for (const auto &ee : potential.point_potential->collisions.m_candidates.ee_candidates) {
        auto dtype = edge_edge_distance_type(
            V.row(mesh.edges()(ee.edge0_id, 0)),
            V.row(mesh.edges()(ee.edge0_id, 1)),
            V.row(mesh.edges()(ee.edge1_id, 0)),
            V.row(mesh.edges()(ee.edge1_id, 1)));
        if (dtype != EdgeEdgeDistanceType::EA_EB)
            continue;

        double mollifier = Math<double>::cubic_spline(sqrt(edge_edge_distance(
            V.row(mesh.edges()(ee.edge0_id, 0)),
            V.row(mesh.edges()(ee.edge0_id, 1)),
            V.row(mesh.edges()(ee.edge1_id, 0)),
            V.row(mesh.edges()(ee.edge1_id, 1)), dtype)) / dhat) * 1.5;

        double x = potential.point_potential->evaluate_potential_at_edge_edge_closest_point(
            V, ee.edge0_id, ee.edge1_id) * mollifier;

        REQUIRE(abs(x) < 1e-12);

        Eigen::MatrixXd g = potential.point_potential->evaluate_potential_gradient_at_edge_edge_closest_point(
            V, ee.edge0_id, ee.edge1_id) * mollifier;

        REQUIRE(g.norm() < 1e-8);
    }

    for (int face_id = 0; face_id < F.rows(); face_id++) {
        double x = potential.evaluate_per_face(V, face_id);
        Eigen::SparseMatrix<double> g = potential.evaluate_per_face_gradient(V, face_id);

        REQUIRE(abs(x) < 1e-12);
        REQUIRE(g.norm() < 1e-8);
    }
}

TEST_CASE("Flat Integrated Potential", "[high_order_potential]")
{
    const auto method = make_default_broad_phase();
    const bool adaptive_dhat = false;
    const bool all_vertices_on_surface = true;
    const double dhat = 0.1;

    Eigen::MatrixXd vertices;
    Eigen::MatrixXi edges, faces;

    const double grid_scale = 5;
    const int N = 6;
    const double grid_h = grid_scale / (N - 1);
    // construct mesh
    {
        // regular grid Z = 0

        vertices.setZero(N * N, 3);
        faces.setZero((N - 1) * (N - 1) * 2, 3);
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                vertices(i * N + j, 0) = i * grid_h;
                vertices(i * N + j, 1) = j * grid_h;
            }
        }

        auto vid_2d_to_1d = [](int i, int j) { return i * N + j; };

        for (int i = 0; i < N - 1; i++)
        {
            for (int j = 0; j < N - 1; j++)
            {
                faces.row((i * (N - 1) + j) * 2 + 0) <<
                    vid_2d_to_1d(i, j), vid_2d_to_1d(i + 1, j), vid_2d_to_1d(i + 1, j + 1);
                faces.row((i * (N - 1) + j) * 2 + 1) <<
                    vid_2d_to_1d(i + 1, j + 1), vid_2d_to_1d(i, j + 1), vid_2d_to_1d(i, j);
            }
        }

        // plus a small tet above the grid
        // Eigen::MatrixXd vertices_tet(4, 3);
        // vertices_tet <<
        //     0.0, 0.0, 0.0,
        //     grid_h * 0.5, 0.0, 0.0,
        //     0.0, grid_h * 0.5, 0.0,
        //     0.0, 0.0, 2 * dhat;
        //
        // vertices_tet.rowwise() += Eigen::Vector3d(grid_scale / 2., grid_scale / 2., dhat / 5.).transpose();
        //
        // Eigen::MatrixXi faces_tet(4, 3);
        // faces_tet << 0, 1, 2,
        //             0, 1, 3,
        //             0, 2, 3,
        //             1, 2, 3;

        Eigen::MatrixXd vertices_tet(1, 3);
        vertices_tet << grid_scale / 2., grid_scale / 2., dhat / 5.;

        // merge both meshes
        Eigen::MatrixXd merged_vertices(vertices.rows() + vertices_tet.rows(), 3);
        merged_vertices << vertices, vertices_tet;

        // Eigen::MatrixXi merged_faces(faces.rows() + faces_tet.rows(), 3);
        // merged_faces << faces, faces_tet.array() + vertices.rows();

        std::swap(merged_vertices, vertices);
        // std::swap(merged_faces, faces);

        // extract edges
        igl::edges(faces, edges);
    }

    CollisionMesh mesh;

    if (all_vertices_on_surface) {
        mesh = CollisionMesh(
            std::vector<bool>(vertices.rows(), true),
            std::vector<bool>(vertices.rows(), false), vertices, edges,
            faces);
    } else {
        mesh = CollisionMesh(
            ipc::CollisionMesh::construct_is_on_surface(vertices.rows(), edges),
            std::vector<bool>(vertices.rows(), false), vertices, edges,
            faces);

        vertices = mesh.vertices(vertices);
    }

    HighOrderContactParameters params(dhat, 0., 2, 0);

    HighOrderCollisions collisions;
    collisions.build(mesh, vertices, params, adaptive_dhat, method);

    CHECK(!collisions.empty());
    CHECK(!has_intersections(mesh, vertices));

    std::cout << collisions.to_string(mesh, vertices, params) << std::endl;

    HighOrderContactPotential potential(params);
    std::cout << "energy: " << potential(collisions, mesh, vertices) << "\n";

    const double h = grid_h / 10;
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            Eigen::MatrixXd vertices_copy = vertices;
            vertices_copy.bottomRows<4>().rowwise() += Eigen::Vector3d(i * h, j * h, 0.).transpose();

            HighOrderCollisions collisions_copy;
            collisions_copy.build(mesh, vertices_copy, params, adaptive_dhat, method);
            std::cout << "energy: " << potential(collisions_copy, mesh, vertices_copy) << "\n";
            if (potential(collisions_copy, mesh, vertices_copy) < 1e-3)
            {
                collisions_copy.build(mesh, vertices_copy, params, adaptive_dhat, method);
                std::cout << collisions_copy.to_string(mesh, vertices_copy, params) << std::endl;
            }

            igl::write_triangle_mesh("debug_" + std::to_string(i) + "_" + std::to_string(j) + ".obj", vertices_copy, faces);
        }
    }
}

TEST_CASE("Zero Potential on Sphere", "[high_order_potential]")
{
    const auto method = make_default_broad_phase();
    const bool adaptive_dhat = false;
    const bool all_vertices_on_surface = true;
    const double dhat = 0.3;

    Eigen::MatrixXd vertices;
    Eigen::MatrixXi edges, faces;

    igl::read_triangle_mesh((tests::DATA_DIR / "../src/tests/potential/sphere.obj").string(), vertices, faces);

    // extract edges
    igl::edges(faces, edges);

    CollisionMesh mesh;

    if (all_vertices_on_surface) {
        mesh = CollisionMesh(
            std::vector<bool>(vertices.rows(), true),
            std::vector<bool>(vertices.rows(), false), vertices, edges,
            faces);
    } else {
        mesh = CollisionMesh(
            ipc::CollisionMesh::construct_is_on_surface(vertices.rows(), edges),
            std::vector<bool>(vertices.rows(), false), vertices, edges,
            faces);

        vertices = mesh.vertices(vertices);
    }

    HighOrderContactParameters params(dhat, 0., 2, 0);

    HighOrderCollisions collisions;
    collisions.build(mesh, vertices, params, adaptive_dhat, method);

    // CHECK(!collisions.empty());
    // CHECK(!has_intersections(mesh, vertices));

    HighOrderContactPotential potential(params);
    std::cout << "triple collisions: " << collisions.triple_collisions.size() << ", pair collisions: " << collisions.collisions.size() << std::endl;
    std::cout << "energy: " << potential(collisions, mesh, vertices) << "\n";
    std::cout << collisions.to_string(mesh, vertices, params) << std::endl;
    REQUIRE(abs(potential(collisions, mesh, vertices)) < 1e-8);
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
    double dhat)
{
    const bool adaptive_dhat = false;
    const bool orientable = false;
    const auto method = make_default_broad_phase();
    const double min_dist_ratio = 1.5;
    Eigen::MatrixXi faces;

    CollisionMesh mesh;
    HighOrderContactParameters params(dhat, 0.1, 1, 0);
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
    CHECK(energy > 0);

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
    CHECK((grad_b - fgrad_b).norm() < 1e-6 * grad_b.norm());
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

    REQUIRE(hess_b.squaredNorm() > 1e-3);
    std::cout << "hess relative error "
              << (hess_b - fhess_b).norm() / hess_b.norm() << "\n";
    CHECK((hess_b - fhess_b).norm() < 1e-6 * hess_b.norm());
    // CHECK(fd::compare_hessian(hess_b, fhess_b, 1e-3));
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

    test_high_order_potential(vertices, edges, dhat);
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
