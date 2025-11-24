#ifndef NGP_XAYAH_NGP_CUDA_BOUNDINGBOX_H
#define NGP_XAYAH_NGP_CUDA_BOUNDINGBOX_H

#include <tiny-cuda-nn/common.h>

namespace ngp::cuda {
    using namespace tcnn;

    inline TCNN_HOST_DEVICE float normdot(const vec3& a, const vec3& b) {
        float div = length(a) * length(b);
        if (div == 0.0f) {
            return 1.0f;
        }

        return dot(a, b) / div;
    }

    inline TCNN_HOST_DEVICE float angle(const vec3& a, const vec3& b) {
        return acosf(tcnn::clamp(normdot(a, b), -1.0f, 1.0f));
    }

    struct Triangle {
        TCNN_HOST_DEVICE vec3 sample_uniform_position(const vec2& sample) const {
            float sqrt_x  = sqrt(sample.x);
            float factor0 = 1.0f - sqrt_x;
            float factor1 = sqrt_x * (1.0f - sample.y);
            float factor2 = sqrt_x * sample.y;

            return factor0 * a + factor1 * b + factor2 * c;
        }

        TCNN_HOST_DEVICE float surface_area() const {
            return 0.5f * length(cross(b - a, c - a));
        }

        TCNN_HOST_DEVICE vec3 normal() const {
            return normalize(cross(b - a, c - a));
        }

        TCNN_HOST_DEVICE const vec3& operator[](uint32_t i) const {
            return i == 0 ? a : (i == 1 ? b : c);
        }

        TCNN_HOST_DEVICE float angle_at_vertex(uint32_t i) const {
            vec3 v1 = (*this)[i] - (*this)[(i + 1) % 3];
            vec3 v2 = (*this)[i] - (*this)[(i + 2) % 3];
            return angle(v1, v2);
        }

        TCNN_HOST_DEVICE uint32_t closest_vertex_idx(const vec3& pos) const {
            float mag1 = length2(pos - a);
            float mag2 = length2(pos - b);
            float mag3 = length2(pos - c);

            float minv = min(vec3{mag1, mag2, mag3});

            if (minv == mag1) {
                return 0;
            } else if (minv == mag2) {
                return 1;
            } else {
                return 2;
            }
        }

        TCNN_HOST_DEVICE float angle_at_pos(const vec3& pos) const {
            return angle_at_vertex(closest_vertex_idx(pos));
        }

        // based on https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
        TCNN_HOST_DEVICE float ray_intersect(const vec3& ro, const vec3& rd, vec3& n) const {
            vec3 v1v0 = b - a;
            vec3 v2v0 = c - a;
            vec3 rov0 = ro - a;
            n         = cross(v1v0, v2v0);
            vec3 q    = cross(rov0, rd);
            float d   = 1.0f / dot(rd, n);
            float u   = d * -dot(q, v2v0);
            float v   = d * dot(q, v1v0);
            float t   = d * -dot(n, rov0);
            if (u < 0.0f || u > 1.0f || v < 0.0f || (u + v) > 1.0f || t < 0.0f) {
                t = std::numeric_limits<float>::max();
            }
            return t;
        }

        TCNN_HOST_DEVICE float ray_intersect(const vec3& ro, const vec3& rd) const {
            vec3 n;
            return ray_intersect(ro, rd, n);
        }

        // based on https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
        TCNN_HOST_DEVICE float distance_sq(const vec3& pos) const {
            vec3 v21 = b - a;
            vec3 p1  = pos - a;
            vec3 v32 = c - b;
            vec3 p2  = pos - b;
            vec3 v13 = a - c;
            vec3 p3  = pos - c;
            vec3 nor = cross(v21, v13);

            return
                // inside/outside test
                (tcnn::sign(dot(cross(v21, nor), p1)) + tcnn::sign(dot(cross(v32, nor), p2)) + tcnn::sign(dot(cross(v13, nor), p3)) <
                 2.0f)
                    ?
                    // 3 edges
                    min(vec3{
                        tcnn::length2(v21 * tcnn::clamp(dot(v21, p1) / length2(v21), 0.0f, 1.0f) - p1),
                        tcnn::length2(v32 * tcnn::clamp(dot(v32, p2) / length2(v32), 0.0f, 1.0f) - p2),
                        tcnn::length2(v13 * tcnn::clamp(dot(v13, p3) / length2(v13), 0.0f, 1.0f) - p3),
                    })
                    :
                    // 1 face
                    dot(nor, p1) * dot(nor, p1) / length2(nor);
        }

        TCNN_HOST_DEVICE float distance(const vec3& pos) const {
            return sqrt(distance_sq(pos));
        }

        TCNN_HOST_DEVICE bool point_in_triangle(const vec3& p) const {
            // Move the triangle so that the point becomes the
            // triangles origin
            vec3 local_a = a - p;
            vec3 local_b = b - p;
            vec3 local_c = c - p;

            // The point should be moved too, so they are both
            // relative, but because we don't use p in the
            // equation anymore, we don't need it!
            // p -= p;

            // Compute the normal vectors for triangles:
            // u = normal of PBC
            // v = normal of PCA
            // w = normal of PAB

            vec3 u = cross(local_b, local_c);
            vec3 v = cross(local_c, local_a);
            vec3 w = cross(local_a, local_b);

            // Test to see if the normals are facing the same direction.
            // If yes, the point is inside, otherwise it isn't.
            return dot(u, v) >= 0.0f && dot(u, w) >= 0.0f;
        }

        TCNN_HOST_DEVICE vec3 closest_point_to_line(const vec3& a, const vec3& b, const vec3& c) const {
            float t = dot(c - a, b - a) / dot(b - a, b - a);
            t       = max(min(t, 1.0f), 0.0f);
            return a + t * (b - a);
        }

        TCNN_HOST_DEVICE vec3 closest_point(vec3 point) const {
            point -= dot(normal(), point - a) * normal();

            if (point_in_triangle(point)) {
                return point;
            }

            vec3 c1 = closest_point_to_line(a, b, point);
            vec3 c2 = closest_point_to_line(b, c, point);
            vec3 c3 = closest_point_to_line(c, a, point);

            float mag1 = length2(point - c1);
            float mag2 = length2(point - c2);
            float mag3 = length2(point - c3);

            float min = tcnn::min(vec3{mag1, mag2, mag3});

            if (min == mag1) {
                return c1;
            } else if (min == mag2) {
                return c2;
            } else {
                return c3;
            }
        }

        TCNN_HOST_DEVICE vec3 centroid() const {
            return (a + b + c) / 3.0f;
        }

        TCNN_HOST_DEVICE float centroid(int axis) const {
            return (a[axis] + b[axis] + c[axis]) / 3;
        }

        TCNN_HOST_DEVICE void get_vertices(vec3 v[3]) const {
            v[0] = a;
            v[1] = b;
            v[2] = c;
        }

        vec3 a, b, c;
    };

    template <int N_POINTS>
    TCNN_HOST_DEVICE inline void project(vec3 points[N_POINTS], const vec3& axis, float& min, float& max) {
        min = std::numeric_limits<float>::infinity();
        max = -std::numeric_limits<float>::infinity();

        TCNN_PRAGMA_UNROLL
        for (uint32_t i = 0; i < N_POINTS; ++i) {
            float val = dot(axis, points[i]);

            if (val < min) {
                min = val;
            }

            if (val > max) {
                max = val;
            }
        }
    }

    struct BoundingBox {
        TCNN_HOST_DEVICE BoundingBox() {
        }

        TCNN_HOST_DEVICE BoundingBox(const vec3& a, const vec3& b) : min{a}, max{b} {
        }

        TCNN_HOST_DEVICE explicit BoundingBox(const Triangle& tri) {
            min = max = tri.a;
            enlarge(tri.b);
            enlarge(tri.c);
        }

        TCNN_HOST_DEVICE BoundingBox(Triangle* begin, Triangle* end) {
            min = max = begin->a;
            for (auto it = begin; it != end; ++it) {
                enlarge(*it);
            }
        }

        TCNN_HOST_DEVICE void enlarge(const BoundingBox& other) {
            min = tcnn::min(min, other.min);
            max = tcnn::max(max, other.max);
        }

        TCNN_HOST_DEVICE void enlarge(const Triangle& tri) {
            enlarge(tri.a);
            enlarge(tri.b);
            enlarge(tri.c);
        }

        TCNN_HOST_DEVICE void enlarge(const vec3& point) {
            min = tcnn::min(min, point);
            max = tcnn::max(max, point);
        }

        TCNN_HOST_DEVICE void inflate(float amount) {
            min -= vec3(amount);
            max += vec3(amount);
        }

        TCNN_HOST_DEVICE vec3 diag() const {
            return max - min;
        }

        TCNN_HOST_DEVICE vec3 relative_pos(const vec3& pos) const {
            return (pos - min) / diag();
        }

        TCNN_HOST_DEVICE vec3 center() const {
            return 0.5f * (max + min);
        }

        TCNN_HOST_DEVICE BoundingBox intersection(const BoundingBox& other) const {
            BoundingBox result = *this;
            result.min         = tcnn::max(result.min, other.min);
            result.max         = tcnn::min(result.max, other.max);
            return result;
        }

        TCNN_HOST_DEVICE bool intersects(const BoundingBox& other) const {
            return !intersection(other).is_empty();
        }

        // Based on the separating axis theorem
        // (https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox_tam.pdf)
        // Code adapted from a C# implementation at stack overflow
        // https://stackoverflow.com/a/17503268
        TCNN_HOST_DEVICE bool intersects(const Triangle& triangle) const {
            float triangle_min, triangle_max;
            float box_min, box_max;

            // Test the box normals (x-, y- and z-axes)
            vec3 box_normals[3] = {
                vec3{1.0f, 0.0f, 0.0f},
                vec3{0.0f, 1.0f, 0.0f},
                vec3{0.0f, 0.0f, 1.0f},
            };

            vec3 triangle_normal = triangle.normal();
            vec3 triangle_verts[3];
            triangle.get_vertices(triangle_verts);

            for (int i = 0; i < 3; i++) {
                project<3>(triangle_verts, box_normals[i], triangle_min, triangle_max);
                if (triangle_max < min[i] || triangle_min > max[i]) {
                    return false; // No intersection possible.
                }
            }

            vec3 verts[8];
            get_vertices(verts);

            // Test the triangle normal
            float triangle_offset = dot(triangle_normal, triangle.a);
            project<8>(verts, triangle_normal, box_min, box_max);
            if (box_max < triangle_offset || box_min > triangle_offset) {
                return false; // No intersection possible.
            }

            // Test the nine edge cross-products
            vec3 edges[3] = {
                triangle.a - triangle.b,
                triangle.a - triangle.c,
                triangle.b - triangle.c,
            };

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    // The box normals are the same as it's edge tangents
                    vec3 axis = cross(edges[i], box_normals[j]);
                    project<8>(verts, axis, box_min, box_max);
                    project<3>(triangle_verts, axis, triangle_min, triangle_max);
                    if (box_max < triangle_min || box_min > triangle_max) return false; // No intersection possible
                }
            }

            // No separating axis found.
            return true;
        }

        TCNN_HOST_DEVICE vec2 ray_intersect(const vec3& pos, const vec3& dir) const {
            float tmin = (min.x - pos.x) / dir.x;
            float tmax = (max.x - pos.x) / dir.x;

            if (tmin > tmax) {
                tcnn::host_device_swap(tmin, tmax);
            }

            float tymin = (min.y - pos.y) / dir.y;
            float tymax = (max.y - pos.y) / dir.y;

            if (tymin > tymax) {
                tcnn::host_device_swap(tymin, tymax);
            }

            if (tmin > tymax || tymin > tmax) {
                return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
            }

            if (tymin > tmin) {
                tmin = tymin;
            }

            if (tymax < tmax) {
                tmax = tymax;
            }

            float tzmin = (min.z - pos.z) / dir.z;
            float tzmax = (max.z - pos.z) / dir.z;

            if (tzmin > tzmax) {
                tcnn::host_device_swap(tzmin, tzmax);
            }

            if (tmin > tzmax || tzmin > tmax) {
                return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
            }

            if (tzmin > tmin) {
                tmin = tzmin;
            }

            if (tzmax < tmax) {
                tmax = tzmax;
            }

            return {tmin, tmax};
        }

        TCNN_HOST_DEVICE bool is_empty() const {
            return max.x < min.x || max.y < min.y || max.z < min.z;
        }

        TCNN_HOST_DEVICE bool contains(const vec3& p) const {
            return p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y && p.z >= min.z && p.z <= max.z;
        }

        /// Calculate the squared point-AABB distance
        TCNN_HOST_DEVICE float distance(const vec3& p) const {
            return sqrt(distance_sq(p));
        }

        TCNN_HOST_DEVICE float distance_sq(const vec3& p) const {
            return length2(tcnn::max(tcnn::max(min - p, p - max), vec3(0.0f)));
        }

        TCNN_HOST_DEVICE float signed_distance(const vec3& p) const {
            vec3 q = abs(p - min) - diag();
            return length(tcnn::max(q, vec3(0.0f))) + std::min(tcnn::max(q), 0.0f);
        }

        TCNN_HOST_DEVICE void get_vertices(vec3 v[8]) const {
            v[0] = {min.x, min.y, min.z};
            v[1] = {min.x, min.y, max.z};
            v[2] = {min.x, max.y, min.z};
            v[3] = {min.x, max.y, max.z};
            v[4] = {max.x, min.y, min.z};
            v[5] = {max.x, min.y, max.z};
            v[6] = {max.x, max.y, min.z};
            v[7] = {max.x, max.y, max.z};
        }

        vec3 min = vec3(std::numeric_limits<float>::infinity());
        vec3 max = vec3(-std::numeric_limits<float>::infinity());
    };

}

#endif //NGP_XAYAH_NGP_CUDA_BOUNDINGBOX_H
